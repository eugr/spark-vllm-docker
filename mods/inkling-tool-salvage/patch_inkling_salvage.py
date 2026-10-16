#!/usr/bin/env python3
"""Salvage tool calls that leak as content in Inkling's streaming path.

vLLM's streaming DelegatingParser drops the content-kind marker that starts a
*turn-initial* tool call in a multi-turn conversation, so the whole block
``<|content_invoke_tool_json|>{...}<|end_message|>`` surfaces as message content
with ``tool_calls`` empty (deterministic: multi-turn + streaming + the assistant
turn starting directly with a tool call; single-turn and non-streaming are fine).

The chat-completion server streams via ``DelegatingParser.parse_delta`` (the old
ToolParser ``extract_tool_calls_streaming`` interface is NOT called on this path),
so this mod appends a monkeypatch to ``vllm/parser/abstract_parser.py`` that wraps
``DelegatingParser.parse_delta``: it buffers streamed content, detects complete
tool-call blocks, and re-emits them as ``tool_call`` deltas. Already-parsed tool
calls and reasoning pass through untouched; markers split across deltas are held
until complete; an unterminated block at stream end is surfaced raw so nothing is
silently dropped.

Patching the shared ``DelegatingParser`` is safe for every other model: the
salvage only fires on content containing Inkling's ``<|content_invoke_tool_json|>``
marker, which no other model emits — the marker string is the guard. Wrapping
here also lets the server's ``tools_streamed`` bookkeeping see the recovered
``tool_calls`` and set ``finish_reason`` accordingly.

Idempotent; refuses to patch if ``DelegatingParser`` is absent.
Remove once vLLM fixes the streaming reasoning->tool handoff upstream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# spark-vllm mod: inkling-tool-salvage v1"
ANCHOR = "class DelegatingParser("

PAYLOAD = '''

# spark-vllm mod: inkling-tool-salvage v1
import json as _sv_json
import html as _sv_html
from vllm.entrypoints.openai.engine.protocol import (
    DeltaMessage as _SvDeltaMessage,
    DeltaToolCall as _SvDeltaToolCall,
    DeltaFunctionCall as _SvDeltaFunctionCall,
)

_SV_TOOL_START = "<|content_invoke_tool_json|>"
_SV_END_MSG = "<|end_message|>"
_SV_MSG_MODEL = "<|message_model|>"
# Framing / terminal markers that must never reach user-visible content. These
# are stripped from emitted content only; the tool-block terminator is consumed
# in the active branch before this runs, so stripping <|end_message|> here only
# removes a genuinely leaked terminal (the class of leak in vllm#49865), never a
# tool-block boundary the salvage relies on.
_SV_STRAY = (
    "<|content_text|>", "<|content_thinking|>", "<|message_model|>",
    "<|message_user|>", "<|message_system|>", "<|message_tool|>",
    "<|end_message|>", "<|content_model_end_sampling|>",
)
_SV_MAX_BUF = 16384


def _sv_partial_tail(text, marker):
    """Longest suffix of ``text`` that is a proper prefix of ``marker`` (a
    marker split across streamed deltas), so it can be held for the next tick."""
    n = min(len(text), len(marker) - 1)
    for k in range(n, 0, -1):
        if marker.startswith(text[-k:]):
            return text[-k:]
    return ""


def _sv_init(self):
    if not hasattr(self, "_sv_active"):
        self._sv_active = False
        self._sv_buf = ""
        self._sv_pending = ""
        self._sv_idx = 0


def _sv_block_to_toolcall(self, buf):
    s = _sv_html.unescape(buf.strip())
    try:
        obj = _sv_json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("args", obj.get("arguments", {}))
    if isinstance(args, str):
        args_str = args
    else:
        try:
            args_str = _sv_json.dumps(args, ensure_ascii=False)
        except Exception:
            args_str = "{}"
    idx = self._sv_idx
    self._sv_idx += 1
    return _SvDeltaToolCall(
        index=idx,
        id="call_inkling_salvage_%d" % idx,
        type="function",
        function=_SvDeltaFunctionCall(name=name, arguments=args_str),
    )


def _sv_process(self, delta):
    _sv_init(self)
    if delta is None:
        return None
    # Real tool calls already parsed: keep our index ahead and pass through.
    if getattr(delta, "tool_calls", None):
        for tc in delta.tool_calls:
            if getattr(tc, "index", None) is not None and tc.index >= self._sv_idx:
                self._sv_idx = tc.index + 1
        return delta
    content = getattr(delta, "content", None)
    if content is None:
        return delta  # reasoning/role only
    # Fast path: nothing Inkling-tool-shaped and no block in progress -> untouched.
    if not self._sv_active and _SV_TOOL_START not in content and "<|" not in content:
        return delta

    text = self._sv_pending + content
    self._sv_pending = ""
    parts = []
    tool_calls = []
    while text:
        if not self._sv_active:
            pos = text.find(_SV_TOOL_START)
            if pos == -1:
                hold = _sv_partial_tail(text, _SV_TOOL_START)
                parts.append(text[: len(text) - len(hold)] if hold else text)
                self._sv_pending = hold
                text = ""
            else:
                pre = text[:pos]
                mm = pre.rfind(_SV_MSG_MODEL)
                if mm != -1:
                    pre = pre[:mm]  # drop framing + slot-name before the block
                parts.append(pre)
                self._sv_active = True
                self._sv_buf = ""
                text = text[pos + len(_SV_TOOL_START):]
        else:
            pos = text.find(_SV_END_MSG)
            if pos == -1:
                hold = _sv_partial_tail(text, _SV_END_MSG)
                self._sv_buf += text[: len(text) - len(hold)] if hold else text
                self._sv_pending = hold
                text = ""
                if len(self._sv_buf) > _SV_MAX_BUF:
                    parts.append(_SV_TOOL_START + self._sv_buf)  # safety valve
                    self._sv_active = False
                    self._sv_buf = ""
                    self._sv_pending = ""
            else:
                self._sv_buf += text[:pos]
                tc = _sv_block_to_toolcall(self, self._sv_buf)
                if tc is not None:
                    tool_calls.append(tc)
                else:
                    parts.append(_SV_TOOL_START + self._sv_buf + _SV_END_MSG)
                self._sv_active = False
                self._sv_buf = ""
                text = text[pos + len(_SV_END_MSG):]

    out = "".join(parts)
    for s in _SV_STRAY:
        out = out.replace(s, "")
    delta.content = out if out else None
    if tool_calls:
        delta.tool_calls = list(getattr(delta, "tool_calls", None) or []) + tool_calls
    if (
        delta.content is None
        and not delta.tool_calls
        and not getattr(delta, "reasoning", None)
        and not getattr(delta, "role", None)
    ):
        return None
    return delta


def _sv_flush(self, delta):
    _sv_init(self)
    leftover = self._sv_pending
    if self._sv_active:
        leftover = _SV_TOOL_START + self._sv_buf + leftover  # surface incomplete raw
    self._sv_pending = ""
    self._sv_buf = ""
    self._sv_active = False
    if not leftover:
        return delta
    if delta is None:
        return _SvDeltaMessage(content=leftover)
    delta.content = (delta.content or "") + leftover
    return delta


if not getattr(DelegatingParser, "_sv_patched", False):
    _sv_orig_parse_delta = DelegatingParser.parse_delta

    def _sv_parse_delta(self, *a, **k):
        delta = _sv_process(self, _sv_orig_parse_delta(self, *a, **k))
        if k.get("finished", False):
            delta = _sv_flush(self, delta)
        return delta

    DelegatingParser.parse_delta = _sv_parse_delta
    DelegatingParser._sv_patched = True
'''


def check(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        print("[inkling-tool-salvage] NOT patched", file=sys.stderr)
        return 1
    if "_sv_parse_delta" not in text:
        print("[inkling-tool-salvage] marker present but payload missing", file=sys.stderr)
        return 1
    print("[inkling-tool-salvage] verified patched")
    return 0


def apply(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print("[inkling-tool-salvage] already patched")
        return 0
    if ANCHOR not in text:
        print(
            "[inkling-tool-salvage] DelegatingParser class not found — vLLM parser "
            "changed; refusing to patch.",
            file=sys.stderr,
        )
        return 2
    path.write_text(text.rstrip() + "\n" + PAYLOAD, encoding="utf-8")
    print("[inkling-tool-salvage] patched", path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.target.is_file():
        print(f"[inkling-tool-salvage] target not found: {args.target}", file=sys.stderr)
        return 2
    return check(args.target) if args.check else apply(args.target)


if __name__ == "__main__":
    raise SystemExit(main())
