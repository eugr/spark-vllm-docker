#!/usr/bin/env python3
"""Issue #55 mitigation (recipe-side, no upstream vLLM change).

When ``max_tokens`` truncates a request mid-tool-call (engine finish_reason =
``FinishReason.LENGTH``), stock vLLM reports ``finish_reason="tool_calls"``
even though the call never completed, and the streaming flush path can hand
the client a ``tool_calls[].function.arguments`` that does **not** parse as
JSON. A harness that replays the broken call then sees the whole transcript
rejected with HTTP 400 ("Unterminated string …").

This hotfix makes two monotonically-safer edits to the chat-completions
serving path (both in
``vllm/entrypoints/openai/chat_completion/serving.py``):

1. **Streaming final chunk** — if the engine truncated for ``length``,
   report ``finish_reason="length"`` (not ``"tool_calls"``), and strip any
   non-JSON-parseable ``tool_calls`` from the trailing ``delta_message`` so
   invalid args are not flushed to the client. Natural-stop tool calls keep
   ``finish_reason="tool_calls"``.
2. **Non-streaming** — if the engine truncated for ``length``, do *not*
   claim ``finish_reason="tool_calls"`` and drop any unparseable
   ``message.tool_calls`` (defense-in-depth; the parser normally does not
   flush partial args in non-streaming mode, but this guarantees it).

The fix is gated entirely on the engine's own ``FinishReason.LENGTH`` signal
so a normal model-terminated tool call is unchanged.

Usage (inside the container, after vLLM source is on disk):
  python3 hotfix-dsv4-issue55-tool-truncation.py
  python3 hotfix-dsv4-issue55-tool-truncation.py /path/to/vllm
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
MARK = "# [issue55-hotfix] tool-call truncation safety"
SERVING = "entrypoints/openai/chat_completion/serving.py"

# Module-level helper, inserted right after `import asyncio` (always present
# at the top of this file, byte-stable).
HELPER_ANCHOR = "import asyncio\n"
HELPER_NEW = (
    "import asyncio\n"
    + MARK
    + "\n"
    "def _dsml_issue55_json_ok(s):\n"
    "    \"\"\"True iff `s` parses as a complete JSON value. None/'' are ok.\"\"\"\n"
    "    if s is None or s == \"\":\n"
    "        return True\n"
    "    try:\n"
    "        import json as _j\n"
    "        _j.loads(s)\n"
    "        return True\n"
    "    except Exception:\n"
    "        return False\n"
)

# --- Streaming final-chunk edit ---

STREAMING_OLD = (
    "                        if tools_streamed[i] and not tool_choice_function_name:\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )"
)

STREAMING_NEW = (
    "                        " + MARK + " gate the tool_calls verdict on engine length,\n"
    "                        # and strip non-JSON args from the trailing delta.\n"
    "                        if (\n"
    "                            tools_streamed[i]\n"
    "                            and not tool_choice_function_name\n"
    "                            and str(output.finish_reason) != \"length\"\n"
    "                        ):\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                str(output.finish_reason)\n"
    "                                if output.finish_reason\n"
    "                                else \"stop\"\n"
    "                            )\n"
    "                        if (\n"
    "                            str(output.finish_reason) == \"length\"\n"
    "                            and delta_message is not None\n"
    "                            and getattr(delta_message, \"tool_calls\", None)\n"
    "                        ):\n"
    "                            _kept = [\n"
    "                                tc\n"
    "                                for tc in delta_message.tool_calls\n"
    "                                if getattr(tc, \"function\", None)\n"
    "                                and _dsml_issue55_json_ok(tc.function.arguments)\n"
    "                            ]\n"
    "                            delta_message.tool_calls = _kept or None"
)

# --- Non-streaming edit ---

NOSTREAM_OLD = (
    "            choice_data = ChatCompletionResponseChoice(\n"
    "                index=output.index,\n"
    "                message=message,\n"
    "                logprobs=logprobs,\n"
    "                finish_reason=\"tool_calls\"\n"
    "                if is_finish_reason_tool_calls\n"
    "                else output.finish_reason\n"
    "                if output.finish_reason\n"
    "                else \"stop\","
)

NOSTREAM_NEW = (
    "            " + MARK + " if the engine truncated for length, do not\n"
    "            # claim tool_calls ended cleanly, and drop unparseable args.\n"
    "            if str(output.finish_reason) == \"length\":\n"
    "                is_finish_reason_tool_calls = False\n"
    "                if getattr(message, \"tool_calls\", None):\n"
    "                    message.tool_calls = [\n"
    "                        tc\n"
    "                        for tc in message.tool_calls\n"
    "                        if getattr(tc, \"function\", None)\n"
    "                        and _dsml_issue55_json_ok(tc.function.arguments)\n"
    "                    ] or None\n"
    "            choice_data = ChatCompletionResponseChoice(\n"
    "                index=output.index,\n"
    "                message=message,\n"
    "                logprobs=logprobs,\n"
    "                finish_reason=\"tool_calls\"\n"
    "                if is_finish_reason_tool_calls\n"
    "                else str(output.finish_reason)\n"
    "                if output.finish_reason\n"
    "                else \"stop\","
)


def _patch_file(path: Path, old: str, new: str, what: str) -> str:
    """Apply `new` over `old` once; idempotent. Returns applied|skipped|missing."""
    source = path.read_text(encoding="utf-8")
    if new in source:
        return "skipped"
    if old not in source:
        return "missing"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    return "applied"


def main() -> int:
    vllm_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VLLM
    path = vllm_root / SERVING
    if not path.is_file():
        print(f"[FAIL] serving file not found: {path}", file=sys.stderr)
        return 1
    print(f"[issue55-hotfix] patching {path}")
    for name, old, new in [
        ("module helper", HELPER_ANCHOR, HELPER_NEW),
        ("streaming final chunk", STREAMING_OLD, STREAMING_NEW),
        ("non-streaming final choice", NOSTREAM_OLD, NOSTREAM_NEW),
    ]:
        status = _patch_file(path, old, new, name)
        if status == "missing":
            print(f"[FAIL] {name}: anchor not found in {path.name}")
            return 1
        print(f"[issue55-hotfix] {name}: {status}")
    print(f"[issue55-hotfix] done (mark: {MARK})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())