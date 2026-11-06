#!/usr/bin/env python3
"""Server-side repair for Inkling tool-call JSON.

Inkling occasionally *samples* an HTML entity (most often ``&quot;``) in place
of a structural JSON character — an artifact of HTML-heavy pretraining, seen
most often as the closing quote right after a URL value. The Inkling tool
parser's hand-rolled scanner (``vllm/parser/inkling.py``) looks for a literal
``"`` to terminate a JSON string; ``&quot;`` contains none, so the string never
closes, ``args`` extraction fails, and vLLM emits the raw
``<|content_invoke_tool_json|>...`` text as content instead of a tool call.

This patch unescapes complete JSON-structural HTML entities at the very top of
``_inkling_arg_converter`` — before the scanner runs. It is prefix-stable
(required by the engine's argument-delta diffing): while streaming, a trailing
incomplete entity is withheld until it completes, so each tick's output extends
the previous one. The patch is idempotent and refuses to run if the target
function's shape has changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# spark-vllm mod: inkling-tool-unescape v1"

HELPER_ANCHOR = "def _inkling_arg_converter(raw_args: str, partial: bool) -> str:"
BODY_ANCHOR = "    span = _args_value_span(raw_args)\n"
BODY_REPLACEMENT = (
    "    raw_args = _inkling_unescape_entities(raw_args, partial)\n"
    "    span = _args_value_span(raw_args)\n"
)
CALL_SITE = "_inkling_unescape_entities(raw_args, partial)"

HELPER_BLOCK = f'''{MARKER}
import re as _inkling_ue_re
import html as _inkling_ue_html

# Complete JSON-structural HTML entities the model occasionally samples in place
# of the literal character. Only forms WITH the trailing ';' match, so a partial
# entity split across streamed tokens is never rewritten mid-way.
_INKLING_UE_ENTITY_RE = _inkling_ue_re.compile(
    r"&(?:quot|amp|lt|gt|apos|#0*34|#0*39|#[xX]0*2[27]);"
)
# A trailing '&...' with no ';' yet: might still be growing into an entity.
_INKLING_UE_PARTIAL_RE = _inkling_ue_re.compile(r"&[A-Za-z0-9#xX]*\\Z")


def _inkling_unescape_entities(raw: str, partial: bool) -> str:
    """Rewrite complete JSON-structural HTML entities to their literal chars.

    Prefix-stable for the engine's argument-delta diffing: during streaming
    (``partial``) a trailing possibly-incomplete entity is withheld until it
    completes, so each tick's output extends the previous one.
    """
    if "&" not in raw:
        return raw
    if partial:
        m = _INKLING_UE_PARTIAL_RE.search(raw)
        if m is not None:
            raw = raw[: m.start()]
    return _INKLING_UE_ENTITY_RE.sub(
        lambda mo: _inkling_ue_html.unescape(mo.group(0)), raw
    )


'''


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check(path: Path) -> int:
    text = _read(path)
    if MARKER not in text:
        print("[inkling-tool-unescape] NOT patched", file=sys.stderr)
        return 1
    if CALL_SITE not in text:
        print(
            "[inkling-tool-unescape] marker present but converter call missing",
            file=sys.stderr,
        )
        return 1
    print("[inkling-tool-unescape] verified patched")
    return 0


def apply(path: Path) -> int:
    text = _read(path)
    if MARKER in text:
        print("[inkling-tool-unescape] already patched")
        return 0
    if HELPER_ANCHOR not in text:
        print(
            "[inkling-tool-unescape] converter definition not found — vLLM parser "
            "changed; refusing to patch.",
            file=sys.stderr,
        )
        return 2
    if text.count(BODY_ANCHOR) != 1:
        print(
            "[inkling-tool-unescape] converter body anchor not uniquely found "
            f"(count={text.count(BODY_ANCHOR)}); refusing to patch.",
            file=sys.stderr,
        )
        return 2
    text = text.replace(HELPER_ANCHOR, HELPER_BLOCK + HELPER_ANCHOR, 1)
    text = text.replace(BODY_ANCHOR, BODY_REPLACEMENT, 1)
    path.write_text(text, encoding="utf-8")
    print("[inkling-tool-unescape] patched", path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.target.is_file():
        print(f"[inkling-tool-unescape] target not found: {args.target}", file=sys.stderr)
        return 2
    return check(args.target) if args.check else apply(args.target)


if __name__ == "__main__":
    raise SystemExit(main())
