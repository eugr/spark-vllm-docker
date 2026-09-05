#!/usr/bin/env python3
"""Hotfix Issue #21: encode_arguments_to_dsml corrupts dict tool arguments.

Upstream: deepseek-ai/DeepSeek-V4-Flash-0731 encoding/encoding_dsv4.py
(installed into vLLM as tokenizers/deepseek_v4_encoding.py).

Broken path always json.loads(arguments); when arguments is already a dict,
json.loads raises and the except wraps it as {"arguments": <dict>}, poisoning
multi-turn tool history.

Usage (inside container, after encoder copy):
  python3 hotfix-encoding-dsv4-issue21.py
  python3 hotfix-encoding-dsv4-issue21.py /path/to/deepseek_v4_encoding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4_encoding.py"
)

OLD = """    try:
        arguments = json.loads(tool_call["arguments"])
    except Exception as err:
        arguments = {"arguments": tool_call["arguments"]}"""

NEW = """    raw = tool_call["arguments"]
    if isinstance(raw, dict):
        arguments = raw
    else:
        try:
            arguments = json.loads(raw)
        except (TypeError, ValueError):
            arguments = {"arguments": raw}"""


def patch_text(source: str) -> tuple[str, str]:
    """Return (updated_text, status) where status is applied|skipped|missing."""
    if NEW in source:
        return source, "skipped"
    if OLD not in source:
        return source, "missing"
    return source.replace(OLD, NEW, 1), "applied"


def patch_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    updated, status = patch_text(text)
    if status == "applied":
        path.write_text(updated, encoding="utf-8")
    return status


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET
    if not target.is_file():
        print(f"[FAIL] encoding file not found: {target}", file=sys.stderr)
        return 1
    status = patch_file(target)
    if status == "applied":
        print(f"[OK] Issue #21 patch applied: {target}")
        return 0
    if status == "skipped":
        print(f"[OK] Issue #21 patch already present: {target}")
        return 0
    print(
        f"[WARN] encode_arguments_to_dsml pattern not found; "
        f"Issue #21 patch skipped: {target}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
