#!/usr/bin/env python3
"""Preserve explicitly independent packed draft heads during MTP loading."""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "QWEN_PRESERVE_PACKED_DRAFT_HEAD_V1"


def patch(path: Path) -> None:
    source = path.read_text()
    if MARKER in source:
        return
    anchor = (
        "    \"\"\"Share when the draft has no own copy, or its copy matches the target.\"\"\"\n\n"
        "    if not getattr(eagle, flag, False) or draft is None:\n"
    )
    replacement = (
        "    \"\"\"Share when the draft has no own copy, or its copy matches the target.\"\"\"\n\n"
        "    # QWEN_PRESERVE_PACKED_DRAFT_HEAD_V1\n"
        "    if getattr(draft, \"_qwen_force_own_lm_head\", False):\n"
        "        return False\n"
        "    if not getattr(eagle, flag, False) or draft is None:\n"
    )
    if anchor not in source:
        raise ValueError("Eagle head-sharing anchor not found")
    path.write_text(source.replace(anchor, replacement, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    patch(args.path)


if __name__ == "__main__":
    main()
