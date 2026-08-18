#!/usr/bin/env python3
"""Pin CUTLASS DSL requirements in checked-out source metadata.

The regular and B12X vLLM refs currently carry older exact pins.  Keep their
wheel metadata aligned with the CUTLASS DSL version installed by this image.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIREMENT = re.compile(
    r"(?m)^(?P<prefix>\s*[\"']?)"
    r"(?P<name>nvidia-cutlass-dsl(?:-libs-(?:base|core|cu12|cu13))?)"
    r"(?P<extra>\[cu13\])?"
    r"(?P<before>\s*)==(?P<after>\s*)"
    r"(?P<version>[^\s,\"']+)"
)


def pin_text(text: str, version: str) -> tuple[str, int]:
    def replace(match: re.Match[str]) -> str:
        return (
            f"{match.group('prefix')}{match.group('name')}"
            f"{match.group('extra') or ''}{match.group('before')}=="
            f"{match.group('after')}{version}"
        )

    return REQUIREMENT.subn(replace, text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", args.version):
        parser.error(f"invalid CUTLASS DSL version: {args.version}")

    updates: list[tuple[Path, str]] = []
    total = 0
    for path in args.paths:
        original = path.read_text()
        updated, count = pin_text(original, args.version)
        total += count
        updates.append((path, updated))

    if total != args.expected_count:
        raise SystemExit(
            "CUTLASS DSL pin failed: expected "
            f"{args.expected_count} requirements, found {total}"
        )

    for path, updated in updates:
        path.write_text(updated)
    print(f"Pinned {total} CUTLASS DSL requirements to {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
