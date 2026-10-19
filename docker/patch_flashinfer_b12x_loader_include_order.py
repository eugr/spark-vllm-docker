#!/usr/bin/env python3
"""Restore B12X loader declarations before their use in FlashInfer source.

FlashInfer commit 601cb127 alphabetized the .c includes in _storage.c.
_batch.c includes _bounce.c, which needs IO_ALIGNMENT and
validate_direct_range from _direct.c. Patch only that known broken block;
upstream fixes and refs without the embedded loader need no workaround.
"""

import argparse
from pathlib import Path


TARGET_REL = Path("flashinfer/experimental/b12x/loader/_storage.c")
BROKEN_INCLUDES = '#include "_batch.c"\n#include "_direct.c"\n'
FIXED_INCLUDES = '#include "_direct.c"\n#include "_batch.c"\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    target = args.source_root / TARGET_REL
    if not target.exists():
        print("FlashInfer B12X loader is absent; include-order workaround not applicable")
        return

    source = target.read_text()
    count = source.count(BROKEN_INCLUDES)
    if count == 0:
        print("Known FlashInfer B12X loader include-order regression is absent; skipping")
        return
    if count != 1:
        raise SystemExit(f"Expected one B12X loader include block in {target}, found {count}")

    target.write_text(source.replace(BROKEN_INCLUDES, FIXED_INCLUDES, 1))
    print("Patched FlashInfer B12X loader to include _direct.c before _batch.c")


if __name__ == "__main__":
    main()
