#!/usr/bin/env python3
"""Apply the temporary upstream CuTe cache-integrity fix to installed B12X."""

import argparse
import importlib.util
from pathlib import Path
import subprocess


PATCH = Path(__file__).with_name("b12x-cache-integrity.patch")


def apply_patch(root: Path) -> bool:
    """Check every package hunk before writing; accept an identical reapply."""
    command = ["git", "apply"]
    if not (root / "b12x/_lib/compile_plan.py").exists():
        # PyPI 1.3.0 has the same object cache but no preparation planner.
        # Its normal compile/load path still receives integrity and fsync fixes.
        command.append("--exclude=b12x/_lib/compile_plan.py")
    command.append("--include=b12x/**")
    result = subprocess.run(
        command + ["--check", str(PATCH)], cwd=root, capture_output=True, text=True,
    )
    if result.returncode:
        repeated = subprocess.run(
            command + ["--reverse", "--check", str(PATCH)],
            cwd=root, capture_output=True, text=True,
        )
        if repeated.returncode == 0:
            return False
        raise RuntimeError(
            "B12X source differs from the reviewed cache implementation; "
            "refresh the upstream patch before building.\n" + result.stderr.strip()
        )
    subprocess.run(command + [str(PATCH)], cwd=root, check=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path,
                        help="Checkout or site-packages directory containing b12x/")
    parser.add_argument("--installed", action="store_true")
    args = parser.parse_args()
    if args.installed:
        if args.source_root is not None:
            parser.error("Use either source_root or --installed")
        # Locate the package without importing B12X, Torch, or initializing CUDA.
        spec = importlib.util.find_spec("b12x")
        if spec is None:
            print("B12X is not installed; cache-integrity patch is not applicable")
            return
        locations = list(spec.submodule_search_locations or [])
        if len(locations) != 1 or Path(locations[0]).name != "b12x":
            raise SystemExit("Unable to locate the installed B12X package")
        root = Path(locations[0]).parent
    elif args.source_root is not None:
        root = args.source_root.resolve()
    else:
        parser.error("Provide source_root or --installed")
    if not (root / "b12x/_lib/compiler.py").is_file():
        raise SystemExit(f"B12X CuTe compiler not found under {root}")
    try:
        changed = apply_patch(root)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Unable to patch B12X cache integrity: {error}") from error
    print("Patched B12X CuTe cache validation and durable writes" if changed else
          "B12X cache integrity is already patched; skipping")


if __name__ == "__main__":
    main()
