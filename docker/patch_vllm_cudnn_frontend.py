#!/usr/bin/env python3
"""Pair the FlashInfer #9765 fix with removal of vLLM #68566's FE cap.

Apply each fix to its source tree before building the wheel.
Remove when supported upstream refs have both changes.
"""

import argparse
from pathlib import Path
import re


TARGET = "flashinfer/cudnn/linear_attention.py"
OLD = "O, fs, _checkpoints = getattr(graph, family)(**ports, **attrs)"
FIXED = "O, fs = getattr(graph, family)(**ports, **attrs)[:2]"


def patch_flashinfer(source: str) -> str:
    if source.count(OLD) > 1:
        raise ValueError("Unexpected duplicate FlashInfer graph output assignment")
    return source.replace(OLD, FIXED)


def patch_vllm(source: str) -> str:
    # Only lift the exact temporary cap introduced for the unpacking bug.
    # Preserve the graph-lifetime floor and every unrelated requirement.
    return re.sub(
        r"(?m)^(nvidia-cudnn-frontend\s*>=\s*1\.31\.0),<1\.33(?=\s*(?:#.*)?$)",
        r"\1", source,
    ).replace(
        "# <1.33 because FlashInfer 0.7.4's cuDNN linear attention unpacks three results\n"
        "# from graph.kda(), and 1.33.0 returns more. Lift with the FlashInfer bump.\n",
        "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("flashinfer", "vllm"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    relative = TARGET if args.mode == "flashinfer" else "requirements/cuda.txt"
    target = args.root / relative
    if not target.exists():
        if args.mode == "flashinfer":
            print("FlashInfer cuDNN linear attention absent; skipping")
            return
        raise ValueError(f"Missing vLLM CUDA requirements: {target}")
    source = target.read_text()
    updated = (patch_flashinfer if args.mode == "flashinfer" else patch_vllm)(source)
    if updated != source:
        target.write_text(updated)
        print(f"Patched cuDNN Frontend compatibility in {target}")
    else:
        print(f"Known cuDNN Frontend regression absent in {target}; skipping")


if __name__ == "__main__":
    main()
