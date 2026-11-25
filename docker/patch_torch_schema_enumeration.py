#!/usr/bin/env python3
"""Avoid repeated schema.arguments lookups in PyTorch's fill_defaults.

Adapted from local-inference-lab/blackwell-llm-docker:
https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/recipes/glm53/torch-schema-enumeration.patch
"""

import argparse
import ast
import importlib.util
from pathlib import Path


ORIGINAL = """    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
"""
PATCHED = """    for i, info in enumerate(schema.arguments):
        if info.kwarg_only:
"""


def patch_fill_defaults(source: str) -> str:
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "fill_defaults"
    ]
    if len(functions) != 1:
        raise ValueError("Expected exactly one fill_defaults function")

    function = functions[0]
    lines = source.splitlines(keepends=True)
    start, end = function.lineno - 1, function.end_lineno
    body = "".join(lines[start:end])
    original_count, patched_count = body.count(ORIGINAL), body.count(PATCHED)
    if original_count == 0 and patched_count == 1:
        return source
    if original_count != 1 or patched_count != 0:
        raise ValueError("Expected exactly one known fill_defaults schema loop")

    lines[start:end] = [body.replace(ORIGINAL, PATCHED, 1)]
    patched = "".join(lines)
    compile(patched, "torch/_library/utils.py", "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", nargs="?", type=Path, help="Path to torch/_library/utils.py"
    )
    parser.add_argument("--installed", action="store_true", help="Patch installed Torch")
    args = parser.parse_args()
    if args.installed:
        if args.target is not None:
            parser.error("Use either target or --installed")
        # Locate the top-level package without importing Torch or initializing CUDA.
        spec = importlib.util.find_spec("torch")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if len(locations) != 1:
            raise SystemExit("Unable to locate the installed Torch package")
        target = Path(locations[0]) / "_library/utils.py"
    elif args.target is not None:
        target = args.target
    else:
        parser.error("Provide target or --installed")

    try:
        source = target.read_text()
        patched = patch_fill_defaults(source)
        if patched == source:
            print("Torch fill_defaults schema enumeration is already patched; skipping")
        else:
            target.write_text(patched)
            print("Patched Torch fill_defaults to enumerate schema.arguments once")
    except (OSError, ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to patch {target}: {exc}") from exc


if __name__ == "__main__":
    main()
