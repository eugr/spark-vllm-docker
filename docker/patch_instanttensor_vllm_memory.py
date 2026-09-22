#!/usr/bin/env python3
"""Use vLLM's platform-aware free-memory accounting for InstantTensor's budget."""

import argparse
import ast
import importlib.util
from pathlib import Path


ORIGINAL = """        free_bytes, total_bytes = torch.cuda.mem_get_info()
        avail_bytes = int(free_bytes * max_free_mem_usage)
"""
PATCHED = """        # Share vLLM's UMA accounting and its CUDA-on-WSL policy.
        from vllm.utils.mem_utils import MemorySnapshot

        free_bytes = MemorySnapshot(device=self.device).free_memory
        avail_bytes = int(free_bytes * max_free_mem_usage)
"""


def patch_memory_query(source: str) -> str:
    methods = [
        method
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "safe_open"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "_determine_io_params"
    ]
    if len(methods) != 1:
        raise ValueError("Expected exactly one safe_open._determine_io_params method")

    method = methods[0]
    lines = source.splitlines(keepends=True)
    start, end = method.lineno - 1, method.end_lineno
    body = "".join(lines[start:end])
    original_count, patched_count = body.count(ORIGINAL), body.count(PATCHED)
    if original_count == 0 and patched_count == 1:
        return source
    if original_count != 1 or patched_count != 0:
        raise ValueError("Expected exactly one known InstantTensor memory-budget query")

    lines[start:end] = [body.replace(ORIGINAL, PATCHED, 1)]
    patched = "".join(lines)
    compile(patched, "instanttensor/_impl.py", "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", nargs="?", type=Path, help="Path to instanttensor/_impl.py"
    )
    parser.add_argument(
        "--installed", action="store_true", help="Patch installed InstantTensor"
    )
    args = parser.parse_args()
    if args.installed:
        if args.target is not None:
            parser.error("Use either target or --installed")
        # Locate the top-level package without importing InstantTensor or CUDA.
        spec = importlib.util.find_spec("instanttensor")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if len(locations) != 1:
            raise SystemExit("Unable to locate the installed InstantTensor package")
        target = Path(locations[0]) / "_impl.py"
    elif args.target is not None:
        target = args.target
    else:
        parser.error("Provide target or --installed")

    try:
        source = target.read_text()
        patched = patch_memory_query(source)
        if patched == source:
            print("InstantTensor memory accounting is already patched; skipping")
        else:
            target.write_text(patched)
            print("Patched InstantTensor to use vLLM's available-memory accounting")
    except (OSError, ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to patch {target}: {exc}") from exc


if __name__ == "__main__":
    main()
