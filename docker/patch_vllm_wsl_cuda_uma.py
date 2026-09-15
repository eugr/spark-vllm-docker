#!/usr/bin/env python3
"""Keep CUDA's memory budget on WSL instead of using guest RAM availability."""

import argparse
import ast
import importlib.util
from pathlib import Path


UMA_CONDITION = "current_platform.is_integrated_gpu(device.index)"
WSL_CONDITION = (
    f"{UMA_CONDITION} and not (current_platform.is_cuda() and in_wsl())"
)


def patch_mem_utils(source: str) -> str:
    tree = ast.parse(source)
    measures = [
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MemorySnapshot"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "measure"
    ]
    if len(measures) != 1:
        raise ValueError("Expected exactly one MemorySnapshot.measure method")

    original = ast.dump(ast.parse(UMA_CONDITION, mode="eval").body)
    patched = ast.dump(ast.parse(WSL_CONDITION, mode="eval").body)
    guards = [
        node
        for node in measures[0].body
        if isinstance(node, ast.If) and ast.dump(node.test) in (original, patched)
    ]
    if len(guards) != 1:
        raise ValueError("Expected exactly one known MemorySnapshot UMA guard")

    guard = guards[0]
    if ast.dump(guard.test) == original:
        lines = source.splitlines(keepends=True)
        indent = " " * guard.col_offset
        condition = guard.test
        prefix = lines[condition.lineno - 1][: condition.col_offset]
        suffix = lines[condition.end_lineno - 1][condition.end_col_offset :]
        lines[condition.lineno - 1 : condition.end_lineno] = [
            f"{prefix}{UMA_CONDITION} and not (\n"
            f"{indent}    current_platform.is_cuda() and in_wsl()\n"
            f"{indent}){suffix}"
        ]
        lines.insert(
            guard.lineno - 1,
            f"{indent}# WSL guest RAM does not describe CUDA's memory allocation budget.\n",
        )
        source = "".join(lines)

    has_wsl_import = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "vllm.platforms.interface"
        and any(alias.name == "in_wsl" and alias.asname is None for alias in node.names)
        for node in tree.body
    )
    if not has_wsl_import:
        anchor = "from vllm.platforms import current_platform\n"
        if source.count(anchor) != 1:
            raise ValueError("Expected exactly one current_platform import")
        source = source.replace(
            anchor, anchor + "from vllm.platforms.interface import in_wsl\n", 1
        )

    compile(source, "mem_utils.py", "exec")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path)
    parser.add_argument("--installed", action="store_true", help="Patch installed vLLM")
    args = parser.parse_args()
    if args.installed:
        if args.source_root is not None:
            parser.error("Use either source_root or --installed")
        # Finding the top-level package avoids importing vLLM or initializing CUDA.
        spec = importlib.util.find_spec("vllm")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if len(locations) != 1:
            raise SystemExit("Unable to locate the installed vLLM package")
        package_root = Path(locations[0])
    else:
        package_root = (args.source_root or Path.cwd()) / "vllm"

    target = package_root / "utils/mem_utils.py"
    source = target.read_text()
    try:
        patched = patch_mem_utils(source)
    except (ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to patch {target}: {exc}") from exc
    if patched == source:
        print("CUDA-on-WSL memory reporting is already patched; skipping")
    else:
        target.write_text(patched)
        print("Preserved CUDA memory reporting on WSL UMA devices")


if __name__ == "__main__":
    main()
