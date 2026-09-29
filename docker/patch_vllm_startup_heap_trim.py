#!/usr/bin/env python3
"""Return unused CPU heap pages after vLLM's startup garbage collection."""

import argparse
import ast
from pathlib import Path
from textwrap import dedent


TARGET_REL = Path("vllm/utils/gc_utils.py")
MARKER = "spark-vllm-docker: trim unused CPU heap pages after startup GC"
TRIM_BLOCK = f'''
    # {MARKER}.
    # Both API servers and workers freeze their heaps after startup/warmup.
    # GC frees objects, but glibc can retain their pages until explicitly trimmed.
    try:
        import ctypes

        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        # Allocators/platforms without glibc's malloc_trim need no action here.
        pass
'''


def is_gc_call(node: ast.stmt, method: str) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "gc"
        and node.value.func.attr == method
    )


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "freeze_gc_heap"
    ]
    if len(functions) != 1:
        raise ValueError("Expected exactly one freeze_gc_heap function")
    function = functions[0]
    body = function.body
    lines = source.splitlines(keepends=True)
    already_patched = MARKER in source
    if already_patched:
        expected = ast.parse(dedent(TRIM_BLOCK)).body[0]
        block = "".join(lines[function.lineno - 1 : function.end_lineno])
        if (
            source.count(MARKER) != 1
            or MARKER not in block
            or ast.dump(body[-1]) != ast.dump(expected)
        ):
            raise ValueError("Incomplete startup CPU heap trim patch")
        body = body[:-1]

    # Restrict the patch to the final startup freeze, never graph-capture GC.
    if (
        not body
        or not is_gc_call(body[-1], "freeze")
        or body[-1].value.args
        or body[-1].value.keywords
    ):
        raise ValueError("Expected freeze_gc_heap to end with gc.freeze()")
    full_gc = [
        node
        for node in body[:-1]
        if is_gc_call(node, "collect")
        and not node.value.keywords
        and (
            not node.value.args
            or (
                len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == 2
            )
        )
    ]
    if not full_gc:
        raise ValueError("Expected full garbage collection before freezing the heap")
    if already_patched:
        return source

    end = function.end_lineno
    if not lines[end - 1].endswith("\n"):
        lines[end - 1] += "\n"
    lines.insert(end, TRIM_BLOCK)
    patched = "".join(lines)
    compile(patched, str(TARGET_REL), "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    target = args.source_root / TARGET_REL
    if not target.is_file():
        raise SystemExit(f"{target} not found; cannot apply startup CPU heap trim")
    source = target.read_text()
    try:
        patched = patch_source(source)
    except (ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to patch {target}: {exc}") from exc
    if patched == source:
        print("Startup CPU heap trim already patched; skipping")
    else:
        target.write_text(patched)
        print("Patched startup CPU heap trimming for API servers and workers")


if __name__ == "__main__":
    main()
