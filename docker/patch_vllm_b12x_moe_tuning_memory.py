#!/usr/bin/env python3
"""Keep b12x MoE trial tensors alive only for the lifetime of their calls."""

import argparse
import ast
from pathlib import Path


TARGET_REL = Path("model_executor/layers/fused_moe/b12x.py")
MARKER = "spark-vllm-docker: keep MoE trial owners out of serving plans"


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    methods = [
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_PreparedMoECall"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "make"
    ]
    if not methods:
        return source
    if len(methods) != 1:
        raise ValueError("Expected exactly one _PreparedMoECall.make method")
    method = methods[0]
    calls = [
        node for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "PreparedCall"
    ]
    if len(calls) != 1:
        raise ValueError("Expected exactly one MoE PreparedCall constructor")
    owners = [kw for kw in calls[0].keywords if kw.arg == "owners"]
    lines = source.splitlines(keepends=True)
    start, end = method.lineno - 1, method.end_lineno
    block = "".join(lines[start:end])
    if MARKER in block:
        if owners or "ids.copy_(route_ids[pattern])" not in block:
            raise ValueError("Incomplete MoE tuning-memory patch")
        return source
    if not owners:
        # Older refs do not export trial buffers into the serving plan.
        return source
    if len(owners) != 1 or not (
        isinstance(owners[0].value, ast.Name) and owners[0].value.id == "tensors"
    ):
        raise ValueError("Unknown MoE trial ownership; review upstream before patching")

    replacements = (
        (
            "        route_patterns = tuple(route_ids.unbind(0))\n",
            f"        # {MARKER}.\n"
            "        # The producer holds route_ids itself, preserving the factory's\n"
            "        # weakref reuse while trials run. PreparedCall.owners would be\n"
            "        # copied into the long-lived plan by PreparationSession._publish.\n",
        ),
        ("ids.copy_(route_patterns[pattern])", "ids.copy_(route_ids[pattern])"),
        ("            owners=tensors,\n", ""),
        ("range(len(route_patterns))", "range(route_ids.shape[0])"),
    )
    for before, after in replacements:
        if block.count(before) != 1:
            raise ValueError(f"Unknown MoE trial layout: expected one {before.strip()!r}")
        block = block.replace(before, after, 1)
    lines[start:end] = [block]
    patched = "".join(lines)
    compile(patched, str(TARGET_REL), "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path)
    args = parser.parse_args()
    package_root = (args.source_root or Path.cwd()) / "vllm"
    target = package_root / TARGET_REL
    if not target.exists():
        print("B12X MoE backend is absent; tuning-memory patch is not applicable")
        return
    source = target.read_text()
    try:
        patched = patch_source(source)
    except (ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to patch {target}: {exc}") from exc
    if patched == source:
        print("B12X MoE trial ownership is unaffected or already patched; skipping")
    else:
        target.write_text(patched)
        print("Patched B12X MoE to release trial buffers before KV cache profiling")


if __name__ == "__main__":
    main()
