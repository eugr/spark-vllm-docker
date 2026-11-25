#!/usr/bin/env python3
"""Bound live B12X tuning candidates on Spark's shared memory."""

import argparse
import ast
from pathlib import Path


TARGET = Path("vllm/model_executor/warmup/b12x_prepare.py")


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "get_b12x_session"]
    if len(functions) != 1:
        raise ValueError("Expected one get_b12x_session function")
    calls = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "PreparationSession"]
    if len(calls) != 1:
        raise ValueError("Expected one PreparationSession constructor")
    call = calls[0]
    configured = [kw for kw in call.keywords if kw.arg == "race_batch"]
    if configured:
        if (
            len(configured) == 1
            and isinstance(configured[0].value, ast.Constant)
            and type(configured[0].value.value) is int
            and configured[0].value.value == 1
        ):
            return source
        raise ValueError("Upstream sets a tuning batch; review its memory policy")
    lines = source.splitlines(keepends=True)
    # The call is on an assignment's RHS; keyword indentation follows the
    # existing first argument rather than the constructor's column.
    if not call.keywords:
        raise ValueError("Expected keyword arguments for PreparationSession")
    first = call.keywords[0]
    indent = lines[first.lineno - 1][:first.col_offset]
    lines[first.lineno - 1:first.lineno - 1] = [
        indent + "# Keep only one challenger plus the incumbent alive. Large\n",
        indent + "# vision trials can each allocate GiB of shared Spark RAM.\n",
        indent + "race_batch=1,\n",
    ]
    patched = "".join(lines)
    compile(patched, str(TARGET), "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    target = args.source_root / TARGET
    if not target.exists():
        print("B12X preparation module is absent; tuning-batch patch is not applicable")
        return
    source = target.read_text()
    try:
        patched = patch_source(source)
    except (ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to patch {target}: {exc}") from exc
    if patched != source:
        target.write_text(patched)
    print("B12X tuning uses one challenger per batch; autotuning remains enabled")


if __name__ == "__main__":
    main()
