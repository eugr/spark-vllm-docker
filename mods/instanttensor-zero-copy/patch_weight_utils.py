#!/usr/bin/env python3
"""Enable zero-copy views in vLLM's InstantTensor weights iterator."""

from __future__ import annotations

import argparse
import ast
import stat
import sys
from pathlib import Path

PREFIX = "[instanttensor-zero-copy]"
MARKER = "# spark-vllm mod: instanttensor-zero-copy v1"
OWNERSHIP_COMMENT = (
    "    # copy=True yields tensors that own their memory, staying valid after the\n"
    "    # context exits or InstantTensor reuses its buffer.\n"
)
ZERO_COPY_COMMENT = (
    "    # This launch opts into InstantTensor views that are consumed inline.\n"
)


def _is_safe_open_call(node: ast.Call) -> bool:
    function = node.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "safe_open"
        and isinstance(function.value, ast.Name)
        and function.value.id == "instanttensor"
    )


def _copy_keyword(text: str) -> tuple[ast.keyword, bool]:
    tree = ast.parse(text)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "instanttensor_weights_iterator"
    ]
    if len(functions) != 1:
        raise ValueError(
            "expected exactly one instanttensor_weights_iterator function; "
            f"found {len(functions)}"
        )

    calls = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call) and _is_safe_open_call(node)
    ]
    if len(calls) != 1:
        raise ValueError(
            "expected exactly one instanttensor.safe_open call in "
            f"instanttensor_weights_iterator; found {len(calls)}"
        )

    keywords = [keyword for keyword in calls[0].keywords if keyword.arg == "copy"]
    if len(keywords) != 1:
        raise ValueError(
            "expected instanttensor.safe_open to have exactly one explicit "
            f"copy keyword; found {len(keywords)}"
        )

    value = keywords[0].value
    if not isinstance(value, ast.Constant) or not isinstance(value.value, bool):
        raise ValueError("instanttensor.safe_open copy keyword is not a bool literal")
    return keywords[0], value.value


def _source_offset(text: str, lineno: int, column: int) -> int:
    lines = text.splitlines(keepends=True)
    if lineno < 1 or lineno > len(lines):
        raise ValueError(f"invalid source line {lineno}")
    # vLLM's keyword line is ASCII. Requiring an ASCII prefix keeps the AST
    # byte-column offset equivalent to a Python string offset.
    prefix = lines[lineno - 1][:column]
    if not prefix.isascii():
        raise ValueError("non-ASCII text before InstantTensor copy keyword")
    return sum(len(line) for line in lines[: lineno - 1]) + column


def patched_text(text: str) -> str:
    marker_count = text.count(MARKER)
    if marker_count > 1:
        raise ValueError(f"zero-copy marker occurs {marker_count} times")

    keyword, copy_enabled = _copy_keyword(text)
    if not copy_enabled:
        compile(text, "<InstantTensor weight_utils.py>", "exec")
        return text
    if marker_count:
        raise ValueError("zero-copy marker exists but copy=True is still enabled")

    value = keyword.value
    if value.end_lineno is None or value.end_col_offset is None:
        raise ValueError("Python AST did not provide a source range for copy=True")
    start = _source_offset(text, value.lineno, value.col_offset)
    end = _source_offset(text, value.end_lineno, value.end_col_offset)
    if text[start:end] != "True":
        raise ValueError("copy=True source range did not contain the expected token")

    line_start = _source_offset(text, keyword.lineno, 0)
    keyword_line = text.splitlines(keepends=True)[keyword.lineno - 1]
    indentation = keyword_line[: keyword.col_offset]
    if not indentation.isspace():
        raise ValueError("copy keyword is not on its own indented argument line")

    patched = text[:start] + "False" + text[end:]
    patched = patched[:line_start] + f"{indentation}{MARKER}\n" + patched[line_start:]
    if patched.count(OWNERSHIP_COMMENT) == 1:
        patched = patched.replace(OWNERSHIP_COMMENT, ZERO_COPY_COMMENT, 1)

    _, patched_copy_enabled = _copy_keyword(patched)
    if patched_copy_enabled or patched.count(MARKER) != 1:
        raise ValueError("zero-copy patch postcondition failed")
    compile(patched, "<patched InstantTensor weight_utils.py>", "exec")
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="validate compatibility without writing"
    )
    args = parser.parse_args()

    if not args.target.is_file():
        print(f"{PREFIX} ERROR: target not found: {args.target}", file=sys.stderr)
        return 1

    original = args.target.read_text()
    try:
        patched = patched_text(original)
        _, copy_enabled = _copy_keyword(original)
    except (SyntaxError, ValueError) as exc:
        print(
            f"{PREFIX} ERROR: refusing to patch {args.target}: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.check:
        if patched != original:
            state = "compatible"
        elif copy_enabled:
            state = "unexpectedly unchanged"
        elif MARKER in original:
            state = "already patched"
        else:
            state = "already zero-copy"
        print(f"{PREFIX} {args.target} is {state}.")
        return 0

    if patched == original:
        state = "already patched" if MARKER in original else "already zero-copy"
        print(f"{PREFIX} InstantTensor iterator is {state}; skipping.")
        return 0

    mode = stat.S_IMODE(args.target.stat().st_mode)
    temporary = args.target.with_suffix(args.target.suffix + ".zero-copy-mod.tmp")
    temporary.write_text(patched)
    temporary.chmod(mode)
    temporary.replace(args.target)
    print(f"{PREFIX} Patched {args.target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
