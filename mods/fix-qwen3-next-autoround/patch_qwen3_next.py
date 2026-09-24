#!/usr/bin/env python3
"""Restore checkpoint-configured quantization for the Qwen3-Next MoE router."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PREFIX = "[fix-qwen3-next-autoround]"


def gate_call(source: str) -> ast.Call:
    tree = ast.parse(source)
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3NextSparseMoeBlock"
    ]
    if len(classes) != 1:
        raise ValueError("expected exactly one Qwen3NextSparseMoeBlock class")
    constructors = [
        node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    ]
    if len(constructors) != 1:
        raise ValueError("expected exactly one Qwen3NextSparseMoeBlock.__init__")
    constructor = constructors[0]
    bindings = [
        node for node in constructor.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "quant_config" for t in node.targets)
    ]
    if len(bindings) != 1 or ast.unparse(bindings[0].value) != "vllm_config.quant_config":
        raise ValueError("expected quant_config = vllm_config.quant_config")

    gates = [
        node for node in ast.walk(constructor)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "gate"
            and isinstance(t.value, ast.Name) and t.value.id == "self"
            for t in node.targets
        )
    ]
    if len(gates) != 1:
        raise ValueError("expected exactly one self.gate assignment")
    call = gates[0].value
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id not in {"ReplicatedLinear", "GateLinear"}
    ):
        raise ValueError("unsupported self.gate constructor")
    if [ast.unparse(arg) for arg in call.args] != [
        "config.hidden_size", "config.num_experts"
    ]:
        raise ValueError("unexpected self.gate dimensions")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("unsupported self.gate keyword expansion")
    prefixes = [kw for kw in call.keywords if kw.arg == "prefix"]
    expected_prefix = ast.parse('f"{prefix}.gate"', mode="eval").body
    if len(prefixes) != 1 or ast.dump(prefixes[0].value) != ast.dump(expected_prefix):
        raise ValueError("unexpected self.gate prefix")
    return call


def patched_text(source: str) -> str:
    call = gate_call(source)
    keywords = [kw for kw in call.keywords if kw.arg == "quant_config"]
    if len(keywords) > 1:
        raise ValueError("duplicate self.gate quant_config keywords")
    if keywords:
        value = keywords[0].value
        if isinstance(value, ast.Name) and value.id == "quant_config":
            compile(source, "<qwen3_next.py>", "exec")
            return source
        if not isinstance(value, ast.Constant) or value.value is not None:
            raise ValueError("unsupported self.gate quant_config expression")
        replacement = "quant_config"
        start_node = value
        end_line, end_column = value.end_lineno, value.end_col_offset
    else:
        # GateLinear defaults to an unquantized gate after upstream PR #58234.
        # Keep GateLinear itself, including its quantized fallback behavior.
        start_node = next(kw for kw in call.keywords if kw.arg == "prefix")
        end_line, end_column = start_node.lineno, start_node.col_offset
        replacement = "quant_config=quant_config, "

    # AST columns are UTF-8 byte offsets, even when surrounding comments aren't ASCII.
    lines = source.encode("utf-8").splitlines(keepends=True)
    start = sum(map(len, lines[:start_node.lineno - 1])) + start_node.col_offset
    end = sum(map(len, lines[:end_line - 1])) + end_column
    raw = source.encode("utf-8")
    patched = (raw[:start] + replacement.encode("utf-8") + raw[end:]).decode("utf-8")
    compile(patched, "<patched qwen3_next.py>", "exec")
    updated = [kw.value for kw in gate_call(patched).keywords if kw.arg == "quant_config"]
    if len(updated) != 1 or not isinstance(updated[0], ast.Name) or updated[0].id != "quant_config":
        raise ValueError("gate quantization patch postcondition failed")
    return patched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    try:
        original = args.target.read_text(encoding="utf-8")
        patched = patched_text(original)
        if patched == original:
            print(f"{PREFIX} Gate already uses the model quantization config: {args.target}")
        else:
            args.target.write_text(patched, encoding="utf-8")
            print(f"{PREFIX} Patched router gate quantization: {args.target}")
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"{PREFIX} ERROR: cannot enable quantized router gate: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
