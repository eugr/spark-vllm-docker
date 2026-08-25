#!/usr/bin/env python3
"""Import the C128A top-k alignment used by the B12X DeepSeek V4 path.

The local-inference-lab B12X branch added a dynamic C128A top-k width that
references ``_C128A_TOPK_ALIGNMENT`` without importing it from
``compressor_utils``.  Keep this workaround opt-in, idempotent, and strict
about the vulnerable source shape so regular vLLM builds remain untouched.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


FLAG_NAME = "VLLM_PATCH_B12X_C128A_ALIGNMENT"
ALIGNMENT_NAME = "_C128A_TOPK_ALIGNMENT"
TARGET_REL = Path("vllm/models/deepseek_v4/sparse_mla.py")
HELPER_REL = Path("vllm/v1/attention/backends/mla/compressor_utils.py")
IMPORT_BLOCK = (
    "from vllm.v1.attention.backends.mla.compressor_utils import (\n"
    "    get_c128a_topk_width,\n"
    "    get_compressed_slot_mapping,\n"
    ")"
)
PATCHED_IMPORT_BLOCK = (
    "from vllm.v1.attention.backends.mla.compressor_utils import (\n"
    f"    {ALIGNMENT_NAME},\n"
    "    get_c128a_topk_width,\n"
    "    get_compressed_slot_mapping,\n"
    ")"
)
TRUE_VALUES = {"1", "true", "TRUE", "yes", "YES"}
FALSE_VALUES = {"", "0", "false", "FALSE", "no", "NO"}


def module_binds_name(tree: ast.Module, name: str) -> bool:
    """Return whether the module establishes ``name`` before runtime use."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if any((alias.asname or alias.name) == name for alias in node.names):
                return True
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return True
    return False


def module_constant_is_128(tree: ast.Module, name: str) -> bool:
    """Return whether ``name`` is assigned the expected value at module scope."""
    for node in tree.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            value = node.value
        if isinstance(value, ast.Constant) and value.value == 128:
            return True
    return False


raw_flag = os.environ.get(FLAG_NAME, "0")
if raw_flag not in TRUE_VALUES | FALSE_VALUES:
    raise SystemExit(f"Invalid {FLAG_NAME} value: {raw_flag!r}")
if raw_flag not in TRUE_VALUES:
    print("B12X C128A alignment workaround not requested; skipping")
    raise SystemExit(0)

source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / TARGET_REL
helper = source_root / HELPER_REL

if not target.exists():
    print(f"{TARGET_REL} is absent; B12X C128A workaround is not applicable")
    raise SystemExit(0)

text = target.read_text()
try:
    tree = ast.parse(text, filename=str(target))
except SyntaxError as exc:
    raise SystemExit(f"Cannot inspect invalid Python source {target}: {exc}") from exc

uses_alignment = any(
    isinstance(node, ast.Name)
    and node.id == ALIGNMENT_NAME
    and isinstance(node.ctx, ast.Load)
    for node in ast.walk(tree)
)
if not uses_alignment:
    print("DeepSeek V4 source does not use the missing C128A alignment; skipping")
    raise SystemExit(0)
if module_binds_name(tree, ALIGNMENT_NAME):
    print("DeepSeek V4 C128A alignment is already defined or imported; skipping")
    raise SystemExit(0)

if not helper.exists():
    raise SystemExit(
        f"{target} uses {ALIGNMENT_NAME}, but {helper} is missing; refusing to guess"
    )
helper_text = helper.read_text()
try:
    helper_tree = ast.parse(helper_text, filename=str(helper))
except SyntaxError as exc:
    raise SystemExit(f"Cannot inspect invalid Python source {helper}: {exc}") from exc
if not module_constant_is_128(helper_tree, ALIGNMENT_NAME):
    raise SystemExit(
        f"{target} uses {ALIGNMENT_NAME}, but {helper} does not define it as 128; "
        "refusing to patch an unknown source shape"
    )

anchor_count = text.count(IMPORT_BLOCK)
if anchor_count != 1:
    raise SystemExit(
        f"Expected one DeepSeek V4 compressor_utils import block, found "
        f"{anchor_count}; refusing to patch an unknown source shape"
    )

updated = text.replace(IMPORT_BLOCK, PATCHED_IMPORT_BLOCK, 1)
compile(updated, str(target), "exec")
updated_tree = ast.parse(updated, filename=str(target))
if not module_binds_name(updated_tree, ALIGNMENT_NAME):
    raise SystemExit("B12X C128A alignment import was not established after patching")

target.write_text(updated)
print("Applied B12X DeepSeek V4 C128A top-k alignment import workaround")
