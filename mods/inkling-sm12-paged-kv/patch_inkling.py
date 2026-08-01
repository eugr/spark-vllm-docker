#!/usr/bin/env python3
"""Patch only Inkling's NVIDIA FA4 dispatch for the vendored SM12 kernel."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "# spark-vllm mod: inkling-sm12-paged-kv v1"

HELPER_ANCHOR = "\n\n@cache\ndef _get_score_mod"
HELPER_REPLACEMENT = f'''

{MARKER}
@cache
def _use_sm12_paged_kv() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major == 12


@cache
def _get_score_mod'''

SEQLEN_IMPORT = (
    "    from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK\n"
)
SEQLEN_IMPORT_REPLACEMENT = """    if _use_sm12_paged_kv():
        from vllm.third_party.inkling_sm120_fa4.seqlen_info import SeqlenInfoQK
    else:
        from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK
"""

SPLIT_GUARD = """    if capability is not None and capability.major == 9:
        return 1
"""
SPLIT_GUARD_REPLACEMENT = """    if capability is not None and capability.major in (9, 12):
        # Hopper is unsplit in upstream Inkling. The vendored SM12 paged-KV
        # kernel is also validated only with a single split.
        return 1
"""

UPSTREAM_DISPATCH = """    else:
        from vllm.vllm_flash_attn.cute import (
            flash_attn_varlen_func as cute_flash_attn_varlen_func,
        )

        flash_attn_varlen_func = cute_flash_attn_varlen_func
        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }
"""

LEGACY_UPSTREAM_DISPATCH = """    else:
        from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }
"""

LEGACY_MODDED_DISPATCH = """    else:
        if _use_sm12_paged_kv():
            from vllm.third_party.inkling_sm120_fa4_adapter import (
                flash_attn_varlen_func,
            )
        else:
            from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }
"""

MODDED_DISPATCH = """    else:
        if _use_sm12_paged_kv():
            from vllm.third_party.inkling_sm120_fa4_adapter import (
                flash_attn_varlen_func as cute_flash_attn_varlen_func,
            )
        else:
            from vllm.vllm_flash_attn.cute import (
                flash_attn_varlen_func as cute_flash_attn_varlen_func,
            )

        flash_attn_varlen_func = cute_flash_attn_varlen_func
        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one {label}; found {count}")
    return text.replace(old, new, 1)


def validate_shape(text: str) -> None:
    tree = ast.parse(text)
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {
        "_use_sheared_bias",
        "_get_score_mod",
        "inkling_fa4_num_splits",
        "inkling_fa4_rel_attention",
    }
    missing = sorted(required - functions)
    if missing:
        raise ValueError(f"unexpected Inkling FA4 module; missing {', '.join(missing)}")


def patched_text(text: str) -> str:
    validate_shape(text)
    if MARKER in text:
        if "vllm.third_party.inkling_sm120_fa4_adapter" not in text:
            raise ValueError("mod marker exists but the SM12 adapter dispatch is missing")
        return text

    text = replace_once(
        text, HELPER_ANCHOR, HELPER_REPLACEMENT, "_get_score_mod anchor"
    )
    text = replace_once(
        text, SEQLEN_IMPORT, SEQLEN_IMPORT_REPLACEMENT, "SeqlenInfoQK import"
    )
    text = replace_once(text, SPLIT_GUARD, SPLIT_GUARD_REPLACEMENT, "split guard")
    if UPSTREAM_DISPATCH in text:
        text = replace_once(
            text, UPSTREAM_DISPATCH, MODDED_DISPATCH, "standard FA4 dispatch block"
        )
    elif LEGACY_UPSTREAM_DISPATCH in text:
        text = replace_once(
            text,
            LEGACY_UPSTREAM_DISPATCH,
            LEGACY_MODDED_DISPATCH,
            "legacy FA4 dispatch block",
        )
    else:
        raise ValueError("no supported standard FA4 dispatch block was found")
    validate_shape(text)
    compile(text, "<patched fa4_rel_attention.py>", "exec")
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="validate compatibility without writing"
    )
    args = parser.parse_args()

    if not args.target.is_file():
        print(f"[inkling-sm12-paged-kv ERROR] target not found: {args.target}", file=sys.stderr)
        return 1

    original = args.target.read_text()
    try:
        patched = patched_text(original)
    except (SyntaxError, ValueError) as exc:
        print(
            f"[inkling-sm12-paged-kv ERROR] refusing to patch {args.target}: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.check:
        state = "already patched" if patched == original else "compatible"
        print(f"[inkling-sm12-paged-kv] {args.target} is {state}.")
        return 0

    if patched == original:
        print("[inkling-sm12-paged-kv] Inkling dispatch is already patched; skipping.")
        return 0

    temporary = args.target.with_suffix(args.target.suffix + ".inkling-sm12-mod.tmp")
    temporary.write_text(patched)
    temporary.replace(args.target)
    print(f"[inkling-sm12-paged-kv] Patched {args.target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
