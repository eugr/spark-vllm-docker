#!/usr/bin/env python3
"""Quantize selected BF16 layers excluded by nvidia/GLM-5.3-Flash-NVFP4 at load time.
Targets KDA/MLA attention and optional shared experts using online MXFP8, with
optional NVFP4 overrides. MXFP8 defaults to BF16 activations (B12X W8A16).

Anchor: ModelOptQuantConfigBase.get_quant_method, "# handle exclusion" branch.
Idempotent; --check validates anchors without writing.
"""
from __future__ import annotations
import argparse, ast, sys
from pathlib import Path

MOD = "glm53-nvidia-online-mxfp8"
REL = "model_executor/layers/quantization/modelopt.py"

OLD = '''        # handle exclusion
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None
'''
NEW = '''        # handle exclusion
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                _g53_rt = _glm53_online_mxfp8_method(layer, prefix)  # glm53-nvidia-online-mxfp8
                if _g53_rt is not None:
                    return _g53_rt
                return UnquantizedLinearMethod()
            _g53_moe = _glm53_online_mxfp8_moe(layer, prefix)  # glm53-nvidia-online-mxfp8
            if _g53_moe is not None:
                return _g53_moe
            return None
'''
HELPER = '''

# --- glm53-nvidia-online-mxfp8 (mod) ---------------------------------------------
import os as _g53_os
import re as _g53_re

_G53_TARGETS = {
    # Dense attention projections: KDA (fused in_proj_qkvgfab, o_proj) and MLA
    "attn": r"\\.self_attn\\.(in_proj_qkvgfab|q_proj|k_proj|v_proj|qkv_proj|o_proj|"
            r"q_a_proj|q_b_proj|fused_qkv_a_proj|kv_a_proj_with_mqa)$",
    "shared": r"\\.mlp\\.shared_experts\\.(gate_up_proj|down_proj)$",
    # MTP layer (45): dense projections; experts use _glm53_online_mxfp8_moe
    "mtp": r"(^|\\.)layers\\.45\\..*\\.(in_proj_qkvgfab|q_proj|k_proj|v_proj|qkv_proj|o_proj|"
           r"q_a_proj|q_b_proj|fused_qkv_a_proj|kv_a_proj_with_mqa|gate_up_proj|down_proj|eh_proj)$",
}
# Always skip MLA absorption, sparse indexer, router, heads/embeddings, vision
_G53_SKIP = r"(kv_b_proj|indexer|lm_head|embed|\\.mlp\\.gate$|visual|vision)"
_G53_MTP = r"(^|\\.)layers\\.45\\."
_g53_logged = set()


def _glm53_online_mxfp8_method(layer, prefix):
    mode = _g53_os.getenv("VLLM_GLM53_ONLINE_MXFP8", "0").strip().lower()
    if mode in ("", "0", "off", "none"):
        return None
    if isinstance(layer, ParallelLMHead):
        return None
    if _g53_re.search(_G53_SKIP, prefix):
        return None
    parts = [p for p in mode.replace("+", ",").split(",") if p]
    if _g53_re.search(_G53_MTP, prefix) and "mtp" not in parts:
        return None
    if not any(_g53_re.search(_G53_TARGETS[p], prefix) for p in parts if p in _G53_TARGETS):
        return None
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineLinearMethod,
    )
    a16 = _g53_os.getenv("VLLM_GLM53_ONLINE_MXFP8_A16", "1") != "0"
    kind = _g53_re.sub(r"^.*layers\\.\\d+\\.", "", prefix)
    # NVFP4 (16-weight blocks, FP8 scales) for groups in VLLM_GLM53_ONLINE_NVFP4:
    #   shared = shared experts | oproj = o_proj | qkv = remaining attention projections
    nv = [x for x in _g53_os.getenv("VLLM_GLM53_ONLINE_NVFP4", "").lower().replace("+", ",").split(",") if x]
    if nv:
        if _g53_re.search(_G53_TARGETS["shared"], prefix):
            group = "shared"
        elif prefix.endswith(".o_proj"):
            group = "oproj"
        else:
            group = "qkv"
        if group in nv:
            from vllm.model_executor.layers.quantization.online.nvfp4 import (
                Nvfp4OnlineLinearMethod,
            )
            nv_a16 = _g53_os.getenv("VLLM_GLM53_ONLINE_NVFP4_A16", "1") != "0"
            if ("nv:" + kind) not in _g53_logged:
                _g53_logged.add("nv:" + kind)
                logger.info(
                    "[GLM53-ONLINE-NVFP4] %s -> NVFP4 online (%s activations)",
                    kind, "BF16" if nv_a16 else "FP4",
                )
            return Nvfp4OnlineLinearMethod(use_a16=nv_a16)
    if kind not in _g53_logged:
        _g53_logged.add(kind)
        logger.info(
            "[GLM53-ONLINE-MXFP8] %s -> MXFP8 online (%s activations)",
            kind, "BF16" if a16 else "MXFP8",
        )
    return Mxfp8OnlineLinearMethod(use_a16=a16)


def _glm53_online_mxfp8_moe(layer, prefix):
    mode = _g53_os.getenv("VLLM_GLM53_ONLINE_MXFP8", "0").strip().lower()
    if "mtp" not in mode.replace("+", ",").split(","):
        return None
    if not isinstance(layer, RoutedExperts) or not _g53_re.search(_G53_MTP, prefix):
        return None
    from vllm.model_executor.layers.quantization.online.mxfp8 import (
        Mxfp8OnlineMoEMethod,
    )
    if "mtp-experts" not in _g53_logged:
        _g53_logged.add("mtp-experts")
        logger.info("[GLM53-ONLINE-MXFP8] %s -> MXFP8 online MoE (expertos MTP)", prefix)
    return Mxfp8OnlineMoEMethod(moe=layer.moe_config)
'''


class PatchError(RuntimeError):
    pass


def patch(src: str) -> str:
    if MOD in src:
        if src.count(NEW) == 1 and "_glm53_online_mxfp8_moe(layer, prefix)" in src:
            return src
        raise PatchError("partial or modified patch; refusing to continue")
    if src.count(OLD) != 1:
        raise PatchError(f"ANCHOR FAILED: exclusion branch found {src.count(OLD)} times")
    if "\nlogger = " not in src:
        raise PatchError("ANCHOR FAILED: modelopt.py has no module logger")
    out = src.replace(OLD, NEW, 1) + HELPER
    ast.parse(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    t = Path(a.vllm_root) / REL
    src = t.read_text()
    out = patch(src)
    if a.check:
        print(f"[{MOD}] check: {'already applied' if out == src else 'anchors OK'} (no writes)")
        return 0
    if out == src:
        print(f"[{MOD}] already applied (no-op)")
        return 0
    t.write_text(out)
    print(f"[{MOD}] applied: {REL}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PatchError, SyntaxError) as e:
        print(f"[{MOD}] FAILED: {e}", file=sys.stderr)
        raise SystemExit(1)
