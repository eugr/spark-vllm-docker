#!/usr/bin/env python3
"""Runtime patches for DSpark MTP + REAP K160 on vllm/vllm-openai:v0.26.0.

1. REAP K160: router fallback (160 experts), MXFP4 memory hygiene, cutedsl, IPC
2. b12x MoE expert file (v0.26.0-only missing piece)
3. nvfp4_ds_mla KV cache dtype
4. b12x FP8 linear kernel enablement
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
DSPARK_OVERLAY = Path(__file__).resolve().parent / "dspark_overlay/vllm"


def _patch_text(target: Path, old: str, new: str, label: str) -> str:
    if not target.is_file():
        return f"SKIP  {label}: not found"
    text = target.read_text()
    if old not in text:
        if new in text:
            return f"SKIP  {label}: already applied"
        return f"FAIL  {label}: old text not found"
    n = text.count(old)
    if n > 1:
        return f"FAIL  {label}: {n} ambiguous matches"
    target.write_text(text.replace(old, new, 1))
    return f"OK    {label}"


def _patch_re(target: Path, pattern: re.Pattern, repl: str, label: str) -> str:
    if not target.is_file():
        return f"SKIP  {label}: not found"
    text = target.read_text()
    new_text, n = pattern.subn(repl, text)
    if n == 0:
        if new_text == text:
            return f"SKIP  {label}: already applied"
        return f"FAIL  {label}: no match"
    target.write_text(new_text)
    return f"OK    {label}: {n} site(s)"


def _compile(target: Path):
    import py_compile
    try:
        py_compile.compile(str(target), doraise=True)
    except py_compile.PyCompileError as e:
        raise SystemExit(f"COMPILE_ERROR {target.name}: {e}")


SUPPORTED = "(16, 32, 64, 128, 192, 256, 320, 384, 512)"


def _reap_router() -> list[str]:
    t = VLLM / "model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
    r = []
    r.append(_patch_text(t,
        '    if not rocm_aiter_ops.is_fused_moe_enabled():',
        '    # REAP K160: nonstandard expert counts need pure-Torch fallback\n'
        '    if not rocm_aiter_ops.is_fused_moe_enabled() or (\n'
        f'        scoring_func == "sqrtsoftplus"\n'
        f'        and gating_output.shape[-1] not in {SUPPORTED}\n'
        '    ):',
        "reap-router:gate"))
    r.append(_patch_text(t,
        '        elif scoring_func == "sqrtsoftplus":\n'
        '            return vllm_topk_softplus_sqrt(',
        '        elif scoring_func == "sqrtsoftplus":\n'
        f'            if gating_output.shape[-1] not in {SUPPORTED}:\n'
        '                return _topk_softplus_sqrt_torch(\n'
        '                    topk_weights,\n'
        '                    topk_ids,\n'
        '                    token_expert_indices,\n'
        '                    gating_output,\n'
        '                    renormalize,\n'
        '                    e_score_correction_bias,\n'
        '                    input_tokens,\n'
        '                    hash_indices_table,\n'
        '                    routed_scaling_factor,\n'
        '                )\n'
        '            return vllm_topk_softplus_sqrt(',
        "reap-router:sqrt"))
    _compile(t)
    return r


def _mxfp4_hygiene() -> list[str]:
    """Add gc.collect + empty_cache after each MoE layer's process_weights."""
    t = VLLM / "model_executor/layers/quantization/mxfp4.py"
    r = []
    hygiene = (
        "\n"
        "        # REAP K160: free cached unified memory after each MoE layer\n"
        "        import torch as _t, gc as _gc\n"
        "        _gc.collect()\n"
        "        _t.cuda.empty_cache()\n"
        "        print('[k160-moe-patch] layer setup alloc=%.1fGB reserved=%.1fGB'\n"
        "              % (_t.cuda.memory_allocated() / 1e9, _t.cuda.memory_reserved() / 1e9), flush=True)"
    )
    # Use the full process_weights_after_loading body (without type annotation)
    # to disambiguate from the annotated variant. v0.26.0 has two copies.
    func_body = (
        "    def process_weights_after_loading(self, layer):\n"
        "        w13 = layer.w13_weight\n"
        "        w2 = layer.w2_weight\n"
        "        w13_scale = layer.w13_weight_scale\n"
        "        w2_scale = layer.w2_weight_scale\n"
        '        w13_bias = getattr(layer, "w13_bias", None)\n'
        '        w2_bias = getattr(layer, "w2_bias", None)\n'
        "\n"
        "        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:\n"
        "            return\n"
        "\n"
        "        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)"
    )
    if t.is_file():
        text = t.read_text()
        if "torch.cuda.empty_cache()" in text:
            r.append("SKIP  mxfp4-hygiene: already applied")
            return r
        if func_body in text:
            t.write_text(text.replace(func_body, func_body + hygiene, 1))
            r.append("OK    mxfp4-hygiene")
            _compile(t)
        else:
            r.append("FAIL  mxfp4-hygiene: function body not found")
    else:
        r.append(f"SKIP  mxfp4-hygiene: {t} not found")
    return r


def _cutedsl_fallback() -> list[str]:
    """Add K160_DISABLE_CUTEDSL env var support to has_cutedsl()."""
    t = VLLM / "utils/import_utils.py"
    r = []
    old = (
        'def has_cutedsl() -> bool:\n'
        '    """Whether the optional `cutelass` package is available."""\n'
        '    return _has_module("cutlass")'
    )
    new = (
        'def has_cutedsl() -> bool:\n'
        '    """Whether the optional `cutlass` package is available."""\n'
        '    import os as _os\n'
        '    if _os.environ.get("K160_DISABLE_CUTEDSL", "0") == "1":\n'
        '        return False\n'
        '    return _has_module("cutlass")'
    )
    r.append(_patch_text(t, old, new, "reap-cutedsl"))
    _compile(t)
    return r


def _flashinfer_ipc() -> list[str]:
    """Fix TileLang libcudart_stub vs real CUDA runtime conflict."""
    t = Path("/usr/local/lib/python3.12/dist-packages/flashinfer/comm/cuda_ipc.py")
    r = []
    old = (
        '        if so_file is None:\n'
        '            so_file = find_loaded_library("libcudart")\n'
        '            assert so_file is not None, "libcudart is not loaded in the current process"'
    )
    new = (
        '        if so_file is None:\n'
        '            loaded_cudart = find_loaded_library("libcudart")\n'
        '            import os as _os\n'
        '            preferred = "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13"\n'
        '            if loaded_cudart and "libcudart_stub" not in loaded_cudart:\n'
        '                so_file = loaded_cudart\n'
        '            elif _os.path.exists(preferred):\n'
        '                so_file = preferred\n'
        '            else:\n'
        '                so_file = loaded_cudart\n'
        '            assert so_file is not None, "libcudart is not loaded in the current process"'
    )
    r.append(_patch_text(t, old, new, "flashinfer-cuda-ipc"))
    return r


DSPARK_OVERLAY_FILES = [
    "model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py",
    "models/deepseek_v4/nvidia/flashinfer_sparse.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/attention/backends/mla/flashmla_sparse.py",
    "models/deepseek_v4/attention.py",
    "models/deepseek_v4/sparse_mla.py",
]

def _overlay_dspark_critical() -> list[str]:
    """Copy critical DSpark-specific files from overlay into vLLM package."""
    r = []
    mounted = DSPARK_OVERLAY.is_dir()
    for rel in DSPARK_OVERLAY_FILES:
        src = DSPARK_OVERLAY / rel
        dst = VLLM / rel
        if not src.is_file():
            r.append(f"SKIP  overlay:{rel}: overlay src not found")
            continue
        if not mounted:
            r.append(f"SKIP  overlay:{rel}: overlay not mounted")
            continue
        overlay_rel = str(dst).replace(str(VLLM), "")
        r.append(f"OK    overlay:{overlay_rel}")
        dst.write_text(src.read_text())
    return r


def _nvfp4_ds_mla() -> list[str]:
    """Add nvfp4_ds_mla to CacheDType, kv_cache_interface, and torch_utils."""
    r = []

    # cache.py: add nvfp4_ds_mla to CacheDType
    t = VLLM / "config/cache.py"
    r.append(_patch_text(t,
        '    "nvfp4",\n',
        '    "nvfp4",\n    "nvfp4_ds_mla",\n',
        "nvfp4_ds_mla:cache.py"))
    _compile(t)

    # kv_cache_interface.py: map dtype to quant mode and page size
    t2 = VLLM / "v1/kv_cache_interface.py"
    r.append(_patch_text(t2,
        '    if kv_cache_dtype == "nvfp4":',
        '    if kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla"):',
        "nvfp4_ds_mla:get_kv_quant_mode"))
    r.append(_patch_text(t2,
        'if self.cache_dtype_str == "fp8_ds_mla":',
        'if self.cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):',
        "nvfp4_ds_mla:page_size_bytes"))
    r.append(_patch_text(t2,
        'self.cache_dtype_str == "fp8_ds_mla":',
        'self.cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):',
        "nvfp4_ds_mla:real_page_size"))

    # torch_utils.py: add nvfp4_ds_mla dtype mapping
    t3 = VLLM / "utils/torch_utils.py"
    r.append(_patch_text(t3,
        '    "fp8_ds_mla": torch.uint8,\n',
        '    "fp8_ds_mla": torch.uint8,\n    "nvfp4_ds_mla": torch.uint8,\n',
        "nvfp4_ds_mla:torch_utils"))

    return r


SOFT_FALLBACK_RE = re.compile(
    r"""        if not filtered:\n"""
    r"""            raise ValueError\(\n"""
    r"""                f"--linear-backend=\{linear_backend\} was requested but no "\n"""
    r"""                f"'\{\}\\{linear_backend\}' kernel exists for[^"]*"\n"""
    r"""            \)\n"""
    r"""        (?:platform_kernels|possible) = filtered""",
    re.MULTILINE,
)

SOFT_FALLBACK_REPL = (
    "        if filtered:\n"
    "            \\g<0> = filtered\n"
    "        else:\n"
    "            from vllm.logger import init_logger\n"
    "            _linear_logger = init_logger(__name__)\n"
    "            _linear_logger.warning_once(\n"
    '                "--linear-backend=%s has no kernel for this layer type; "\n'
    '                "falling back to auto selection.",\n'
    "                linear_backend,\n"
    "            )"
)


def _b12x_linear() -> list[str]:
    r = []
    t = VLLM / "model_executor/kernels/linear/__init__.py"
    r.append(_patch_re(t, SOFT_FALLBACK_RE, SOFT_FALLBACK_REPL, "b12x-linear:soft-fallback"))
    r.append(_patch_text(t,
        '        FlashInferCuteDslNvFp4LinearKernel,\n'
        '        # FlashInferB12xNvFp4LinearKernel excluded from auto-selection until\n'
        '        # upstream CUTLASS SM121 MMA op guard is resolved; use\n'
        '        # --linear-backend flashinfer_b12x to opt in explicitly.\n'
        '        FlashInferCutlassNvFp4LinearKernel,',
        '        FlashInferCuteDslNvFp4LinearKernel,\n'
        '        FlashInferB12xNvFp4LinearKernel,\n'
        '        FlashInferCutlassNvFp4LinearKernel,',
        "b12x-linear:nvfp4-kernel"))
    return r


def _b12x_moe_wire() -> list[str]:
    """Wire the overlaid B12xExperts class into the MXFP4 oracle so
    --moe-backend b12x actually selects it. b12x_mxfp4_moe.py (dropped in by
    _overlay_dspark_critical) defines B12xExperts but stock v0.26.0's oracle
    has no enum value, kernel_cls branch, or moe_backend string mapping for
    it — this adds the same three edits vLLM already has for every other
    vendor backend (see e.g. Mxfp4MoeBackend.MARLIN)."""
    t = VLLM / "model_executor/layers/fused_moe/oracle/mxfp4.py"
    r = []
    r.append(_patch_text(t,
        '    # Marlin\n'
        '    BATCHED_MARLIN = "BATCHED_MARLIN"\n'
        '    MARLIN = "MARLIN"\n',
        '    # Marlin\n'
        '    BATCHED_MARLIN = "BATCHED_MARLIN"\n'
        '    MARLIN = "MARLIN"\n'
        '    # B12X (DeepSeek V4 native MXFP4, ported from vllm-starter dspark)\n'
        '    B12X = "B12X"\n',
        "b12x-moe:enum"))
    r.append(_patch_text(t,
        '    elif backend == Mxfp4MoeBackend.EMULATION:\n'
        '        from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (\n'
        '            OCP_MXQuantizationEmulationTritonExperts,\n'
        '        )\n'
        '\n'
        '        return [OCP_MXQuantizationEmulationTritonExperts]\n'
        '\n'
        '    else:\n'
        '        raise ValueError(f"Unknown MXFP4 MoE backend: {backend.value}")',
        '    elif backend == Mxfp4MoeBackend.EMULATION:\n'
        '        from vllm.model_executor.layers.fused_moe.experts.ocp_mx_emulation_moe import (\n'
        '            OCP_MXQuantizationEmulationTritonExperts,\n'
        '        )\n'
        '\n'
        '        return [OCP_MXQuantizationEmulationTritonExperts]\n'
        '\n'
        '    elif backend == Mxfp4MoeBackend.B12X:\n'
        '        from vllm.model_executor.layers.fused_moe.experts.b12x_mxfp4_moe import (\n'
        '            B12xExperts,\n'
        '        )\n'
        '\n'
        '        return [B12xExperts]\n'
        '\n'
        '    else:\n'
        '        raise ValueError(f"Unknown MXFP4 MoE backend: {backend.value}")',
        "b12x-moe:kernel_cls"))
    r.append(_patch_text(t,
        '        "emulation": [Mxfp4MoeBackend.EMULATION],\n'
        '    }',
        '        "emulation": [Mxfp4MoeBackend.EMULATION],\n'
        '        "b12x": [Mxfp4MoeBackend.B12X],\n'
        '    }',
        "b12x-moe:mapping"))
    _compile(t)
    return r


def main() -> int:
    results = []

    patches = [
        _overlay_dspark_critical,
        _reap_router,
        _mxfp4_hygiene,
        _cutedsl_fallback,
        _flashinfer_ipc,
        _nvfp4_ds_mla,
        _b12x_linear,
        _b12x_moe_wire,
    ]

    for fn in patches:
        for result in fn():
            print(result, flush=True)
            results.append(result)

    ok = sum(1 for r in results if r.startswith("OK"))
    fail = sum(1 for r in results if r.startswith("FAIL"))
    print(f"\nPATCHES ok={ok} fail={fail} total={len(results)}", flush=True)
    # Only critical patches cause exit 1: router + mxfp4 hygiene + b12x experts
    critical_labels = ("reap-router", "mxfp4-hygiene", "b12x-experts")
    critical_fail = any(
        r.startswith("FAIL") and any(c in r for c in critical_labels)
        for r in results
    )
    return 1 if critical_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
