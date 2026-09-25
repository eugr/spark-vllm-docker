#!/usr/bin/env python3
"""sm_121 (GB10) o-proj fix.

DeepSeek-V4 attention's output projection routes through
vllm.utils.deep_gemm.fp8_einsum (deep_gemm_fp8_o_proj). DeepGEMM's scale-factor
layout transform has no arch_major==12 branch and aborts with
't.dim() == N' (vLLM issue #41063); pristine v0.26.0's bundled DeepGEMM ships
no sm120/sm121 fp8 einsum at all. All three CUDA attention modules
(FlashMLA / FlashInfer / b12x) share this same o-proj, so it is the single
blocker for the forward pass on GB10.

This appends a bf16 fallback to the installed nvidia/ops/o_proj.py: when the
device is capability major==12 it reuses the model's OWN fused inverse-RoPE +
block-scaled fp8 quant triton kernel (arch-agnostic; already runs on sm_121),
dequantizes o and wo_a back to bf16, and performs the grouped o-proj GEMM with
torch.einsum -- bypassing DeepGEMM entirely. Numerically equivalent to the
intended fp8 path (identical fp8 rounding of o and wo_a); only the GEMM backend
differs. Idempotent; safe on non-sm12 (falls through to the original).

wo_a is a bmm ColumnParallelLinear: weight is stored 2D (n_groups*o_lora_rank,
R) and the original fp8_einsum reshaped it to (G, D, R) via bmm_batch_size --
we do that reshape explicitly here.
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(
    '/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py'
)
SENTINEL = '_sm121_bf16_o_proj'

APPEND = '''

# ---- sm_121 (GB10) bf16 o-proj fallback (appended by b12x-native-backend mod) ----
def _sm121_dequant(w, s):
    import torch
    w = w.to(torch.float32)
    s = s.to(torch.float32)
    for _d in range(w.dim()):
        _bs = w.shape[_d] // s.shape[_d]
        if _bs > 1:
            s = s.repeat_interleave(_bs, dim=_d)
        if s.shape[_d] > w.shape[_d]:
            s = s.narrow(_d, 0, w.shape[_d])
    return w * s


def _sm121_bf16_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, *,
                       n_groups, heads_per_group, nope_dim, rope_dim, o_lora_rank):
    import torch
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o, positions, cos_sin_cache,
        n_groups=n_groups, heads_per_group=heads_per_group,
        nope_dim=nope_dim, rope_dim=rope_dim, tma_aligned_scales=False,
    )
    o_bf16 = _sm121_dequant(o_fp8, o_scale)                      # (T, G, R)
    w_bf16 = _sm121_dequant(wo_a.weight, wo_a.weight_scale_inv)  # (G*D, R) 2D
    w_bf16 = w_bf16.reshape(n_groups, o_lora_rank, -1)           # (G, D, R)
    z = torch.einsum('tgr,gdr->tgd', o_bf16, w_bf16).to(torch.bfloat16)
    return wo_b(z.reshape(z.shape[0], -1))


_deep_gemm_fp8_o_proj_orig = deep_gemm_fp8_o_proj


def deep_gemm_fp8_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, *,
                         n_groups, heads_per_group, nope_dim, rope_dim,
                         o_lora_rank, einsum_recipe, tma_aligned_scales):
    _cap = current_platform.get_device_capability()
    if _cap is not None and _cap.major == 12:
        return _sm121_bf16_o_proj(
            o, positions, cos_sin_cache, wo_a, wo_b,
            n_groups=n_groups, heads_per_group=heads_per_group,
            nope_dim=nope_dim, rope_dim=rope_dim, o_lora_rank=o_lora_rank,
        )
    return _deep_gemm_fp8_o_proj_orig(
        o, positions, cos_sin_cache, wo_a, wo_b,
        n_groups=n_groups, heads_per_group=heads_per_group,
        nope_dim=nope_dim, rope_dim=rope_dim, o_lora_rank=o_lora_rank,
        einsum_recipe=einsum_recipe, tma_aligned_scales=tma_aligned_scales,
    )
'''


def main() -> int:
    if not TARGET.is_file():
        print(f'FAIL sm121-o-proj: {TARGET} not found')
        return 1
    text = TARGET.read_text()
    if SENTINEL in text:
        print('SKIP sm121-o-proj: already applied')
        return 0
    for name in ('deep_gemm_fp8_o_proj', 'fused_inv_rope_fp8_quant', 'current_platform'):
        if name not in text:
            print(f'FAIL sm121-o-proj: expected symbol {name!r} not in o_proj.py')
            return 1
    TARGET.write_text(text + APPEND)
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as e:
        print(f'COMPILE_ERROR sm121-o-proj: {e}')
        return 1
    pc = TARGET.parent / '__pycache__'
    if pc.is_dir():
        for f in pc.glob('o_proj*.pyc'):
            f.unlink()
    print('OK sm121-o-proj: bf16 fallback appended')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
