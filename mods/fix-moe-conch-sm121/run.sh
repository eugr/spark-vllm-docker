#!/bin/bash
set -e

# Patch GPTQMarlinMoEMethod to use conch Triton GEMM per-expert on SM121/GB10 Blackwell.
#
# Root cause: ops.moe_wna16_marlin_gemm (CUDA Marlin MoE GEMM) crashes on
# SM121/GB10 Blackwell with cudaErrorIllegalAddress. Both use_atomic_add=False
# and use_atomic_add=True paths fail — the CUDA kernel itself is broken for
# Blackwell's warp layout, not just the output accumulation step.
#
# Fix: On SM121, bypass moe_wna16_marlin_gemm entirely in GPTQMarlinMoEMethod by:
# 1. Skipping gptq_marlin_moe_repack in process_weights_after_loading — keeps
#    weights in standard GPTQ packed-INT4 format (not Marlin-repacked format).
# 2. Skipping marlin_moe_permute_scales — keeps scales in GPTQ layout.
# 3. In apply(), routing to _apply_conch_sm121() which uses per-expert calls
#    to conch mixed_precision_gemm (Triton-based, SM121-compatible).
#
# conch mixed_precision_gemm is the SAME Triton kernel used by ConchLinearKernel
# for non-MoE linear layers. It works correctly on SM121/GB10 Blackwell.
# ConchLinearKernel is already active for all linear layers (via install-conch mod).
#
# Weight format (GPTQ, before Marlin repack) matches conch expected format exactly:
#   w13_qweight[e]: (K//pack_factor, 2N)  packed INT4, gate+up proj
#   w13_scales[e]:  (K//group_size, 2N)   scale per K-group per output channel
#   w2_qweight[e]:  (N//pack_factor, K)   packed INT4, down proj
#   w2_scales[e]:   (N//group_size, K)    scale per N-group per output channel
# K=hidden_size, N=intermediate_size_per_partition, pack_factor=8 (for 4-bit),
# group_size=128 (for Intel AutoRound). Shapes confirmed against create_weights().
#
# Performance: per-expert loop is ~10-20x slower than fused MoE GEMM for large
# batches, but functionally correct. For profile_run (1 token) it is fast.
# Decode throughput: sufficient for Noah's 2 max_num_seqs configuration.

GPTQ_MARLIN=$(python3 -c "
import vllm.model_executor.layers.quantization.gptq_marlin as m
print(m.__file__)
" 2>/dev/null)

if [ -z "$GPTQ_MARLIN" ]; then
    echo "[fix-moe-conch-sm121] ERROR: cannot find gptq_marlin.py"
    exit 1
fi
echo "[fix-moe-conch-sm121] Patching $GPTQ_MARLIN"

python3 - "$GPTQ_MARLIN" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

if 'SM121/GB10 patch: conch per-expert MoE' in content:
    print("[fix-moe-conch-sm121] Already patched")
    sys.exit(0)

# -----------------------------------------------------------------------
# Patch 1: process_weights_after_loading — skip Marlin repack on SM121
#
# Insert an early-return BEFORE "# Repack weights" block so on SM121 the
# weights remain in GPTQ format (compatible with conch mixed_precision_gemm).
# -----------------------------------------------------------------------
old_repack_marker = '        # Repack weights\n        marlin_w13_qweight = ops.gptq_marlin_moe_repack('
if old_repack_marker not in content:
    print(f"[fix-moe-conch-sm121] ERROR: '# Repack weights' marker not found in {path}")
    sys.exit(1)

new_repack_block = (
    '        # SM121/GB10 patch: conch per-expert MoE — skip Marlin repack\n'
    '        # moe_wna16_marlin_gemm (CUDA) crashes on Blackwell. Keep weights in\n'
    '        # standard GPTQ format so conch mixed_precision_gemm can use them.\n'
    '        if torch.cuda.get_device_capability()[0] >= 12:  # SM121 patch\n'
    '            layer._moe_use_conch = True\n'
    '            layer._moe_group_size = self.quant_config.group_size\n'
    '            return  # Skip Marlin repack; weights stay in GPTQ packed-INT4 format\n'
    '        # Repack weights\n'
    '        marlin_w13_qweight = ops.gptq_marlin_moe_repack('
)
content = content.replace(old_repack_marker, new_repack_block, 1)
print("[fix-moe-conch-sm121] Patch 1 applied: SM121 early-return in process_weights_after_loading")

# -----------------------------------------------------------------------
# Patch 2: apply() — dispatch to _apply_conch_sm121 on SM121
#
# Insert SM121 check before "return fused_marlin_moe(" in apply().
# -----------------------------------------------------------------------
old_apply_return = '        return fused_marlin_moe(\n            x,'
if old_apply_return not in content:
    # Try single-line variant
    old_apply_return = '        return fused_marlin_moe(x,'
    if old_apply_return not in content:
        print(f"[fix-moe-conch-sm121] ERROR: 'return fused_marlin_moe' not found in apply()")
        idx = content.find('class GPTQMarlinMoEMethod(')
        apply_idx = content.find('    def apply(', idx)
        if apply_idx >= 0:
            print("apply() method start:")
            print(content[apply_idx:apply_idx+400])
        sys.exit(1)

new_apply_return = (
    '        # SM121/GB10 patch: bypass moe_wna16_marlin_gemm on Blackwell\n'
    '        if getattr(layer, \'_moe_use_conch\', False):\n'
    '            return self._apply_conch_sm121(layer, x, topk_weights, topk_ids)\n'
    + old_apply_return
)
content = content.replace(old_apply_return, new_apply_return, 1)
print("[fix-moe-conch-sm121] Patch 2 applied: SM121 dispatch in apply()")

# -----------------------------------------------------------------------
# Patch 3: add _apply_conch_sm121 method to GPTQMarlinMoEMethod
#
# Implement per-expert SiLU-gate MoE using conch mixed_precision_gemm.
# Inserted before the next top-level class definition (or at EOF).
# -----------------------------------------------------------------------
NEW_METHOD = '''
    def _apply_conch_sm121(
        self,
        layer: "torch.nn.Module",
        x: "torch.Tensor",
        topk_weights: "torch.Tensor",
        topk_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        """SM121/GB10 Blackwell: per-expert conch Triton INT4 GEMM.

        Replaces moe_wna16_marlin_gemm (CUDA, broken on SM121) with
        conch mixed_precision_gemm (Triton, SM121-compatible).
        Weights must be in GPTQ packed-INT4 format (not Marlin-repacked).

        SM121/GB10 patch: conch per-expert MoE
        """
        import torch.nn.functional as F
        from conch.ops.quantization.gemm import mixed_precision_gemm

        M, K = x.shape
        E = layer.w13_qweight.shape[0]           # num local experts
        N2 = layer.w13_qweight.shape[2]          # 2*N: gate + up channels
        N = N2 // 2                               # intermediate size per partition
        topk = topk_ids.shape[1]                 # top-k experts per token

        group_size = layer._moe_group_size
        if group_size == -1:
            group_size = K  # channel-wise: one group = full K

        weight_bits = self.quant_config.quant_type.size_bits  # 4 for INT4
        weight_bias = self.quant_type.bias                    # 8 for uint4b8

        w13_bias = getattr(layer, "w13_bias", None)
        w2_bias = getattr(layer, "w2_bias", None)

        output = torch.zeros(M, K, dtype=x.dtype, device=x.device)

        # Per-expert loop: select tokens for each (top-k slot, expert) pair.
        # For each selected group: gate+up proj → SiLU gate → down proj.
        for k_idx in range(topk):
            for e_idx in range(E):
                mask = topk_ids[:, k_idx] == e_idx
                if not mask.any():
                    continue

                x_e = x[mask].contiguous()            # (n, K)
                w = topk_weights[mask, k_idx]         # (n,) float32

                # Gate + Up projection: (n, K) -> (n, 2N)
                # w13_qweight[e]: (K//pack_factor, 2N) packed INT4
                # w13_scales[e]:  (K//group_size, 2N)
                out1 = mixed_precision_gemm(
                    x=x_e,
                    w_q_packed=layer.w13_qweight[e_idx].contiguous(),
                    w_s=layer.w13_scales[e_idx].contiguous(),
                    w_zp=None,
                    weight_size_bits=weight_bits,
                    weight_bias=weight_bias,
                    group_size=group_size,
                )

                if w13_bias is not None:
                    out1 = out1 + w13_bias[e_idx]

                # SiLU-gated activation: gate=out1[:,:N], up=out1[:,N:]
                inter = F.silu(out1[:, :N]) * out1[:, N:]  # (n, N)

                # Down projection: (n, N) -> (n, K)
                # w2_qweight[e]: (N//pack_factor, K) packed INT4
                # w2_scales[e]:  (N//group_size, K)
                down = mixed_precision_gemm(
                    x=inter.contiguous(),
                    w_q_packed=layer.w2_qweight[e_idx].contiguous(),
                    w_s=layer.w2_scales[e_idx].contiguous(),
                    w_zp=None,
                    weight_size_bits=weight_bits,
                    weight_bias=weight_bias,
                    group_size=group_size,
                )

                if w2_bias is not None:
                    down = down + w2_bias[e_idx]

                # Accumulate with routing weight (topk_weights is float32; cast to x.dtype)
                output[mask] += w.to(x.dtype).unsqueeze(1) * down

        return output

'''

# Insert before the next top-level class after GPTQMarlinMoEMethod, or at EOF
class_idx = content.find('class GPTQMarlinMoEMethod(')
if class_idx < 0:
    print("[fix-moe-conch-sm121] ERROR: GPTQMarlinMoEMethod class not found")
    sys.exit(1)

next_class_idx = content.find('\nclass ', class_idx + 10)
insert_idx = next_class_idx if next_class_idx >= 0 else len(content)

content = content[:insert_idx] + NEW_METHOD + content[insert_idx:]
print("[fix-moe-conch-sm121] Patch 3 applied: _apply_conch_sm121 method added")

with open(path, 'w') as f:
    f.write(content)
print(f"[fix-moe-conch-sm121] Wrote {len(content)} bytes to {path}")
PYEOF

# Syntax check and verification
python3 -c "
import ast, sys
path = '$GPTQ_MARLIN'
with open(path) as f:
    src = f.read()
try:
    ast.parse(src)
    print('[fix-moe-conch-sm121] Syntax check: OK')
except SyntaxError as e:
    print(f'[fix-moe-conch-sm121] SYNTAX ERROR after patch: {e}')
    sys.exit(1)

checks = [
    'SM121/GB10 patch: conch per-expert MoE',
    '_moe_use_conch',
    '_apply_conch_sm121',
    '_moe_group_size',
    'mixed_precision_gemm',
    'Skip Marlin repack',
]
all_ok = True
for c in checks:
    if c in src:
        print(f'[fix-moe-conch-sm121] Verified: {c!r} present')
    else:
        print(f'[fix-moe-conch-sm121] WARNING: {c!r} NOT found')
        all_ok = False
if all_ok:
    print('[fix-moe-conch-sm121] All verification checks passed')
"

echo "[fix-moe-conch-sm121] Done"
