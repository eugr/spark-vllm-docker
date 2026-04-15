#!/bin/bash
set -e

# Patch _fused_marlin_moe to use use_atomic_add=True on SM121/GB10 Blackwell.
#
# Root cause: ops.moe_wna16_marlin_gemm with use_atomic_add=False crashes on
# SM121 with an illegal memory access (async CUDA error). The workspace-based
# output accumulation path (use_atomic_add=False) has an SM121-specific bug,
# likely in the workspace indexing or write pattern for Blackwell's warp layout.
#
# The error is ASYNC — moe_wna16_marlin_gemm returns to Python before the GPU
# fault is raised. The fault surfaces at the next CUDA synchronization point
# downstream:
#   - tf5 (0.18.1rc1): torch.sum(moe_output, out=output) [sync via out= param]
#   - noah-patched (0.18.3.dev17): propagates past moe_sum() through
#     set_rng_state (TorchDynamo) → triton_mrope (load_binary) → torch.cat
#     (forward_native in common.py:176 — cudaMalloc triggers CUDA error flush)
#
# Fix: on SM121 (device capability >= 12), use the atomic add code path
# (use_atomic_add=True). The atomic add variant accumulates results directly
# into the output tensor using atomicAdd CUDA ops instead of workspace-based
# reduction, bypassing the buggy code path.
#
# _fused_marlin_moe is called by both fused_marlin_moe (regular MoE) and
# batched_fused_marlin_moe (decode-phase dispatch), so this patch covers both.
# use_atomic_add=False is hardcoded twice in _fused_marlin_moe; VLLM_MARLIN_USE_ATOMIC_ADD
# env var has no effect on this code path (the env var controls a different check).

MARLIN_MOE=$(python3 -c "
import vllm.model_executor.layers.fused_moe.fused_marlin_moe as m
print(m.__file__)
" 2>/dev/null)

if [ -z "$MARLIN_MOE" ]; then
    echo "[fix-marlin-moe-sm121] ERROR: cannot find fused_marlin_moe.py"
    exit 1
fi
echo "[fix-marlin-moe-sm121] Patching $MARLIN_MOE"

python3 - "$MARLIN_MOE" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

if 'SM121/GB10 patch: moe_wna16_marlin_gemm' in content:
    print("[fix-marlin-moe-sm121] Already patched")
    sys.exit(0)

# Step 1: Insert SM121 device capability check at start of _fused_marlin_moe body.
# Match the unique assert that begins the function body.
old_assert = (
    '    assert hidden_states.ndim == 2\n'
    '    M, K = hidden_states.size()'
)
new_assert = (
    '    assert hidden_states.ndim == 2\n'
    '    # SM121/GB10 patch: moe_wna16_marlin_gemm use_atomic_add=False crashes on\n'
    '    # Blackwell with illegal memory access. Use atomic add path on SM121.\n'
    '    _sm121_atomic = torch.cuda.get_device_capability()[0] >= 12  # SM121 patch\n'
    '    M, K = hidden_states.size()'
)

if old_assert not in content:
    print(f"[fix-marlin-moe-sm121] ERROR: assert pattern not found in {path}")
    import re
    m = re.search(r'def _fused_marlin_moe.*?(?=\ndef |\Z)', content, re.DOTALL)
    if m:
        print("_fused_marlin_moe content (first 500 chars):")
        print(m.group(0)[:500])
    sys.exit(1)

content = content.replace(old_assert, new_assert, 1)

# Step 2: Replace both hardcoded use_atomic_add=False with the SM121-aware variable.
# There are exactly 2 occurrences in _fused_marlin_moe (w1 GEMM and w2 GEMM).
old_atomic = '        use_atomic_add=False,'
new_atomic = '        use_atomic_add=_sm121_atomic,  # SM121 patch (was False)'
n = content.count(old_atomic)
if n == 0:
    print("[fix-marlin-moe-sm121] ERROR: use_atomic_add=False pattern not found")
    sys.exit(1)

content = content.replace(old_atomic, new_atomic)

with open(path, 'w') as f:
    f.write(content)
print(f"[fix-marlin-moe-sm121] Patched {n} use_atomic_add=False → _sm121_atomic in _fused_marlin_moe")
PYEOF

# Verify patch: syntax check + confirmation
python3 -c "
import ast, sys
path = '$MARLIN_MOE'
with open(path) as f:
    src = f.read()
try:
    ast.parse(src)
    print('[fix-marlin-moe-sm121] Syntax check: OK')
except SyntaxError as e:
    print(f'[fix-marlin-moe-sm121] SYNTAX ERROR after patch: {e}')
    sys.exit(1)
if 'SM121/GB10 patch: moe_wna16_marlin_gemm' in src and '_sm121_atomic' in src:
    print('[fix-marlin-moe-sm121] Verification: SM121 atomic add patch present')
else:
    print('[fix-marlin-moe-sm121] WARNING: patch verification failed')
import re
count = src.count('use_atomic_add=False')
if count > 0:
    print(f'[fix-marlin-moe-sm121] WARNING: {count} unreplaced use_atomic_add=False remain')
else:
    print('[fix-marlin-moe-sm121] Verification: all use_atomic_add=False replaced')
"

echo "[fix-marlin-moe-sm121] Done"
