#!/bin/bash
set -e

# Patch GemmaRMSNorm.forward_cuda to skip torch.compile on SM121/GB10 Blackwell.
#
# Root cause: torch.compile() wraps _forward_static_no_residual and
# _forward_static_with_residual in GemmaRMSNorm.forward_cuda. On first call,
# TorchDynamo compiles the frame and calls set_rng_state to restore RNG state
# after compilation. This is the first CUDA synchronization point after a
# Triton/Marlin kernel poisons the CUDA context on SM121, causing:
#   torch._dynamo.exc.InternalTorchDynamoError: AcceleratorError: CUDA error:
#   an illegal memory access was encountered
#
# Why TORCHDYNAMO_DISABLE=1 env var doesn't work: Ray workers (remote nodes)
# don't inherit TORCHDYNAMO_DISABLE from the head node. ray_env.py only copies
# VLLM_/NCCL_/HF_ prefixed vars, so TORCHDYNAMO_ vars are silently dropped.
#
# Fix: patch layernorm.py at code level to skip the torch.compile() calls.
# GemmaRMSNorm.forward_cuda still calls forward_native() which uses the custom
# CUDA kernel for RMSNorm, so there is no performance regression.

LAYERNORM=$(python3 -c "import vllm.model_executor.layers.layernorm as m; print(m.__file__)" 2>/dev/null)
if [ -z "$LAYERNORM" ]; then
    echo "[fix-torchdynamo-sm121] ERROR: cannot find layernorm.py"
    exit 1
fi
echo "[fix-torchdynamo-sm121] Patching $LAYERNORM"

python3 - "$LAYERNORM" <<'PYEOF'
import sys
import re

path = sys.argv[1]
with open(path) as f:
    content = f.read()

if 'SM121/GB10 patch: skip torch.compile' in content:
    print("[fix-torchdynamo-sm121] Already patched")
    sys.exit(0)

# Match the _is_compiled guard block that contains torch.compile calls.
# The actual vllm 0.18.3.dev17 code looks like:
#         if not getattr(self, "_is_compiled", False):
#             self._forward_static_no_residual = torch.compile(  # type: ignore
#                 self._forward_static_no_residual
#             )
#             self._forward_static_with_residual = torch.compile(  # type: ignore
#                 self._forward_static_with_residual
#             )
#             self._is_compiled = True
#
# Use a regex that matches the entire block regardless of number of compile calls.
old_pattern = re.compile(
    r'        if not getattr\(self, "_is_compiled", False\):\n'
    r'(?:            .*\n)*?'
    r'            self\._is_compiled = True',
    re.MULTILINE
)

new_block = (
    '        if not getattr(self, "_is_compiled", False):\n'
    '            # SM121/GB10 patch: skip torch.compile to avoid set_rng_state\n'
    '            # CUDA crash. TorchDynamo is not propagated to Ray workers\n'
    '            # (ray_env.py only copies VLLM_/NCCL_/HF_ prefixes).\n'
    '            self._is_compiled = True'
)

result, n = re.subn(old_pattern, new_block, content)
if n > 0:
    # Validate we didn't accidentally remove too much
    if 'def forward_cuda' in result and 'def forward_native' in result:
        with open(path, 'w') as f:
            f.write(result)
        print(f"[fix-torchdynamo-sm121] Patched GemmaRMSNorm.forward_cuda: removed {n} torch.compile block(s)")
    else:
        print("[fix-torchdynamo-sm121] ERROR: regex matched too broadly, aborting patch")
        sys.exit(1)
else:
    print(f"[fix-torchdynamo-sm121] ERROR: _is_compiled pattern not found in {path}")
    # Print context for debugging
    m = re.search(r'class GemmaRMSNorm.*?(?=\nclass )', content, re.DOTALL)
    if m:
        print("GemmaRMSNorm class content:")
        print(m.group(0)[:800])
    sys.exit(1)
PYEOF

# Verify patch: syntax check
python3 -c "
import ast, sys
path = '$LAYERNORM'
with open(path) as f:
    src = f.read()
try:
    ast.parse(src)
    print('[fix-torchdynamo-sm121] Syntax check: OK')
except SyntaxError as e:
    print(f'[fix-torchdynamo-sm121] SYNTAX ERROR after patch: {e}')
    sys.exit(1)
import re
if re.search(r'torch\.compile\(', src) and 'SM121' not in src:
    print('[fix-torchdynamo-sm121] WARNING: torch.compile still present and not explained by patch')
else:
    print('[fix-torchdynamo-sm121] Verification: torch.compile removed from _is_compiled block')
"

echo "[fix-torchdynamo-sm121] Done"
