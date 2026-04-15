#!/bin/bash
set -e

# Patch MRotaryEmbedding.forward_cuda to skip triton_mrope on SM121/GB10 Blackwell.
#
# Root cause: triton_mrope() loads a Triton JIT kernel via _init_handles → load_binary.
# On SM121, this is the first CUDA synchronization point after a prior kernel
# (in FLA/GDN linear_attention layers 0-2 or ConchLinearKernel qkv_proj in layer 3)
# poisons the CUDA context, causing:
#   RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
#
# Why this is an async CUDA error: Triton/Marlin/FLA kernels can return to Python
# before the GPU fault is raised. The fault surfaces at the next CUDA sync point —
# here, _init_handles → load_binary when Triton compiles/loads the mrope kernel.
#
# Fix: on SM121 (device capability >= 12), route MRoPE through forward_native
# (pure PyTorch) instead of the Triton kernel. Mathematically equivalent output.
# Only affects the positions.ndim == 2 (multimodal MRoPE) code path.
# Qwen3.5-397B text-only inference always uses positions.ndim == 2 due to the
# multimodal-capable architecture (MRotaryEmbedding is always used).

MROPE=$(python3 -c "
import vllm.model_executor.layers.rotary_embedding.mrope as m
print(m.__file__)
" 2>/dev/null)

if [ -z "$MROPE" ]; then
    echo "[fix-mrope-sm121] ERROR: cannot find mrope.py"
    exit 1
fi
echo "[fix-mrope-sm121] Patching $MROPE"

python3 - "$MROPE" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    content = f.read()

if 'SM121/GB10 patch: triton_mrope' in content:
    print("[fix-mrope-sm121] Already patched")
    sys.exit(0)

# Match the triton_mrope call block inside forward_cuda (positions.ndim == 2 branch).
# The actual vllm 0.18.3.dev17 code is:
#             assert self.mrope_section
#
#             q, k = triton_mrope(
#                 query,
#                 key,
#                 cos,
#                 sin,
#                 self.mrope_section,
#                 self.head_size,
#                 self.rotary_dim,
#                 self.mrope_interleaved,
#             )
#
#             return q.reshape(query_shape), k.reshape(key_shape)
old = (
    '            assert self.mrope_section\n'
    '\n'
    '            q, k = triton_mrope(\n'
    '                query,\n'
    '                key,\n'
    '                cos,\n'
    '                sin,\n'
    '                self.mrope_section,\n'
    '                self.head_size,\n'
    '                self.rotary_dim,\n'
    '                self.mrope_interleaved,\n'
    '            )\n'
    '\n'
    '            return q.reshape(query_shape), k.reshape(key_shape)'
)

new = (
    '            assert self.mrope_section\n'
    '            # SM121/GB10 patch: triton_mrope Triton kernel crashes on Blackwell\n'
    '            # with illegal memory access (async CUDA error from FLA/GDN/Marlin\n'
    '            # kernel surfaces at first Triton kernel load_binary call).\n'
    '            # Use forward_native (pure PyTorch) instead — same math output.\n'
    '            if torch.cuda.get_device_capability()[0] >= 12:  # SM121 patch\n'
    '                return self.forward_native(positions, query, key, offsets)\n'
    '\n'
    '            q, k = triton_mrope(\n'
    '                query,\n'
    '                key,\n'
    '                cos,\n'
    '                sin,\n'
    '                self.mrope_section,\n'
    '                self.head_size,\n'
    '                self.rotary_dim,\n'
    '                self.mrope_interleaved,\n'
    '            )\n'
    '\n'
    '            return q.reshape(query_shape), k.reshape(key_shape)'
)

if old in content:
    result = content.replace(old, new, 1)
    # Validate key symbols are still present
    if 'def forward_cuda' in result and 'def forward_native' in result and 'triton_mrope' in result:
        with open(path, 'w') as f:
            f.write(result)
        print("[fix-mrope-sm121] Patched MRotaryEmbedding.forward_cuda: added SM121 forward_native fallback")
    else:
        print("[fix-mrope-sm121] ERROR: patch removed critical symbols, aborting")
        sys.exit(1)
else:
    print(f"[fix-mrope-sm121] ERROR: triton_mrope pattern not found in {path}")
    # Print context for debugging
    import re
    m = re.search(r'def forward_cuda.*?(?=\n    def |\Z)', content, re.DOTALL)
    if m:
        print("forward_cuda method content:")
        print(m.group(0)[:1000])
    sys.exit(1)
PYEOF

# Verify patch: syntax check
python3 -c "
import ast, sys
path = '$MROPE'
with open(path) as f:
    src = f.read()
try:
    ast.parse(src)
    print('[fix-mrope-sm121] Syntax check: OK')
except SyntaxError as e:
    print(f'[fix-mrope-sm121] SYNTAX ERROR after patch: {e}')
    sys.exit(1)
if 'SM121/GB10 patch: triton_mrope' in src:
    print('[fix-mrope-sm121] Verification: SM121 fallback present in mrope.py')
else:
    print('[fix-mrope-sm121] WARNING: patch string not found after applying')
"

echo "[fix-mrope-sm121] Done"
