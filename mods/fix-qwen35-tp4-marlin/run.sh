#!/bin/bash
# Fix: replace in_proj_ba MergedColumnParallelLinear with two ReplicatedLinear
# modules (in_proj_b, in_proj_a) for Qwen3.5 (gqa_interleaved_layout=False).
#
# ROOT CAUSE:
#   MergedColumnParallelLinear with output_sizes=[64,64] and TP=4 gives each rank
#   N=32. ConchLinearKernel (Triton) has BLOCK_N=64 with an UNMASKED scale load:
#   it reads 64 scale values even when the tensor only has 32 columns. At the last
#   K-group the read crosses the tensor boundary → cudaErrorIllegalAddress.
#
# FIX:
#   Use separate ReplicatedLinear(output_size=64) for in_proj_b and in_proj_a.
#   No TP split → N=64 on every rank → BLOCK_N=64 safe.
#   Checkpoint keys in_proj_b.qweight / in_proj_a.qweight load directly (no
#   stacked_params_mapping entry needed).
#
# TARGETS: /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py
#
# NOTE: qwen3_next.py / qwen3_5.py do NOT need changes. In vllm 0.18.3.dev17+,
#       the GDN layer is in gdn_linear_attn.py (refactored from qwen3_next.py).
#       Old patch files (qwen3_next.patch, qwen3_5.patch) targeted the wrong file.

set -e
MOD_DIR="$(dirname "$0")"
GDN_FILE="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py"

echo "[fix-qwen35-tp4-marlin] Patching $GDN_FILE ..."

python3 - "$GDN_FILE" <<'PYEOF'
import sys, ast

path = sys.argv[1]
with open(path) as f:
    content = f.read()

SENTINEL = "# SM121/GB10 ReplicatedLinear fix applied"
if SENTINEL in content:
    print("[fix-qwen35-tp4-marlin] Already applied, skipping")
    sys.exit(0)

# ──────────────────────────────────────────────────────────────────
# 1. Add ReplicatedLinear to the linear import block
# ──────────────────────────────────────────────────────────────────
OLD_IMPORT = (
    "    MergedColumnParallelLinear,\n"
    "    RowParallelLinear,\n"
)
NEW_IMPORT = (
    "    MergedColumnParallelLinear,\n"
    "    ReplicatedLinear,\n"
    "    RowParallelLinear,\n"
)
if OLD_IMPORT not in content:
    print(f"[fix-qwen35-tp4-marlin] ERROR: import pattern not found in {path}")
    sys.exit(1)
content = content.replace(OLD_IMPORT, NEW_IMPORT, 1)
print("[fix-qwen35-tp4-marlin] Step 1: Added ReplicatedLinear import")

# ──────────────────────────────────────────────────────────────────
# 2. Replace unconditional create_ba_proj() call with conditional
#    ReplicatedLinear for Qwen3.5 (gqa_interleaved_layout=False)
# ──────────────────────────────────────────────────────────────────
OLD_INIT = (
    "        # ba_proj doesn't support blockwise fp8 quantization.\n"
    "        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint\n"
    "        # layouts, so we use a factory method to create the projection.\n"
    "        self.in_proj_ba = self.create_ba_proj(\n"
    "            hidden_size=self.hidden_size,\n"
    "            num_v_heads=self.num_v_heads,\n"
    "            quant_config=quant_config,\n"
    "            prefix=f\"{prefix}.in_proj_ba\",\n"
    "        )\n"
)
NEW_INIT = (
    "        # ba_proj doesn't support blockwise fp8 quantization.\n"
    "        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint\n"
    "        # layouts, so we use a factory method to create the projection.\n"
    "        " + SENTINEL + "\n"
    "        if self.gqa_interleaved_layout:\n"
    "            # Qwen3-Next: single fused MergedColumnParallelLinear\n"
    "            self.in_proj_ba = self.create_ba_proj(\n"
    "                hidden_size=self.hidden_size,\n"
    "                num_v_heads=self.num_v_heads,\n"
    "                quant_config=quant_config,\n"
    "                prefix=f\"{prefix}.in_proj_ba\",\n"
    "            )\n"
    "        else:\n"
    "            # SM121/GB10: separate ReplicatedLinear for Qwen3.5.\n"
    "            # MergedColumnParallelLinear TP=4 -> N=32 < ConchLinearKernel BLOCK_N=64\n"
    "            # -> unmasked scale OOB reads -> cudaErrorIllegalAddress.\n"
    "            self.in_proj_b = ReplicatedLinear(\n"
    "                input_size=self.hidden_size,\n"
    "                output_size=self.num_v_heads,\n"
    "                bias=False,\n"
    "                quant_config=quant_config,\n"
    "                prefix=f\"{prefix}.in_proj_b\",\n"
    "            )\n"
    "            self.in_proj_a = ReplicatedLinear(\n"
    "                input_size=self.hidden_size,\n"
    "                output_size=self.num_v_heads,\n"
    "                bias=False,\n"
    "                quant_config=quant_config,\n"
    "                prefix=f\"{prefix}.in_proj_a\",\n"
    "            )\n"
)
if OLD_INIT not in content:
    print(f"[fix-qwen35-tp4-marlin] ERROR: __init__ create_ba_proj pattern not found")
    sys.exit(1)
content = content.replace(OLD_INIT, NEW_INIT, 1)
print("[fix-qwen35-tp4-marlin] Step 2: Fixed __init__ to use conditional ReplicatedLinear")

# ──────────────────────────────────────────────────────────────────
# 3. Fix LoRA forward path (Qwen3.5 LoRA: in_proj_qkv exists)
# ──────────────────────────────────────────────────────────────────
OLD_LORA_FWD = (
    "        if hasattr(self, \"in_proj_qkv\"):\n"
    "            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z\n"
    "            mixed_qkv, _ = self.in_proj_qkv(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
    "            z, _ = self.in_proj_z(hidden_states)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            b, a = ba.chunk(2, dim=-1)\n"
    "            b = b.contiguous()\n"
    "            a = a.contiguous()\n"
)
NEW_LORA_FWD = (
    "        if hasattr(self, \"in_proj_qkv\"):\n"
    "            # LoRA path (Qwen3.5 only): separate in_proj_qkv and in_proj_z\n"
    "            mixed_qkv, _ = self.in_proj_qkv(hidden_states)\n"
    "            z, _ = self.in_proj_z(hidden_states)\n"
    "            z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "            # SM121/GB10 fix: in_proj_b/a are ReplicatedLinear, slice per TP rank\n"
    "            b_full, _ = self.in_proj_b(hidden_states)\n"
    "            a_full, _ = self.in_proj_a(hidden_states)\n"
    "            _ba_chunk = self.num_v_heads // self.tp_size\n"
    "            _ba_start = self.tp_rank * _ba_chunk\n"
    "            b = b_full[:, _ba_start:_ba_start + _ba_chunk].contiguous()\n"
    "            a = a_full[:, _ba_start:_ba_start + _ba_chunk].contiguous()\n"
)
if OLD_LORA_FWD not in content:
    print(f"[fix-qwen35-tp4-marlin] ERROR: LoRA forward pattern not found")
    sys.exit(1)
content = content.replace(OLD_LORA_FWD, NEW_LORA_FWD, 1)
print("[fix-qwen35-tp4-marlin] Step 3: Fixed LoRA forward path")

# ──────────────────────────────────────────────────────────────────
# 4. Fix non-LoRA forward path: move ba call inside gqa_interleaved=True branch,
#    add in_proj_b/a calls in the Qwen3.5 (else) branch
# ──────────────────────────────────────────────────────────────────
OLD_NONLORA_FWD = (
    "        else:\n"
    "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "            ba, _ = self.in_proj_ba(hidden_states)\n"
    "\n"
    "            if self.gqa_interleaved_layout:\n"
    "                # Qwen3-Next: unpack the interleaved GQA layout\n"
    "                query, key, value, z, b, a = self.fix_query_key_value_ordering(\n"
    "                    mixed_qkvz, ba\n"
    "                )\n"
    "                query, key, value = map(\n"
    "                    lambda x: rearrange(x, \"l p d -> l (p d)\"), (query, key, value)\n"
    "                )\n"
    "                mixed_qkv = torch.cat((query, key, value), dim=-1)\n"
    "            else:\n"
    "                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size\n"
    "                z_size = self.value_dim // self.tp_size\n"
    "                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)\n"
    "                z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "                b, a = ba.chunk(2, dim=-1)\n"
    "                b = b.contiguous()\n"
    "                a = a.contiguous()\n"
)
NEW_NONLORA_FWD = (
    "        else:\n"
    "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
    "\n"
    "            if self.gqa_interleaved_layout:\n"
    "                # Qwen3-Next: unpack the interleaved GQA layout\n"
    "                ba, _ = self.in_proj_ba(hidden_states)\n"
    "                query, key, value, z, b, a = self.fix_query_key_value_ordering(\n"
    "                    mixed_qkvz, ba\n"
    "                )\n"
    "                query, key, value = map(\n"
    "                    lambda x: rearrange(x, \"l p d -> l (p d)\"), (query, key, value)\n"
    "                )\n"
    "                mixed_qkv = torch.cat((query, key, value), dim=-1)\n"
    "            else:\n"
    "                # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order\n"
    "                qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size\n"
    "                z_size = self.value_dim // self.tp_size\n"
    "                mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)\n"
    "                z = z.reshape(z.size(0), -1, self.head_v_dim)\n"
    "                # SM121/GB10 fix: in_proj_b/a are ReplicatedLinear, slice per TP rank\n"
    "                b_full, _ = self.in_proj_b(hidden_states)\n"
    "                a_full, _ = self.in_proj_a(hidden_states)\n"
    "                _ba_chunk = self.num_v_heads // self.tp_size\n"
    "                _ba_start = self.tp_rank * _ba_chunk\n"
    "                b = b_full[:, _ba_start:_ba_start + _ba_chunk].contiguous()\n"
    "                a = a_full[:, _ba_start:_ba_start + _ba_chunk].contiguous()\n"
)
if OLD_NONLORA_FWD not in content:
    print(f"[fix-qwen35-tp4-marlin] ERROR: non-LoRA forward pattern not found")
    sys.exit(1)
content = content.replace(OLD_NONLORA_FWD, NEW_NONLORA_FWD, 1)
print("[fix-qwen35-tp4-marlin] Step 4: Fixed non-LoRA forward path")

# ──────────────────────────────────────────────────────────────────
# Write and verify
# ──────────────────────────────────────────────────────────────────
with open(path, "w") as f:
    f.write(content)

import subprocess
result = subprocess.run(
    ["python3", "-c", f"import ast; ast.parse(open('{path}').read()); print('OK')"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(f"[fix-qwen35-tp4-marlin] SYNTAX ERROR after patch: {result.stderr}")
    sys.exit(1)
print(f"[fix-qwen35-tp4-marlin] Syntax check: {result.stdout.strip()}")

# Verify sentinel and key patterns
with open(path) as f:
    final = f.read()
checks = [
    ("ReplicatedLinear import", "    ReplicatedLinear,\n    RowParallelLinear,"),
    ("sentinel comment", SENTINEL),
    ("conditional init", "if self.gqa_interleaved_layout:\n            # Qwen3-Next: single fused"),
    ("in_proj_b created", "self.in_proj_b = ReplicatedLinear("),
    ("in_proj_a created", "self.in_proj_a = ReplicatedLinear("),
    ("LoRA fwd uses in_proj_b", "b_full, _ = self.in_proj_b(hidden_states)"),
    ("nonLoRA fwd uses in_proj_b", "b_full, _ = self.in_proj_b(hidden_states)"),
]
all_ok = True
for desc, pattern in checks:
    ok = pattern in final
    print(f"[fix-qwen35-tp4-marlin]   {desc}: {'OK' if ok else 'MISSING'}")
    if not ok:
        all_ok = False

if not all_ok:
    print("[fix-qwen35-tp4-marlin] ERROR: verification failed")
    sys.exit(1)

print("[fix-qwen35-tp4-marlin] All checks passed.")
PYEOF

echo "[fix-qwen35-tp4-marlin] Applying rope fix..."
python3 "$MOD_DIR/fix_rope.py"
echo "[fix-qwen35-tp4-marlin] Done."
