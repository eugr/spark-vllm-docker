#!/bin/bash
# Mod: step-3.7-mtp-bf16-draft
# Step-3.7-Flash checkpoints bundle their 3 MTP nextn layers in bf16
# (model-mtp-bf16.safetensors) even when the main weights are ModelOpt NVFP4.
# For self-drafted MTP, vLLM force-inherits the target's fp4 quant onto the
# draft (config/speculative.py: self.quantization = target.quantization) and
# reuses the target ModelConfig, so the MTP shared_head is built fp4-packed
# ([vocab, hidden/2]) and loading the bf16 [vocab, hidden] weight crashes:
#   RuntimeError: size of tensor a (2048) must match b (4096) at dim 1
# The shared_head is created without a prefix, so a config.json `ignore` entry
# cannot reach it. Fix: build the MTP draft (shared_head + mtp_block) with
# quant_config=None -> unquantized bf16, which matches the checkpoint. Only the
# draft path is touched; the fp4 main model is loaded by its own loader.
set -euo pipefail
PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
F="$PYTHON_ROOT/vllm/model_executor/models/step3p5_mtp.py"
[ -f "$F" ] || { echo "[step-3.7-mtp-bf16-draft] $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import sys
f = sys.argv[1]
s = open(f).read()
if "step-3.7-mtp-bf16-draft" in s:
    print("[step-3.7-mtp-bf16-draft] already applied; skipping.")
    sys.exit(0)
old = (
    "        config = vllm_config.model_config.hf_config\n"
    "        quant_config = vllm_config.quant_config\n"
    "        self.enorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)\n"
)
new = (
    "        config = vllm_config.model_config.hf_config\n"
    "        # step-3.7-mtp-bf16-draft: MTP nextn layers ship bf16 even when the\n"
    "        # target is NVFP4; build the draft unquantized or the fp4-packed\n"
    "        # shapes mismatch the bf16 weights (shared_head 2048 vs 4096).\n"
    "        import copy as _ptc_copy\n"
    "        vllm_config = _ptc_copy.copy(vllm_config)\n"
    "        vllm_config.quant_config = None\n"
    "        quant_config = None\n"
    "        self.enorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)\n"
)
if old not in s:
    print("[step-3.7-mtp-bf16-draft] anchor not found; vLLM layout changed.", file=sys.stderr)
    sys.exit(1)
open(f, "w").write(s.replace(old, new, 1))
print("[step-3.7-mtp-bf16-draft] patched step3p5_mtp.py (MTP draft -> bf16).")
PY
