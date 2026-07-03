#!/bin/bash
set -euo pipefail

# GLM-5.2 (GlmMoeDsaForCausalLM) support on DGX Spark (GB10 / sm_121).
#
# GLM-5.2's MLA decode (Lk=576: kv_lora_rank=512 + qk_rope_head_dim=64) maps
# to BLOCK_DMODEL=512 in the Triton grouped decode kernel. With the default
# num_stages=2 the kernel needs 102400 bytes of shared memory per SM, but
# sm_121 only has 101376 -> triton.runtime.errors.OutOfResources at decode.
# Upstream vLLM only drops to num_stages=1 for BLOCK_DMODEL >= 1024; this mod
# lowers that threshold to 512.
#
# Applied as an exact-string edit (not a git patch) so it survives context
# drift when building vLLM from main.

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
TARGET="$PYTHON_ROOT/vllm/v1/attention/ops/triton_decode_attention.py"

if [ ! -f "$TARGET" ]; then
  echo "[glm-5.2] vLLM triton_decode_attention.py not found at $TARGET" >&2
  exit 1
fi

python3 - "$TARGET" <<'PY'
from pathlib import Path
import py_compile
import sys

path = Path(sys.argv[1])
text = path.read_text()

old = "elif not is_hip_ and BLOCK_DMODEL >= 1024:"
new = "elif not is_hip_ and BLOCK_DMODEL >= 512:"

# Sanity check: the MLA tile alignment (Lk==576 -> BLOCK_DMODEL=512) must be
# present, otherwise this vLLM predates the upstream MLA tiling fix and needs
# more than this one-liner.
if "Lk == 576" not in text:
    raise SystemExit(
        "[glm-5.2] Installed vLLM lacks the MLA tile alignment (Lk == 576) "
        "in triton_decode_attention.py; this mod only lowers the num_stages "
        "threshold and is not sufficient for this vLLM version."
    )

if new in text:
    print("[glm-5.2] sm_121 shared-memory fix already present; skipping.")
elif old in text:
    if text.count(old) != 1:
        raise SystemExit(
            f"[glm-5.2] Expected exactly one occurrence of {old!r}, "
            f"found {text.count(old)}; refusing to patch."
        )
    path.write_text(text.replace(old, new))
    py_compile.compile(str(path), doraise=True)
    print(
        "[glm-5.2] Lowered num_stages threshold to BLOCK_DMODEL >= 512 "
        "(sm_121 shared-memory fix for GLM-5.2 MLA decode)."
    )
else:
    raise SystemExit(
        "[glm-5.2] Could not find the num_stages threshold line in "
        "triton_decode_attention.py; upstream code has changed. "
        "Check whether the shared-memory guard now covers BLOCK_DMODEL=512."
    )
PY

echo "=====> vLLM Triton MLA decode kernel patched for GLM-5.2 on sm_121"
