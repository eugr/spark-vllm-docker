#!/usr/bin/env bash
set -euo pipefail

site=/usr/local/lib/python3.12/dist-packages
bundle=/opt/qwen38-r12

python3 - <<'PY'
from importlib import metadata
import sys

expected = {
    "torch": "2.13.0+cu130",
    "triton": "3.8.0",
    "vllm": "0.1.dev20073+g8e685d198",
    "flashinfer-python": "0.6.18",
    "flashinfer-cubin": "0.6.18",
    "flashinfer-jit-cache": "0.6.18",
    "fastsafetensors": "0.3.3",
    "nvidia-cutlass-dsl": "4.7.0",
    "apache-tvm-ffi": "0.1.11",
    "transformers": "5.15.1",
    "compressed-tensors": "0.17.0",
}
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"unsupported Python: {sys.version}; expected 3.12")
for package, wanted in expected.items():
    found = metadata.version(package)
    if found != wanted:
        raise SystemExit(
            f"incompatible base image: {package}={found}; expected {wanted}"
        )
print("Validated pinned Qwen3.8 base runtime:", expected)
from pathlib import Path
assert len(list(Path('/usr/local/lib/python3.12/dist-packages').glob('vllm-*.dist-info'))) == 1
PY

test -d "$site/vllm"
test -d "$site/triton/language"
test -f /usr/local/lib/qwen38/libple_linux_aio.so

install -m 0644 "$bundle/triton-compat/core.py" "$site/triton/language/core.py"
install -m 0644 "$bundle/triton-compat/semantic.py" "$site/triton/language/semantic.py"
install -m 0644 "$bundle/triton-compat/__init__.py" "$site/triton/language/__init__.py"

python3 - "$site/triton/language/core.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
old_type = 'address_space: str = "global"'
new_type = "address_space: int = 1"
old_constant = 'address_space = "constant" if self.const else self.address_space'
new_constant = "address_space = 4 if self.const else self.address_space"
if source.count(old_type) != 1 or source.count(old_constant) != 1:
    raise SystemExit("unexpected vendored Triton compatibility frontend")
path.write_text(source.replace(old_type, new_type).replace(old_constant, new_constant))
PY

install -D -m 0644 "$bundle/modules/ple_linux_aio.py" \
  "$site/vllm/v1/ple_offload/ple_linux_aio.py"
install -D -m 0644 "$bundle/modules/qwen_draft_head_mxfp4.py" \
  "$site/vllm/model_executor/models/qwen_draft_head_mxfp4.py"
install -D -m 0644 "$bundle/modules/gdn_overlap.py" \
    "$site/vllm/models/qwen3_8_flash_next/nvidia/gdn_overlap.py"
install -m 0644 "$bundle/modules/image_policy.py" "$site/qwen38_image_policy.py"
install -m 0644 "$bundle/modules/image_resize.py" "$site/qwen38_image_resize.py"
install -m 0644 "$bundle/modules/image_encoder.py" "$site/qwen38_image_encoder.py"

python3 "$bundle/patches/patch_early_image_resize.py"
python3 "$bundle/patches/patch_sequential_image_encoder.py"

python3 "$bundle/patches/patch_gpu_worker.py"
python3 "$bundle/patches/patch_mixed_fp8_mxfp4_layers.py"
python3 "$bundle/patches/patch_index_filtered_safetensors.py"
python3 "$bundle/patches/patch_rank_local_fastsafetensors.py"
python3 "$bundle/patches/patch_deferred_mm_processor_warmup.py"
python3 "$bundle/patches/patch_fp8_qsa_kv.py"
python3 "$bundle/patches/patch_fp8_cutlass_afp8_alias.py"
python3 "$bundle/patches/patch_mixed_qwen_fp8_ple.py"
python3 "$bundle/patches/patch_nvme_ple_offload.py" \
  --nvme-source "$bundle/modules/ple_nvme.py"
python3 "$bundle/patches/patch_mamba_resume_block_size.py"
python3 "$bundle/patches/patch_qwen3_8_flash_next_mtp.py" \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/mtp.py"
python3 "$bundle/patches/patch_eagle_head_sharing.py" \
  "$site/vllm/v1/worker/gpu/spec_decode/eagle/utils.py"
python3 "$bundle/patches/patch_flashinfer_moe_tactic_override.py"
python3 "$bundle/patches/patch_gdn_model.py"
python3 "$bundle/patches/patch_ple_linux_aio.py"

python3 -m py_compile \
  "$site/triton/language/core.py" \
  "$site/triton/language/semantic.py" \
  "$site/vllm/v1/ple_offload/ple_linux_aio.py" \
  "$site/vllm/v1/ple_offload/nvme_table.py" \
  "$site/vllm/model_executor/models/qwen_draft_head_mxfp4.py" \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/gdn_overlap.py" \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/model.py" \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/mtp.py" \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/qsa.py" \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py" \
  "$site/vllm/v1/worker/gpu/spec_decode/eagle/utils.py"

grep -q 'QWEN_LOCAL_NVME_PLE_OFFLOAD_V1' \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
grep -q 'QWEN_PLE_LINUX_AIO_PRODUCTION_V1' \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
grep -q 'QWEN_DRAFT_HEAD_MXFP4_V1' \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/mtp.py"
grep -q 'prepare_gdn_projection_overlap(self)' \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/model.py"
grep -q 'QWEN38_QSA_FP8_KV_V1' \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/qsa.py"
grep -q 'QWEN38_QSA_FP8_KV_V1' \
  "$site/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"

python3 - <<'PY'
import torch
import triton

assert torch.__version__ == "2.13.0+cu130"
assert torch.version.cuda == "13.0"
assert triton.__version__ == "3.8.0"
print("Installed Qwen3.8 retain-12 runtime with Triton", triton.__version__)
PY
