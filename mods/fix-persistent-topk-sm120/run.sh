#!/bin/bash
# Gate vLLM's persistent_topk on shared-memory CAPACITY so low-smem (GB10 /
# consumer Blackwell, ~101KB opt-in) parts fall back to the generic
# top_k_per_row_decode kernel instead of hard-failing in persistent_topk.
#
# Root cause (csrc/libtorch_stable/topk.cu): persistent_topk's oversubscription
# fallback, FilteredTopKRaggedTransform, requires >=128KB opt-in shared memory
# per block. GB10 only has 101376 B, so once the decode grid oversubscribes the
# 48 SMs (total_ctas > num_sms*occupancy) the kernel aborts:
#     "persistent_topk would oversubscribe and the FilteredTopK fallback
#      requires >=128KB smem per block (have 101376). total_ctas=62 > ...".
# The clean, rebuild-free fix is to stop CALLING persistent_topk on those parts
# and let the indexer route to ops.top_k_per_row_decode.
#
# GLM-5.3-Flash's MLA indexer lives in sparse_attn_indexer_kpool.py (the
# DeepSeek-V3 path is sparse_attn_indexer.py). BOTH files call persistent_topk,
# so both are gated here. The gate is on smem CAPACITY (not capability family)
# so RTX 50-series (also family-120 with <128KB smem) is covered too.
#
# Durable sources of truth, kept in sync:
#   * mods/fix-persistent-topk-sm120/upstream-source.patch  (the .py diff)
#   * this run.sh (container-side patch of the same logic)
#
# Idempotent: reruns are no-ops. Escape hatch for A/B + diagnostics:
# VLLM_SPARSE_IDX_FORCE_PERSISTENT_TOPK=1 forces persistent_topk back on even
# for low-smem parts (mirrors topk.cu's VLLM_TOPK_DISABLE_NONCOOP knob).
set -e
echo "--- Patching sparse indexers: gate persistent_topk on device smem capacity..."

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
LAYERS="$SITE_PACKAGES/vllm/model_executor/layers"

# Resolve possibly-relocated vllm installs via `python -c`.
resolve() {
    local rel="$1"
    python3 -c "import vllm.model_executor.layers.${rel} as m; print(m.__file__)" 2>/dev/null \
        || echo "$LAYERS/$(basename "$rel").py"
}

TARGETS=(
    "$(resolve sparse_attn_indexer)"
    "$(resolve sparse_attn_indexer_kpool)"
)

python3 - "${TARGETS[@]}" << 'PYEOF'
import os
import re
import sys

targets = sys.argv[1:]

# Injected once per file, immediately after `logger = init_logger(__name__)`.
HELPER = '''

_PERSISTENT_TOPK_SMEM_OK: bool | None = None


def _device_has_persistent_topk_smem() -> bool:
    """True when the active CUDA device can run persistent_topk's fallbacks.

    persistent_topk's oversubscription fallback (FilteredTopKRaggedTransform)
    requires >=128KB opt-in shared memory per block (topk.cu). GB10 and other
    consumer/edge parts expose only ~101KB, so persistent_topk hard-fails
    ("requires >=128KB smem per block (have 101376)") as soon as the decode grid
    oversubscribes the 48 SMs. Gate on smem CAPACITY (not capability family) so
    the same check also catches RTX 50-series (family-120 with <128KB smem).
    """
    global _PERSISTENT_TOPK_SMEM_OK
    if _PERSISTENT_TOPK_SMEM_OK is None:
        if not torch.cuda.is_available():
            _PERSISTENT_TOPK_SMEM_OK = False
        else:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            max_smem = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
            num_sms = int(getattr(props, "multi_processor_count", 0) or 0)
            force = os.environ.get("VLLM_SPARSE_IDX_FORCE_PERSISTENT_TOPK", "0") == "1"
            _PERSISTENT_TOPK_SMEM_OK = force or max_smem >= 128 * 1024
            logger.info(
                "Sparse indexer persistent_topk smem gate: "
                "num_sms=%d shared_memory_per_block_optin=%d -> persistent_topk=%s",
                num_sms, max_smem, _PERSISTENT_TOPK_SMEM_OK,
            )
    return _PERSISTENT_TOPK_SMEM_OK
'''

ANCHOR = "logger = init_logger(__name__)"
# Distinct marker for the helper *definition* (idempotency check).
DEF = "def _device_has_persistent_topk_smem("


# GLM MLA / K-pool gate:  if current_platform.is_cuda() and select_k in (512,1024,2048):
def gate_rewrites_kpool(content: str) -> str:
    old = "if current_platform.is_cuda() and select_k in (512, 1024, 2048):"
    if old in content:
        return content.replace(
            old,
            "if (\n            current_platform.is_cuda()\n"
            "            and select_k in (512, 1024, 2048)\n"
            "            and _device_has_persistent_topk_smem()\n        ):",
            1,
        )
    return content


def gate_rewrites_dsv3(content: str) -> str:
    # Modern split form: use_persistent_topk = ... and topk_tokens in (512,1024,2048)
    m = re.search(
        r"use_persistent_topk\s*=\s*current_platform\.is_cuda\(\) and topk_tokens in \((.*?)\)",
        content, flags=re.S)
    if m:
        block = m.group(0)
        return content.replace(
            block, block.rstrip() + " and _device_has_persistent_topk_smem()", 1)
    # Legacy inline form: elif current_platform.is_cuda() and topk_tokens in (512,1024,2048):
    legacy = "elif current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):"
    if legacy in content:
        return content.replace(
            legacy,
            "elif current_platform.is_cuda() and topk_tokens in (512, 1024, 2048) "
            "and _device_has_persistent_topk_smem():  # FIXED: persistent_topk smem gate",
            1)
    return content


def patch_file(path: str) -> str:
    if not os.path.exists(path):
        return f"SKIP (not found): {path}"
    with open(path) as f:
        content = f.read()
    if DEF in content:
        return f"OK (already patched): {path}"

    base = os.path.basename(path)
    # 1) Wire the gate CALL first (original file has no helper yet).
    if "kpool" in base:
        content = gate_rewrites_kpool(content)
    else:
        content = gate_rewrites_dsv3(content)
    if DEF not in content and "_device_has_persistent_topk_smem()" not in content:
        return f"FAIL (gate pattern not matched): {path}"

    # 2) Define the helper right after the module logger.
    if ANCHOR in content:
        content = content.replace(ANCHOR, ANCHOR + HELPER, 1)
    if DEF not in content:
        return f"FAIL (logger anchor not found): {path}"

    with open(path, "w") as f:
        f.write(content)
    return f"PATCHED: {path}"


for t in targets:
    print(patch_file(t))
PYEOF
