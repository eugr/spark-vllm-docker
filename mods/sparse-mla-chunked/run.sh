#!/bin/bash
set -euo pipefail

# GLM-5.2 / DSA prefill fix for DGX Spark (sm_121).
#
# FlashInfer 0.6.14's SM120 sparse-MLA path dispatches attention calls with
# num_tokens <= 64 to the autotuned decode-dsv3_2 kernel (stable on sm_121)
# and everything larger to a generic sparse_mla_sm120_paged_attention
# fallback kernel that hangs on sm_121. Since each query token attends
# independently over its own top-k indices (MQA-style), a large call is
# mathematically identical to a sequence of <=64-token calls. This mod
# patches flashinfer's _paged_attention to chunk oversized dsv3_2/GLM calls
# through the working kernel, which lets vLLM run real prefill chunk sizes
# (max_num_batched_tokens >> 64) without wedging the cluster.

SITE_PACKAGES="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
TARGET="$SITE_PACKAGES/flashinfer/mla/_sparse_mla_sm120.py"

if [ ! -f "$TARGET" ]; then
  echo "[sparse-mla-chunked] $TARGET not found" >&2
  exit 1
fi

python3 - "$TARGET" <<'PY'
from pathlib import Path
import py_compile
import sys

path = Path(sys.argv[1])
text = path.read_text()

MARKER = "spark-vllm-docker: chunk oversized sparse-MLA calls"
if MARKER in text:
    print("[sparse-mla-chunked] already patched; skipping.")
    raise SystemExit(0)

anchor = """        if model_type in (
            _MODEL_TYPE_DSV3_2,
            _MODEL_TYPE_GLM_NSA,
        ) and _decode_dsv3_2_dispatchable(num_tokens, num_heads, topk, d_qk, kv_pbs):"""

if anchor not in text:
    raise SystemExit(
        "[sparse-mla-chunked] dsv3_2 dispatch anchor not found — flashinfer "
        "layout changed; refusing to patch."
    )

block = """        # spark-vllm-docker: chunk oversized sparse-MLA calls into <=64-token
        # slices so they dispatch to the working decode-dsv3_2 kernel instead
        # of the generic fallback, which hangs on sm_121 for num_tokens > 64.
        # Queries are independent in this MQA-style kernel, so slicing on the
        # token dim is exact.
        if (
            model_type in (_MODEL_TYPE_DSV3_2, _MODEL_TYPE_GLM_NSA)
            and num_tokens > _DECODE_MAX_TOKENS
            and extra_kv_cache is None
            and extra_indices is None
            and _decode_dsv3_2_dispatchable(
                _DECODE_MAX_TOKENS, num_heads, topk, d_qk, kv_pbs
            )
        ):
            _num_splits = (topk + _BI - 1) // _BI
            _c_mid_out = torch.empty(
                (_DECODE_MAX_TOKENS, num_heads, _num_splits, d_v),
                dtype=torch.bfloat16,
                device=q.device,
            )
            _c_mid_lse = torch.empty(
                (_DECODE_MAX_TOKENS, num_heads, _num_splits),
                dtype=torch.float32,
                device=q.device,
            )
            for _s in range(0, num_tokens, _DECODE_MAX_TOKENS):
                _e = min(_s + _DECODE_MAX_TOKENS, num_tokens)
                _n = _e - _s
                sparse_mla_sm120_decode_dsv3_2(
                    q[_s:_e],
                    kv_cache,
                    indices[_s:_e],
                    _c_mid_out[:_n],
                    _c_mid_lse[:_n],
                    output[_s:_e],
                    out_lse[_s:_e],
                    sm_scale,
                    topk_length=(
                        topk_length[_s:_e]
                        if topk_length is not None and topk_length.dim() > 0
                        else topk_length
                    ),
                    attn_sink=attn_sink,
                    model_type=model_type,
                )
            return

"""

text = text.replace(anchor, block + anchor)
path.write_text(text)
py_compile.compile(str(path), doraise=True)
print("[sparse-mla-chunked] patched _paged_attention with 64-token chunking.")
PY

find "$SITE_PACKAGES/flashinfer" -name "__pycache__" -path "*mla*" -exec rm -rf {} + 2>/dev/null || true
echo "=====> FlashInfer sparse-MLA sm_121 chunking workaround applied"
