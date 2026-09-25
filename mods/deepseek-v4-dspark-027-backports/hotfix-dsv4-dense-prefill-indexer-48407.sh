#!/usr/bin/env bash
# hotfix-dsv4-dense-prefill-indexer-48407.sh — Skip indexer scoring on dense prefills (Stage A, DORMANT)
#
# Backport of upstream vLLM PR #48407 ("skip indexer scoring on short dense
# prefills") to the Anemll dspark-vllm-gx10 0.1.1 image
# (vLLM 0.25.2.dev0+g752a3a504.d20260714). STAGE A ONLY — portable machinery,
# deliberately impossible to trigger.
#
# WHAT UPSTREAM #48407 DOES: when the main MLA attention decides a short prefill
# is routed to dense MHA (prefill_max_seq_len <= topk_tokens) and the batch has
# zero decode tokens (decode always consumes top-k), the indexer's
# compress+score+top-k work is pure waste. The patch makes the indexer op check
# the main MLA metadata (use_dense_mha && num_decode_tokens == 0 && not graph
# capturing) and return the untouched buffer early (still caching K).
#
# WHY THIS FORK SHIPS IT DORMANT (Stage A only):
#   The fork has NO dense-MHA route for sparse-MLA prefills: use_dense_mha /
#   dense_mha / num_mha_tokens / force_mqa do not exist in vllm/models/
#   deepseek_v4/ or the MLA backends. #48407's premise ("top-k is not consumed
#   this step") is therefore FALSE on this fork until a dense route exists.
#   Shipping the skip live would silently drop valid KV selection = wrong
#   attention. So this port:
#     - adds dense_mha_metadata_layer_name to sparse_attn_indexer() and
#       sparse_attn_indexer_fake() (fork has no use_pcp — not added; the gate
#       never sees a dense route),
#     - adds the mainline skip gate verbatim (minus PCP),
#     - populates DeepseekV4FlashMLAMetadata.num_decode_tokens in build() for
#       diagnostics / future staging,
#     - binds DeepseekV4Indexer.indexer_op.dense_mha_metadata_layer_name = "".
#   Because the binding is "", _resolve_layer_name("") is falsy, so the skip
#   can never fire. Correct non-empty value later = the main MLA cache prefix
#   (Stage B card; NOT the indexer's own .k_cache.prefix).
#
# DO NOT ENABLE Stage B from this script. Stage B = dense-MHA route (or kill
# #48407 entirely); see docs/vllm-027-new-patches.md.
#
# Usage:
#   docker cp hotfix-dsv4-dense-prefill-indexer-48407.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-dense-prefill-indexer-48407.sh
#   # then stop + start the server yourself (this script never restarts it)
#
# Validation (run from the HOST, not in the container):
#   bash hotfix-dsv4-dense-prefill-indexer-48407.sh --before  # pre-restart KV + histogram
#   ... apply + restart yourself ...
#   bash hotfix-dsv4-dense-prefill-indexer-48407.sh --after   # KV must be unchanged
#
# Apply on BOTH nodes (each node runs its own container). Idempotent.
set -euo pipefail

# ---- config -----------------------------------------------------------------
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
CONTAINER="${CONTAINER:-deepseek-v4-flash-vllm-dspark-1}"
API_URL="${API_URL:-http://127.0.0.1:8888}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")" && pwd)/../results}"
BASELINE_FILE="$RESULTS_DIR/hotfix-48407-baseline.txt"
AFTER_FILE="$RESULTS_DIR/hotfix-48407-after.txt"

ACTION="${1:-patch}"

# VLLM_ROOT is only needed for patch / --status (which run inside the container
# via docker exec). The host-side --before/--after capture modes only need the
# docker/curl tooling, so skip the check there.
if [ "$ACTION" != "--before" ] && [ "$ACTION" != "--baseline" ] \
   && [ "$ACTION" != "--after" ] && [ "$ACTION" != "--verify" ]; then
  if [ ! -d "$VLLM_ROOT" ]; then
    echo "ERROR: vLLM not found at $VLLM_ROOT" >&2
    exit 1
  fi
fi

# ---- validation helpers (host side) -----------------------------------------
capture_kv_snapshot() {
  local tag="$1" outfile="$2"
  {
    echo "=== KV snapshot ($tag) $(date -Is) ==="
    echo "-- boot log (docker logs ${CONTAINER}) --"
    docker logs "$CONTAINER" 2>&1 \
      | grep -E "Available KV cache memory:|GPU KV cache size:|Maximum concurrency for" \
      | tail -6 || true
    echo "-- live /metrics --"
    curl -s -m 8 "$API_URL/metrics" 2>/dev/null \
      | grep -E "cache_config_info" \
      | grep -oE '(num_gpu_blocks|kv_cache_size_tokens|kv_cache_max_concurrency)="[0-9.]+"' \
      || true
    echo "-- request prompt-length histogram (_count/sum + buckets) --"
    curl -s -m 8 "$API_URL/metrics" 2>/dev/null \
      | grep -E "vllm:request_prompt_tokens_(count|sum|bucket)" \
      || true
  } > "$outfile"
  echo "[OK] $tag snapshot -> $outfile"
}

print_histogram_summary() {
  local f="$1"
  python3 - "$f" <<'PY'
import math, re, sys
f = sys.argv[1]
counts = {}
try:
    for line in open(f):
        m = re.match(r'vllm:request_prompt_tokens_bucket\{[^}]*le="([^"]+)"[^}]*\}\s+([0-9.eE+-]+)', line)
        if m:
            counts[float(m.group(1))] = float(m.group(2))
except FileNotFoundError:
    print("  (no histogram captured)")
    sys.exit(0)
if not counts:
    print("  (no request_prompt_tokens histogram yet on this server)")
    sys.exit(0)
total = max(counts.values(), default=0.0)
prev_c = 0.0
print("  prompt-tokens per request (bucket -> requests -> %):")
for b in sorted(counts):
    n = counts[b] - prev_c
    prev_c = counts[b]
    if n > 0 or b in (100.0, 1000.0, 5000.0, 20000.0, 65536.0, 262144.0, 1048576.0):
        print(f"    <= {b:>10.0f}: {n:>10.0f}  ({100 * n / total:5.1f}%)")
short = 0
try:
    for line in open(f):
        m = re.match(r'vllm:request_prompt_tokens_bucket\{[^}]*le="512.0"[^}]*\}\s+([0-9.eE+-]+)', line)
        if m:
            short = float(m.group(1))
except FileNotFoundError:
    short = 0
if total and short:
    print(f"  requests with prompt <= 512 tokens (dormant #48407 window): {short:>10.0f}  ({100*short/total:5.1f}%)")
PY
}

# ---- actions ----------------------------------------------------------------
if [ "$ACTION" = "--before" ] || [ "$ACTION" = "--baseline" ]; then
  mkdir -p "$RESULTS_DIR"
  capture_kv_snapshot "BEFORE (pre-restart)" "$BASELINE_FILE"
  echo ""; echo "Baseline prompt-length histogram:"; print_histogram_summary "$BASELINE_FILE"
  echo ""
  echo "Next: apply the patch, restart the server yourself, then run:"
  echo "  bash $(basename "$0") --after"
  exit 0
fi

if [ "$ACTION" = "--after" ] || [ "$ACTION" = "--verify" ]; then
  if [ ! -f "$BASELINE_FILE" ]; then
    echo "ERROR: no baseline at $BASELINE_FILE — run --before first" >&2
    exit 1
  fi
  mkdir -p "$RESULTS_DIR"
  capture_kv_snapshot "AFTER (post-restart)" "$AFTER_FILE"
  echo ""; echo "=== KV budget diff (BEFORE -> AFTER) ==="
  python3 - "$BASELINE_FILE" "$AFTER_FILE" <<'PY'
import sys
def grab(f, pat, end=None):
    try:
        for line in open(f):
            if pat in line:
                val = line.split(pat, 1)[1].strip()
                if end is not None:
                    val = val.split(end, 1)[0].strip()
                return val
    except FileNotFoundError:
        return None
    return None
B, A = sys.argv[1], sys.argv[2]
same = True
for label, pat, end in [
    ("Available KV cache memory", "Available KV cache memory: ", None),
    ("GPU KV cache size", "GPU KV cache size: ", None),
    ("num_gpu_blocks", 'num_gpu_blocks="', '"'),
]:
    b, a = grab(B, pat, end), grab(A, pat, end)
    chg = "" if (b is None or a is None or b == a) else "  <-- CHANGED (unexpected for dormant #48407)"
    if chg: same = False
    print(f"  {label:24} {str(b):>24} -> {str(a):>24}{chg}")
if same:
    print("  -> KV budget unchanged: expected (Stage A dormant machinery, no behavior delta). Good.")
PY
  echo ""; echo "After-start prompt-length histogram:"
  print_histogram_summary "$AFTER_FILE"
  echo ""
  echo "Expected #48407 (Stage A) effect: none, by design — dormant skip."
  exit 0
fi

if [ "$ACTION" = "--status" ]; then
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
si = (root / "model_executor/layers/sparse_attn_indexer.py").read_text()
sm = (root / "models/deepseek_v4/sparse_mla.py").read_text()
at = (root / "models/deepseek_v4/attention.py").read_text()
v32 = (root / "models/deepseek_v32/nvidia/attention.py").read_text()
ok = True
def check(name, cond):
    global ok
    print(f"{name:52}:", "APPLIED" if cond else "NOT APPLIED")
    if not cond: ok = False
check("sparse_attn_indexer() param", "dense_mha_metadata_layer_name: LayerNameType" in si)
check("sparse_attn_indexer_fake() param", si.count("dense_mha_metadata_layer_name: LayerNameType") >= 2)
check("skip gate present", "[#48407 Stage A]" in si and "cudagraph_runtime_mode != CUDAGraphMode.FULL" in si)
check("__init__ dormant binding \"\"", 'self.dense_mha_metadata_layer_name = ""' in si)
check("forward_cuda passes binding", "_encode_layer_name(self.dense_mha_metadata_layer_name)" in si)
check("MLA num_decode_tokens field", "num_decode_tokens: int = 0" in sm)
check("build() populates split", "(_, _, num_decode_tokens, _) = split_decodes_and_prefills(" in sm)
check("DSV4 indexer binding == \"\"", 'self.indexer_op.dense_mha_metadata_layer_name = ""' in at)
check("v32 fused caller keywords", 'dense_mha_metadata_layer_name="",  # [#48407]' in v32)
# Dormancy gate: the binding string on the live DeepseekV4Indexer is "" only if
# the seatbelt comment (and the literal "") is present. _resolve_layer_name("")
# is falsy => the skip can never fire.
print("\nDormancy gate: binding literal is \"\" -> _resolve_layer_name(\"\") is falsy -> skip never fires.")
sys.exit(0 if ok else 1)
PY
  exit $?
fi

# ---- patch (default action) -------------------------------------------------
echo "=== Hotfix: DSV4 dense-prefill indexer scoring skip — STAGE A DORMANT (upstream #48407 backport) ==="
echo "vLLM root: $VLLM_ROOT  image: 0.25.2.dev0+g752a3a504.d20260714"

if [ -n "${WORKER_HOST:-}" ]; then
  echo ""
  echo "NOTE: WORKER_HOST is set — this is the HEAD node only."
  echo "Run the same script on the worker container too."
fi

python3 <<PYEOF
import sys
from pathlib import Path

root = Path("$VLLM_ROOT")
applied = 0
skipped = 0
errors = []

def patch(path: str, old: str, new: str, label: str, expect: int = 1) -> None:
    global applied, skipped
    p = root / path
    if not p.exists():
        errors.append(f"File not found: {path}")
        return
    text = p.read_text()
    if new in text:
        print(f"  [skip] {label} (already applied)")
        skipped += 1
        return
    n = text.count(old)
    if n == 0 or (expect and n != expect):
        errors.append(f"[ERR] anchor x{n} (expect {expect}) for {label} in {path}")
        return
    text = text.replace(old, new)
    p.write_text(text)
    print(f"  [OK]   {label} (replaced {n})")
    applied += 1


# ---- sparse_attn_indexer.py: CUDAGraphMode import ----
patch(
    "model_executor/layers/sparse_attn_indexer.py",
    "from vllm.config import get_current_vllm_config",
    "from vllm.config import CUDAGraphMode, get_current_vllm_config",
    "sparse_attn_indexer.py: import CUDAGraphMode",
)

# ---- sparse_attn_indexer.py: sparse_attn_indexer() signature param ----
patch(
    "model_executor/layers/sparse_attn_indexer.py",
    """    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,""",
    """    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    dense_mha_metadata_layer_name: LayerNameType,  # [#48407] dormant: caller passes """"",
    "sparse_attn_indexer.py: sparse_attn_indexer() signature param",
)

# ---- sparse_attn_indexer.py: capture forward_context (used by the skip gate) -
patch(
    "model_executor/layers/sparse_attn_indexer.py",
    """    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()""",
    """    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()""",
    "sparse_attn_indexer.py: capture forward_context",
)

# ---- sparse_attn_indexer.py: thread param into sparse_attn_indexer_fake() ----
patch(
    "model_executor/layers/sparse_attn_indexer.py",
    """            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )""",
    """            topk_indices_buffer,
            skip_k_cache_insert,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
        )""",
    "sparse_attn_indexer.py: pass param into sparse_attn_indexer_fake()",
)

# ---- sparse_attn_indexer.py: sparse_attn_indexer_fake() signature param -----
patch(
    "model_executor/layers/sparse_attn_indexer.py",
    """    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,""",
    """    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    dense_mha_metadata_layer_name: LayerNameType,  # [#48407] dormant: ignored by fake""",
    "sparse_attn_indexer.py: sparse_attn_indexer_fake() signature param",
)

# ---- sparse_attn_indexer.py: dormant skip gate (mainline verbatim, minus PCP)
patch(
    "model_executor/layers/sparse_attn_indexer.py",
    """    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the""",
    """    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    # [#48407 Stage A] PORTED DORMANT: dense_mha_metadata_layer_name is bound to
    # "" on this fork (no dense-MHA route for sparse-MLA prefills), so
    # _resolve_layer_name("") is falsy and this skip can never fire.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the""",
    "sparse_attn_indexer.py: dormant scoring-skip gate (mainline #48407)",
)

# ---- sparse_attn_indexer.py: SparseAttnIndexer.__init__ dormant binding -----
patch(
    "model_executor/layers/sparse_attn_indexer.py",
    """        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache""",
    """        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        # [#48407] Main MLA metadata key this op may consult (Stage A: dormant).
        self.dense_mha_metadata_layer_name = \"\"""",
    "sparse_attn_indexer.py: __init__ dormant binding field",
)

# ---- sparse_attn_indexer.py: forward_cuda passes the (dormant) binding -------
patch(
    "model_executor/layers/sparse_attn_indexer.py",
    """            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            self.dcp_rank,""",
    """            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,""",
    "sparse_attn_indexer.py: forward_cuda passes binding",
)

# ---- sparse_mla.py: DeepseekV4FlashMLAMetadata.num_decode_tokens field ------
patch(
    "models/deepseek_v4/sparse_mla.py",
    """    block_size: int
    topk_tokens: int
""",
    """    block_size: int
    topk_tokens: int

    # [#48407] number of decode tokens in this batch; 0 => a pure prefill batch.
    # Stage A: populated for diagnostics only. The indexer skip is dormant
    # (binding == ""), so nothing consumes this yet.
    num_decode_tokens: int = 0
""",
    "sparse_mla.py: DeepseekV4FlashMLAMetadata.num_decode_tokens field",
)

# ---- sparse_mla.py: build() populates num_decode_tokens ----------------------
patch(
    "models/deepseek_v4/sparse_mla.py",
    """        return DeepseekV4FlashMLAMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            query_start_loc=cm.query_start_loc,""",
    """        # [#48407] Number of decode tokens (0 => pure prefill batch), for the
        # (dormant) indexer scoring-skip decision. Same split + threshold the
        # C128A builder uses.
        (_, _, num_decode_tokens, _) = split_decodes_and_prefills(
            cm,
            decode_threshold=self.reorder_batch_threshold or 1,
        )

        return DeepseekV4FlashMLAMetadata(
            num_reqs=cm.num_reqs,
            max_query_len=cm.max_query_len,
            max_seq_len=cm.max_seq_len,
            num_actual_tokens=cm.num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            query_start_loc=cm.query_start_loc,""",
    "sparse_mla.py: build() populates num_decode_tokens",
)

# ---- attention.py: DeepseekV4Indexer.__init__ dormant binding (kept OFF) -----
patch(
    "models/deepseek_v4/attention.py",
    """            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

        # None on ROCm — maybe_execute_in_parallel falls back to sequential.""",
    """            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

        # [#48407] Binding that lets the indexer op find the main MLA metadata.
        # Kept "" (skip never fires) until the short-prefill dense-MHA route
        # lands: on this fork there is no dense route, so top-k IS consumed —
        # enabling this early would silently produce wrong attention.
        self.indexer_op.dense_mha_metadata_layer_name = ""

        # None on ROCm — maybe_execute_in_parallel falls back to sequential.""",
    "attention.py: DeepseekV4Indexer dormant binding \"\"",
)

# ---- deepseek_v32/nvidia/attention.py: keyword args + dormant "" ------------
# The v3.2 fused caller previously passed positional args; switch the tail to
# keywords so the new param slot is unambiguous (mirrors upstream #48407).
patch(
    "models/deepseek_v32/nvidia/attention.py",
    """                self.topk_indices_buffer,
                True,  # skip_k_cache_insert
                False,  # use_fp4_cache
                # fused_norm_rope already cleared the topk buffer this forward.
                skip_topk_buffer_clear=True,
            )""",
    """                self.topk_indices_buffer,
                skip_k_cache_insert=True,
                dense_mha_metadata_layer_name="",  # [#48407] dormant: "" never fires
                use_fp4_cache=False,
                # fused_norm_rope already cleared the topk buffer this forward.
                skip_topk_buffer_clear=True,
            )""",
    "deepseek_v32/nvidia/attention.py: keyword args + dormant binding",
)


print(f"\nApplied: {applied}, Skipped: {skipped}, Errors: {len(errors)}")
for e in errors:
    print(f"  {e}", file=sys.stderr)

if errors:
    print("\nWARNING: Some patches could not be applied. Nothing was left half-applied.")
    sys.exit(1)

if applied == 0 and skipped > 0:
    print("Patch already applied. No changes needed.")
elif applied > 0:
    print("\nHotfix applied (Stage A dormant). Stop + start the vLLM process (or the container) to take effect.")
    print("Validate with: bash <this-script> --before  (pre-restart; run BEFORE restarting)")
    print("             bash <this-script> --after   (post-restart; KV budget must be unchanged)")
PYEOF

echo ""
echo "=== Verification ==="
bash "$0" --status
