#!/usr/bin/env bash
# hotfix-dsv4-skip-topk-49486.sh — Skip indexer topk/router on short contexts (DSV4 Flash)
#
# Backport of upstream vLLM PR #49486 ("skip indexer topk/router when every
# candidate is selected") applied to the Anemll dspark-vllm-gx10 0.1.1 image
# (vLLM 0.25.2.dev0+g752a3a504.d20260714).
#
# WHAT IT FIXES: the Lightning-Indexer (C4) normally runs
#   wq_b projection -> RoPE -> MXFP4/FP8 quant -> QK logits -> top-k -> indices
# every forward step. When the context is short enough that ALL compressed
# candidates would be selected (max_seq_len / compress_ratio <= topk_tokens),
# that pipeline is a no-op: the answer is just [0..n-1, -1, -1, ...]. The patch
# short-circuits it: still compress + write the K cache, fill an all-candidates
# buffer with a tiny Triton kernel, and return. Output is bit-identical to the
# full path (top-k over <=k candidates == all candidates).
#
# TRIGGER WINDOW ON THIS FORK (DO NOT WIDEN):
#   max_seq_len <= 512 (index_topk) * 4 (compress_ratio) = 2048 tokens (C4).
#   At >2048 tokens neither this short-circuit nor its Triton kernel runs.
#
# FULL PORT, OUTPUT-IDENTICAL (no special-casing of decode prefill paths):
#   - Indexer metadata at attn_metadata[self.k_cache.prefix] is a
#     DeepseekV32IndexerMetadata, so the fast path uses upstream verbatim:
#     num_tokens = num_decode_tokens + num_prefill_tokens.
#   - No CUDA-graph guard (faithful to upstream): max_seq_len is a
#     per-captured-batch constant, so the branch is stable inside a capture.
#
# Usage:
#   docker cp hotfix-dsv4-skip-topk-49486.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-skip-topk-49486.sh
#   # then stop + start the server yourself (this script never restarts it)
#
# Validation (run from the HOST, not in the container):
#   bash hotfix-dsv4-skip-topk-49486.sh --before   # capture pre-restart KV + prompt histogram
#   ... apply + restart yourself ...
#   bash hotfix-dsv4-skip-topk-49486.sh --after    # KV budget must be unchanged
#
# Apply on BOTH nodes (each node runs its own container). Idempotent.
set -euo pipefail

# ---- config -----------------------------------------------------------------
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
CONTAINER="${CONTAINER:-deepseek-v4-flash-vllm-dspark-1}"
API_URL="${API_URL:-http://127.0.0.1:8888}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")" && pwd)/../results}"
BASELINE_FILE="$RESULTS_DIR/hotfix-49486-baseline.txt"
AFTER_FILE="$RESULTS_DIR/hotfix-49486-after.txt"

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
    # le="2048.0" bucket = requests whose prompt is <= 2048 tokens -> #49486 candidate
    for line in open(f):
        m = re.match(r'vllm:request_prompt_tokens_bucket\{[^}]*le="2048.0"[^}]*\}\s+([0-9.eE+-]+)', line)
        if m:
            short = float(m.group(1))
except FileNotFoundError:
    short = 0
if total and short:
    print(f"  requests with prompt <= 2048 tokens (#49486 trigger window): {short:>10.0f}  ({100*short/total:5.1f}%)")
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
    chg = "" if (b is None or a is None or b == a) else "  <-- CHANGED (unexpected for #49486)"
    if chg: same = False
    print(f"  {label:24} {str(b):>24} -> {str(a):>24}{chg}")
if same:
    print("  -> KV budget unchanged: expected for #49486 (pure compute saving). Good.")
PY
  echo ""; echo "After-start prompt-length histogram:"
  print_histogram_summary "$AFTER_FILE"
  echo ""
  echo "Expected #49486 effect: same KV budget, same accuracy; short-context"
  echo "(<=2048 token) indexer steps skip the wq_b/RoPE/quant/QK/topk pipeline."
  exit 0
fi

if [ "$ACTION" = "--status" ]; then
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
t = (root / "models/deepseek_v4/attention.py").read_text()
ok = True
def check(name, cond):
    global ok
    print(f"{name:44}:", "APPLIED" if cond else "NOT APPLIED")
    if not cond: ok = False
check("import tl, triton", "from vllm.triton_utils import tl, triton" in t)
check("_fill_short_context_topk_indices kernel", "@triton.jit\ndef _fill_short_context_topk_indices" in t)
check("[PORT #49486] early return", "[PORT #49486]" in t and "_fill_short_context_topk_indices[(num_tokens,)](" in t)
sys.exit(0 if ok else 1)
PY
  exit $?
fi

# ---- patch (default action) -------------------------------------------------
echo "=== Hotfix: DSV4 indexer short-context topk skip (upstream #49486 backport) ==="
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


# ---- attention.py: import tl, triton ----
patch(
    "models/deepseek_v4/attention.py",
    """from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.utils.multi_stream_utils import (""",
    """from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import (""",
    "attention.py: import tl, triton (upstream #49486)",
)

# ---- attention.py: all-candidates filler kernel (verbatim upstream #49486) --
patch(
    "models/deepseek_v4/attention.py",
    """logger = init_logger(__name__)


def _resolve_dsv4_kv_cache_dtype(""",
    """logger = init_logger(__name__)


@triton.jit
def _fill_short_context_topk_indices(
    output,
    positions,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    # small triton kernel that selects every candidate, -1 otherwise
    row = tl.program_id(0)
    offsets = tl.arange(0, PADDED_TOP_K)
    num_compressed = (tl.load(positions + row) + 1) // COMPRESS_RATIO
    tl.store(
        output + row * TOP_K + offsets,
        tl.where(offsets < num_compressed, offsets, -1),
        mask=offsets < TOP_K,
    )


def _resolve_dsv4_kv_cache_dtype(""",
    "attention.py: _fill_short_context_topk_indices kernel (upstream #49486)",
)

# ---- attention.py: short-context early return in DeepseekV4Indexer.forward ---
# Uses upstream verbatim: indexer metadata at self.k_cache.prefix is a
# DeepseekV32IndexerMetadata (num_decode_tokens + num_prefill_tokens).
patch(
    "models/deepseek_v4/attention.py",
    """    ) -> torch.Tensor:
        compressor = self.compressor

        def wq_b_and_q_quant():""",
    """    ) -> torch.Tensor:
        compressor = self.compressor

        # [PORT #49486] Short-context fast path: when every compressed candidate
        # would be selected (max_seq_len/ratio <= topk), the wq_b -> RoPE -> quant
        # -> QK logits -> top-k pipeline is a no-op: the result is all candidates.
        # Fill the buffer directly (bit-identical output; still build the K cache).
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
            if indexer_metadata.max_seq_len // self.compress_ratio <= self.topk_tokens:
                # candidates num smaller than topk, every candidate is selected
                # but we still need to build k cache
                compressor(compressed_kv_score, positions, rotary_emb)
                assert self.topk_indices_buffer is not None
                num_tokens = (
                    indexer_metadata.num_decode_tokens
                    + indexer_metadata.num_prefill_tokens
                )
                if num_tokens > 0:
                    _fill_short_context_topk_indices[(num_tokens,)](
                        self.topk_indices_buffer,
                        positions,
                        TOP_K=self.topk_tokens,
                        COMPRESS_RATIO=self.compress_ratio,
                        PADDED_TOP_K=triton.next_power_of_2(self.topk_tokens),
                        num_warps=8,
                    )
                return self.topk_indices_buffer

        def wq_b_and_q_quant():""",
    "attention.py: short-context early return (upstream #49486)",
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
    print("\nHotfix applied. Stop + start the vLLM process (or the container) to take effect.")
    print("Validate with: bash <this-script> --before  (pre-restart; run BEFORE restarting)")
    print("             bash <this-script> --after   (post-restart; KV budget must be unchanged)")
PYEOF

echo ""
echo "=== Verification ==="
bash "$0" --status
