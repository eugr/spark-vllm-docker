#!/usr/bin/env bash
# hotfix-dsv4-skip-empty-c128-48957.sh — Skip empty C128 compressor launches
#
# Backport of upstream vLLM PR #48957 ("~2x kernel improvement by skipping
# empty c128 launches") applied to the Anemll dspark-vllm-gx10 0.1.1 image
# (vLLM 0.25.2.dev0+g752a3a504.d20260714).
#
# WHAT IT FIXES: every forward step, the C128 (compress_ratio=128, head_dim=512)
# compressor runs its compress -> RMSNorm -> RoPE -> quant -> K-cache-write
# kernel — even on steps where NO request crosses a 128-token boundary, i.e.
# steps where zero new compressed entries could be written. At long context
# that is the large majority of decode steps (~127 of 128 per request).
#
# THE FIX (upstream #48957, ported close to verbatim):
#   - _get_c128_boundary(): per-step CPU check over num_computed_tokens and
#     query_start_loc — True if ANY request emits a compressed token this step.
#   - CompressorMetadata.c128_boundary: computed in build() for C128 layers
#     only (block_size == 8); None for C4 layers and dummy runs.
#   - Compressor.forward(): after save_partial_states (the raw state cache is
#     ALWAYS written — correctness preserved), return early when
#     c128_boundary is False. Gate conditions (upstream verbatim):
#     CUDA platform, head_dim == 512, compress_ratio == 128, NOT full-graph
#     capture (a captured graph cannot branch on per-step CPU metadata), and
#     c128_boundary is False.
#
# NOTES / LIMITS:
#   - Fires only when cudagraph_runtime_mode != FULL. Under piecewise (the
#     default here) decode steps take the skip; FULL-graph decode cannot.
#   - The C4 indexer compressor (head_dim=128, ratio=4) is never skipped.
#   - Pure compute saving: KV budget, numerics, and accuracy are unchanged.
#
# Usage:
#   docker cp hotfix-dsv4-skip-empty-c128-48957.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-skip-empty-c128-48957.sh
#   # then stop + start the server yourself (this script never restarts it)
#
# Validation (run from the HOST, not in the container):
#   bash hotfix-dsv4-skip-empty-c128-48957.sh --before   # capture pre-restart KV + prompt histogram
#   ... apply + restart yourself ...
#   bash hotfix-dsv4-skip-empty-c128-48957.sh --after    # KV budget must be unchanged
#
# Apply on BOTH nodes (each node runs its own container). Idempotent.
set -euo pipefail

# ---- config -----------------------------------------------------------------
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
CONTAINER="${CONTAINER:-deepseek-v4-flash-vllm-dspark-1}"
API_URL="${API_URL:-http://127.0.0.1:8888}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")" && pwd)/../results}"
BASELINE_FILE="$RESULTS_DIR/hotfix-48957-baseline.txt"
AFTER_FILE="$RESULTS_DIR/hotfix-48957-after.txt"

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
import re, sys
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
    chg = "" if (b is None or a is None or b == a) else "  <-- CHANGED (unexpected for #48957)"
    if chg: same = False
    print(f"  {label:24} {str(b):>24} -> {str(a):>24}{chg}")
if same:
    print("  -> KV budget unchanged: expected for #48957 (pure compute saving). Good.")
PY
  echo ""; echo "After-start prompt-length histogram:"
  print_histogram_summary "$AFTER_FILE"
  echo ""
  echo "Expected #48957 effect: same KV budget, same accuracy; on decode steps"
  echo "where no request crosses a 128-token boundary (most of them at long"
  echo "context), the C128 compress->quant->K-cache kernel is skipped entirely."
  exit 0
fi

if [ "$ACTION" = "--status" ]; then
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
t = (root / "models/deepseek_v4/compressor.py").read_text()
ok = True
def check(name, cond):
    global ok
    print(f"{name:46}:", "APPLIED" if cond else "NOT APPLIED")
    if not cond: ok = False
check("import CUDAGraphMode", "from vllm.config import CUDAGraphMode, VllmConfig" in t)
check("_get_c128_boundary helper", "def _get_c128_boundary(" in t)
check("CompressorMetadata.c128_boundary field", "c128_boundary: bool | None = None" in t)
check("build() populates c128_boundary", "c128_boundary=(" in t)
check("forward skip gate", "state_metadata.c128_boundary is False" in t)
sys.exit(0 if ok else 1)
PY
  exit $?
fi

# ---- patch (default action) -------------------------------------------------
echo "=== Hotfix: DSV4 skip empty C128 compressor launches (upstream #48957 backport) ==="
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


# ---- compressor.py: import CUDAGraphMode (upstream #48957) -------------------
patch(
    "models/deepseek_v4/compressor.py",
    "from vllm.config import VllmConfig, get_current_vllm_config",
    "from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config",
    "compressor.py: import CUDAGraphMode (upstream #48957)",
)

# ---- compressor.py: _get_c128_boundary helper (verbatim upstream #48957) -----
patch(
    "models/deepseek_v4/compressor.py",
    """@dataclass
class CompressorMetadata:
    block_table: torch.Tensor""",
    """def _get_c128_boundary(metadata: CommonAttentionMetadata) -> bool | None:
    starts = metadata._num_computed_tokens_cpu
    if starts is None:
        return None

    starts_list = starts.tolist()
    query_start_loc = metadata.query_start_loc_cpu.tolist()
    return any(
        start % 128 + query_start_loc[i + 1] - query_start_loc[i] >= 128
        for i, start in enumerate(starts_list)
    )


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor""",
    "compressor.py: _get_c128_boundary helper (upstream #48957)",
)

# ---- compressor.py: CompressorMetadata.c128_boundary field -------------------
patch(
    "models/deepseek_v4/compressor.py",
    "    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]\n",
    """    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    # [PORT #48957] True if any request crosses a 128-token compressed-KV
    # boundary this step; None when unknown (dummy run). C128 layers only.
    c128_boundary: bool | None = None
""",
    "compressor.py: CompressorMetadata.c128_boundary field (upstream #48957)",
)

# ---- compressor.py: build() populates c128_boundary (C128 layers only) -------
patch(
    "models/deepseek_v4/compressor.py",
    """        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )""",
    """        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            # [PORT #48957] C128 compressor blocks are 8 wide; only those
            # layers pay the CPU tolist() for the boundary check.
            c128_boundary=(
                _get_c128_boundary(common_attn_metadata)
                if self.block_size == 8
                else None
            ),
        )""",
    "compressor.py: build() populates c128_boundary (upstream #48957)",
)

# ---- compressor.py: capture forward_context (gate needs runtime mode) --------
patch(
    "models/deepseek_v4/compressor.py",
    """        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return""",
    """        # Get the metadata and handle dummy profiling run.
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return""",
    "compressor.py: capture forward_context (upstream #48957)",
)

# ---- compressor.py: skip gate after save_partial_states ----------------------
# The state cache write above ALWAYS runs; only the compress->K-cache kernel is
# skipped when no request emits a compressed token this step.
patch(
    "models/deepseek_v4/compressor.py",
    """            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.""",
    """            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # [PORT #48957] full graph cannot branch on per-step CPU metadata
        # after capture, so the skip is disabled under CUDAGraphMode.FULL.
        if (
            current_platform.is_cuda()
            and self.head_dim == 512
            and self.compress_ratio == 128
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            and state_metadata.c128_boundary is False
        ):
            return

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.""",
    "compressor.py: skip empty C128 launch gate (upstream #48957)",
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
