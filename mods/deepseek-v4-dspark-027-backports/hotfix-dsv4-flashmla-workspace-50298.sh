#!/usr/bin/env bash
# hotfix-dsv4-flashmla-workspace-50298.sh — Workspace reuse for combined topk indices
#
# Backport of upstream vLLM PR #50298 ("1.88x kernel improvement by removing a
# redundant full kernel" / workspace reuse for combined topk+SWA indices)
# applied to the Anemll dspark-vllm-gx10 0.1.1 image
# (vLLM 0.25.2.dev0+g752a3a504.d20260714).
#
# WHAT IT FIXES: on the FlashMLA sparse prefill path, every chunk calls
# combine_topk_swa_indices(), which allocates its (num_tokens, combined_topk)
# int32 indices buffer with torch.full(-1) and its lens buffer with
# torch.empty — fresh allocations plus a full-width -1 fill kernel on every
# chunk of every prefill step. Upstream's bit-exact test (rtol=0, atol=0 vs
# reference with topk_length=1) confirms the padding slots are never consumed:
# flash_mla_sparse_fwd reads only what topk_length (= combined_lens) allows.
#
# THE FIX (upstream #50298, ported verbatim):
#   - cache_utils.py: combine_topk_swa_indices() gains an optional
#     `out=(indices, lens)` parameter. When provided, the buffers are reused
#     and the full-width -1 pre-fill is skipped (padding is unread anyway).
#     Defaulted kwarg — the warmup and XPU callers are unaffected.
#   - flashmla.py forward_mqa() dummy path: reserve the two int32 buffers
#     alongside the bf16 gather workspace so capture-time reservation matches
#     the runtime request.
#   - flashmla.py _forward_prefill(): request all three buffers in one
#     get_simultaneous() call, slice to the chunk's token count, and pass them
#     via out=.
#
# MEMORY NOTE: the workspace reservation now includes the combined-topk
# buffers (max_num_batched_tokens x combined_topk x 4 B) that were previously
# transient caching-allocator peaks. Net effect on the KV budget should be
# ~neutral; --after flags anything beyond a small shift.
#
# Usage:
#   docker cp hotfix-dsv4-flashmla-workspace-50298.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-flashmla-workspace-50298.sh
#   # then stop + start the server yourself (this script never restarts it)
#
# Validation (run from the HOST, not in the container):
#   bash hotfix-dsv4-flashmla-workspace-50298.sh --before   # capture pre-restart KV + prompt histogram
#   ... apply + restart yourself, replay typical load ...
#   bash hotfix-dsv4-flashmla-workspace-50298.sh --after    # KV ~unchanged (+- small)
#
# Apply on BOTH nodes (each node runs its own container). Idempotent.
set -euo pipefail

# ---- config -----------------------------------------------------------------
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
CONTAINER="${CONTAINER:-deepseek-v4-flash-vllm-dspark-1}"
API_URL="${API_URL:-http://127.0.0.1:8888}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")" && pwd)/../results}"
BASELINE_FILE="$RESULTS_DIR/hotfix-50298-baseline.txt"
AFTER_FILE="$RESULTS_DIR/hotfix-50298-after.txt"

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
for label, pat, end in [
    ("Available KV cache memory", "Available KV cache memory: ", None),
    ("GPU KV cache size", "GPU KV cache size: ", None),
    ("num_gpu_blocks", 'num_gpu_blocks="', '"'),
]:
    b, a = grab(B, pat, end), grab(A, pat, end)
    arrow = "" if (b is None or a is None or b == a) else "  <-- shifted (small shift ok: workspace now reserves the combined-topk buffers)"
    print(f"  {label:24} {str(b):>24} -> {str(a):>24}{arrow}")
PY
  echo ""; echo "After-start prompt-length histogram:"
  print_histogram_summary "$AFTER_FILE"
  echo ""
  echo "Expected #50298 effect: KV budget ~unchanged; prefill chunks reuse"
  echo "workspace buffers for combined topk+SWA indices instead of allocating"
  echo "and -1-filling fresh ones per chunk (~1.88x on that kernel upstream)."
  exit 0
fi

if [ "$ACTION" = "--status" ]; then
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
fm = (root / "models/deepseek_v4/nvidia/flashmla.py").read_text()
cu = (root / "models/deepseek_v4/common/ops/cache_utils.py").read_text()
ok = True
def check(name, cond):
    global ok
    print(f"{name:52}:", "APPLIED" if cond else "NOT APPLIED")
    if not cond: ok = False
check("flashmla.py: import round_up", "from vllm.utils.math_utils import round_up" in fm)
check("flashmla.py: combined_topk width x2", fm.count("combined_topk = round_up(top_k + self.window_size, 128)") == 2)
check("flashmla.py: workspace 3-buffer unpack", "kv, combined_indices_out, combined_lens_out = workspace" in fm)
check("flashmla.py: out= kwarg passed", "out=(combined_indices_out, combined_lens_out)," in fm)
check("cache_utils.py: out param", "out: tuple[torch.Tensor, torch.Tensor] | None = None," in cu)
check("cache_utils.py: reuse branch", "combined_indices, combined_lens = out" in cu)
sys.exit(0 if ok else 1)
PY
  exit $?
fi

# ---- patch (default action) -------------------------------------------------
echo "=== Hotfix: DSV4 FlashMLA combined-topk workspace reuse (upstream #50298 backport) ==="
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


# ---- flashmla.py: import round_up (upstream #50298) ---------------------------
patch(
    "models/deepseek_v4/nvidia/flashmla.py",
    """from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.v1.attention.ops.flashmla import (""",
    """from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.utils.math_utils import round_up
from vllm.v1.attention.ops.flashmla import (""",
    "flashmla.py: import round_up (upstream #50298)",
)

# ---- flashmla.py: dummy path reserves the combined-topk buffers --------------
patch(
    "models/deepseek_v4/nvidia/flashmla.py",
    """            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return""",
    """            M = N + self.window_size + self.max_num_batched_tokens
            # [PORT #50298] Reserve the combined-topk int32 buffers alongside
            # the gather workspace so capture-time reservation matches the
            # runtime request in _forward_prefill below.
            assert self.topk_indices_buffer is not None
            top_k = 0 if swa_only else self.topk_indices_buffer.shape[-1]
            combined_topk = round_up(top_k + self.window_size, 128)
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            output.zero_()
            return""",
    "flashmla.py: dummy path reserves combined-topk buffers (upstream #50298)",
)

# ---- flashmla.py: _forward_prefill requests all three buffers ----------------
patch(
    "models/deepseek_v4/nvidia/flashmla.py",
    """        workspace_manager = current_workspace_manager()
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            kv = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
            )[0]""",
    """        workspace_manager = current_workspace_manager()
        # [PORT #50298] Combined topk+SWA index/lens buffers come from the
        # workspace (reserved in forward_mqa's dummy pass) instead of fresh
        # torch.full / torch.empty allocations per chunk.
        combined_topk = round_up(top_k + self.window_size, 128)
        for chunk_start, chunk_end, chunk_N, chunk_M in chunk_plan:
            chunk_size = chunk_end - chunk_start
            workspace = workspace_manager.get_simultaneous(
                ((chunk_size, chunk_M, q.shape[-1]), torch.bfloat16),
                ((self.max_num_batched_tokens, combined_topk), torch.int32),
                ((self.max_num_batched_tokens,), torch.int32),
            )
            kv, combined_indices_out, combined_lens_out = workspace""",
    "flashmla.py: _forward_prefill workspace request (upstream #50298)",
)

# ---- flashmla.py: slice the reused buffers to the chunk's token count --------
patch(
    "models/deepseek_v4/nvidia/flashmla.py",
    """            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(""",
    """            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )
            combined_indices_out = combined_indices_out[: query_end - query_start]
            combined_lens_out = combined_lens_out[: query_end - query_start]

            combined_indices, combined_lens = combine_topk_swa_indices(""",
    "flashmla.py: slice reused buffers per chunk (upstream #50298)",
)

# ---- flashmla.py: pass the reused buffers via out= ----------------------------
patch(
    "models/deepseek_v4/nvidia/flashmla.py",
    """                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
            )""",
    """                self.window_size,
                self.compress_ratio,
                top_k,
                chunk_M,
                chunk_N,
                out=(combined_indices_out, combined_lens_out),
            )""",
    "flashmla.py: combine_topk_swa_indices out= kwarg (upstream #50298)",
)

# ---- cache_utils.py: combine_topk_swa_indices gains optional out= ------------
# Padding slots are capped by combined_lens (topk_length) and never read by
# flash_mla_sparse_fwd, so the -1 pre-fill is skipped when reusing buffers.
patch(
    "models/deepseek_v4/common/ops/cache_utils.py",
    """    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )""",
    """    topk: int,
    M: int,
    N: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    if out is None:
        combined_indices = torch.full(
            (num_tokens, combined_topk),
            fill_value=-1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
        combined_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        combined_indices, combined_lens = out""",
    "cache_utils.py: combine_topk_swa_indices out= param (upstream #50298)",
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
    print("             bash <this-script> --after   (post-restart; compare KV budget)")
PYEOF

echo ""
echo "=== Verification ==="
bash "$0" --status
