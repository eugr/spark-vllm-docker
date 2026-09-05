#!/usr/bin/env bash
# hotfix-dsv4-mtp-buffer-50312.sh — Free the unused DSV4 MTP hidden-state PP buffer
#
# Backport of upstream vLLM PR #50312 ("Fix redundant memory allocation and copy
# for dsv4 pp buffer, 448 MiB GPU memory saved") applied to the Anemll
# dspark-vllm-gx10 0.1.1 image (vLLM 0.25.2.dev0+g752a3a504.d20260714).
#
# WHAT IT FIXES: nvidia/model.py allocates `_mtp_hidden_buffer`
# (max_num_batched_tokens x hc_mult*hidden_size bf16 = 256 MiB per rank for
# DeepSeek-V4-Flash: 8192 x 16384 x 2 B) unconditionally on the last PP rank and
# runs a `copy_` into it on EVERY target forward step. The buffer is only
# consumed by the MTP / Eagle draft path (`get_mtp_target_hidden_states`).
# This deployment runs `method=dspark`; dspark/speculator.py explicitly does NOT
# consume it ("DSpark does not use the same pre-allocated buffer that
# DeepSeek-V4's MTP uses" — it drafts from mean-pooled aux hidden states), so
# on DSpark the allocation is dead weight and the per-step copy_ is wasted GPU
# work. Guarded on `use_eagle() or uses_draft_model()` (both False for dspark),
# matching upstream #50312.
#
# CRITICAL SAFETY NET (not in upstream #50312): gpu/model_runner.py calls
# `get_mtp_target_hidden_states()` and slices the result with NO None-guard at
# two sites (`pre_hc_hidden_states[: n]`). Once the buffer can be None, both
# sites must guard or DSpark decode crashes with `TypeError: 'NoneType' object
# is not subscriptable`. This backport adds those guards (mainline v0.27.0
# ships the same unguarded code — see analysis).
#
# Usage:
#   docker cp hotfix-dsv4-mtp-buffer-50312.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-mtp-buffer-50312.sh
#   # then stop + start the server yourself (this script never restarts it)
#
# Validation (run from the HOST, not in the container):
#   bash hotfix-dsv4-mtp-buffer-50312.sh --before   # capture pre-restart KV budget + prompt histogram
#   ... apply + restart yourself ...
#   bash hotfix-dsv4-mtp-buffer-50312.sh --after    # capture + diff vs baseline
#
# Apply on BOTH nodes (each node runs its own container). Idempotent.
set -euo pipefail

# ---- config -----------------------------------------------------------------
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
CONTAINER="${CONTAINER:-deepseek-v4-flash-vllm-dspark-1}"
API_URL="${API_URL:-http://127.0.0.1:8888}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")" && pwd)/../results}"
BASELINE_FILE="$RESULTS_DIR/hotfix-50312-kv-baseline.txt"
AFTER_FILE="$RESULTS_DIR/hotfix-50312-kv-after.txt"

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
# Approximate the C128A adaptive topk width this workload would use: within each
# bucket assume prompts spread to the bucket bound, take the midpoint, then
# width = clamp(next_pow2(mid/128), 128, 8192).
active_sum = 0.0
prev_c = 0.0
prev_b = 0.0
for b in sorted(counts):
    n = counts[b] - prev_c
    if n > 0:
        mid = (prev_b + b) / 2.0
        w = max(128, 2 ** math.ceil(math.log2(max(mid / 128.0, 1))))
        w = min(w, 8192)
        active_sum += n * w
    prev_b, prev_c = b, counts[b]
if total:
    avg_w = active_sum / total
    print(f"  est. avg C128A kernel loop width: {avg_w:.0f} / 8192"
          f"  (~{100 * (1 - avg_w / 8192):.0f}% fewer metadata-kernel iterations)")
PY
}

# ---- actions ----------------------------------------------------------------
if [ "$ACTION" = "--before" ] || [ "$ACTION" = "--baseline" ]; then
  mkdir -p "$RESULTS_DIR"
  capture_kv_snapshot "BEFORE (pre-restart)" "$BASELINE_FILE"
  echo ""; echo "Baseline histogram:"; print_histogram_summary "$BASELINE_FILE"
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
    arrow = "" if (b is None or a is None or b == a) else "  <-- CHANGED (expected: KV grows)"
    print(f"  {label:24} {str(b):>24} -> {str(a):>24}{arrow}")
PY
  echo ""; echo "After-start prompt-length histogram:"
  print_histogram_summary "$AFTER_FILE"
  echo ""
  echo "Expected #50312 effect: GPU KV cache size / num_gpu_blocks INCREASE"
  echo "(~256 MiB per rank ≈ +1-2k blocks / ~+37k tokens at 148k tokens/GiB)."
  exit 0
fi

if [ "$ACTION" = "--status" ]; then
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
m = (root / "models/deepseek_v4/nvidia/model.py").read_text()
r = (root / "v1/worker/gpu/model_runner.py").read_text()
guard = "needs_mtp_hidden_states"
cp = "if self._mtp_hidden_buffer is not None:"
rg = "if pre_hc_hidden_states is not None:"
print("model.py alloc guard  :", "APPLIED" if guard in m else "NOT APPLIED")
print("model.py copy_ guard  :", "APPLIED" if cp in m else "NOT APPLIED")
print("runner None-guard x2  :", "APPLIED" if r.count(rg) >= 2 else "NOT APPLIED")
PY
  exit 0
fi

# ---- patch (default action) -------------------------------------------------
echo "=== Hotfix: DSV4 MTP hidden-state PP buffer (upstream #50312 backport) ==="
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


# ---- model.py: only allocate the MTP buffer when a speculator consumes it ----
patch(
    "models/deepseek_v4/nvidia/model.py",
    """        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None""",
    """        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None""",
    "model.py: conditional _mtp_hidden_buffer allocation (dspark -> None)",
)

# ---- model.py: skip the per-step copy_ when the buffer does not exist -------
patch(
    "models/deepseek_v4/nvidia/model.py",
    """        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))""",
    """        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))""",
    "model.py: skip copy_ when buffer is None",
)

# ---- model_runner.py: None-guard BOTH draft-feeding sites (crash safety) ----
# The exact same 3-line block appears at model_runner.py:599 and :1463.
patch(
    "v1/worker/gpu/model_runner.py",
    """            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]""",
    """            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                if pre_hc_hidden_states is not None:
                    spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]""",
    "model_runner.py: None-guard at both get_mtp_target_hidden_states sites",
    expect=2,
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
