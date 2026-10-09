#!/usr/bin/env bash
# hotfix-dsv4-adaptive-topk-50004.sh — Adaptive C128A topk width
#
# Backport of upstream vLLM PR #50004 ("Adaptive topk width, 1.0% E2E throughput
# improvement") applied to the Anemll dspark-vllm-gx10 0.1.1 image
# (vLLM 0.25.2.dev0+g752a3a504.d20260714).
#
# WHAT IT FIXES: the C128A decode/prefill topk buffers are pre-allocated at the
# MAX width (c128a_max_compressed = max_model_len/128 = 8192 for 1M context).
# Every decode step the _build_c128a_topk_metadata_kernel loops
# `0 .. max_compressed_tokens` (8 x 1024-wide blocks at 1M) even when the live
# context only reaches a few hundred compressed columns. The consumer caps via
# decode_lens (topk_length), but the kernel still burns the full loop.
#
# THE FIX (exactly upstream #50004): compute active_topk_width from the batch's
# max_seq_len each step, pass it as max_compressed_tokens, and return PACKED
# buffer views (`buffer.view(-1)[:T*W].view(T,W)`) so stride == active width
# and the kernel loops only over live columns. Buffers stay pre-allocated at
# full size (CUDA-graph address stability preserved); only the per-step views
# narrow. Downstream consumers are unchanged (decode wraps via
# `view(n, 1, -1)`; lens still capped by topk_length; pad slots stay -1).
#
# This image is at the exact pre-#50004 state: slices + buffer.stride(0) passed
# as the kernel stride. The Triton kernel already takes both *_stride and
# max_compressed_tokens args, so no kernel change is needed.
#
# Usage:
#   docker cp hotfix-dsv4-adaptive-topk-50004.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-adaptive-topk-50004.sh
#   # then stop + start the server yourself (this script never restarts it)
#
# Validation (run from the HOST, not in the container):
#   bash hotfix-dsv4-adaptive-topk-50004.sh --before   # capture pre-restart KV + prompt histogram
#   ... apply + restart yourself, replay typical load ...
#   bash hotfix-dsv4-adaptive-topk-50004.sh --after    # capture + diff
#
# Expected effects (per your live boot): NO KV-budget change (pure compute
# saving — KV blocks identical), correctness preserved, decode-step loop count
# reduced from 8192 to 128/256/512/1024/2048/4096 depending on context. The
# histogram summary prints the estimated avg loop width for your workload.
set -euo pipefail

# ---- config -----------------------------------------------------------------
VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
CONTAINER="${CONTAINER:-deepseek-v4-flash-vllm-dspark-1}"
API_URL="${API_URL:-http://127.0.0.1:8888}"
RESULTS_DIR="${RESULTS_DIR:-$(cd "$(dirname "$0")" && pwd)/../results}"
BASELINE_FILE="$RESULTS_DIR/hotfix-50004-baseline.txt"
AFTER_FILE="$RESULTS_DIR/hotfix-50004-after.txt"

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
  echo ""; echo "Baseline prompt-length histogram:"; print_histogram_summary "$BASELINE_FILE"
  echo ""
  echo "Next: apply the patch, restart the server yourself, replay typical load, then run:"
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
    chg = "" if (b is None or a is None or b == a) else "  <-- CHANGED (unexpected for #50004)"
    if chg: same = False
    print(f"  {label:24} {str(b):>24} -> {str(a):>24}{chg}")
if same:
    print("  -> KV budget unchanged: expected for #50004 (pure compute saving). Good.")
PY
  echo ""; echo "After-start prompt-length histogram:"
  print_histogram_summary "$AFTER_FILE"
  echo ""
  echo "Expected #50004 effect: same KV budget, same accuracy; decode-step C128A"
  echo "metadata-kernel loop reduced to the estimated width printed above."
  exit 0
fi

if [ "$ACTION" = "--status" ]; then
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
p = root / "models/deepseek_v4/sparse_mla.py"
t = p.read_text()
print("active_topk_width (adaptive) :", "APPLIED" if "active_topk_width" in t else "NOT APPLIED")
print("packed buffer views          :", "APPLIED" if "global_decode_buffer.view(-1)" in t else "NOT APPLIED")
print("kernel stride = active width :", "APPLIED" if t.count("        max_compressed_tokens,") >= 2 else "NOT APPLIED")
PY
  exit 0
fi

# ---- patch (default action) -------------------------------------------------
echo "=== Hotfix: DSV4 adaptive C128A topk width (upstream #50004 backport) ==="
echo "vLLM root: $VLLM_ROOT  image: 0.25.2.dev0+g752a3a504.d20260714"

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


# ---- sparse_mla.py: compute the active width from the live batch context ----
patch(
    "models/deepseek_v4/sparse_mla.py",
    """        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
            cm.positions[:num_total],
            self.compress_ratio,
            num_decode_tokens,
            req_id_per_token,
            cm.block_table_tensor[:num_decodes],
            block_size,
            cm.slot_mapping,
            self.c128a_global_decode_buffer,
            self.c128a_decode_lens_buffer,
            self.c128a_prefill_buffer,
            max_compressed_tokens=self.c128a_max_compressed,
        )""",
    """        block_size = self.kv_cache_spec.block_size // self.compress_ratio
        # Adaptive C128A topk width (upstream #50004): only the compressed
        # columns the current context can reach are valid, so run the kernel at
        # that width (stride == active width) instead of the full 1M width.
        active_topk_width = min(
            max(
                triton.next_power_of_2(
                    max(cm.max_seq_len // self.compress_ratio, 1)
                ),
                _C128A_TOPK_ALIGNMENT,
            ),
            self.c128a_max_compressed,
        )
        global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
            cm.positions[:num_total],
            self.compress_ratio,
            num_decode_tokens,
            req_id_per_token,
            cm.block_table_tensor[:num_decodes],
            block_size,
            cm.slot_mapping,
            self.c128a_global_decode_buffer,
            self.c128a_decode_lens_buffer,
            self.c128a_prefill_buffer,
            max_compressed_tokens=active_topk_width,
        )""",
    "sparse_mla.py: adaptive active_topk_width from cm.max_seq_len",
)

# ---- sparse_mla.py: return packed views and stride the kernel by live width --
patch(
    "models/deepseek_v4/sparse_mla.py",
    """    global_decode = global_decode_buffer[:num_decode_tokens]
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer[:num_prefill_tokens]

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    _build_c128a_topk_metadata_kernel[(num_tokens,)](
        global_decode_buffer,
        global_decode_buffer.stride(0),
        decode_lens_buffer,
        prefill_buffer,
        prefill_buffer.stride(0),""",
    """    # view(-1) as 1-d array and then expanded to
    # [num_decode_tokens, max_compressed_tokens] / [num_prefill_tokens, ...]
    global_decode = global_decode_buffer.view(-1)[
        : num_decode_tokens * max_compressed_tokens
    ].view(num_decode_tokens, max_compressed_tokens)
    decode_lens = decode_lens_buffer[:num_decode_tokens]
    prefill_local = prefill_buffer.view(-1)[
        : num_prefill_tokens * max_compressed_tokens
    ].view(num_prefill_tokens, max_compressed_tokens)

    if num_tokens == 0:
        return global_decode, decode_lens, prefill_local

    _build_c128a_topk_metadata_kernel[(num_tokens,)](
        global_decode_buffer,
        max_compressed_tokens,
        decode_lens_buffer,
        prefill_buffer,
        max_compressed_tokens,""",
    "sparse_mla.py: packed views + kernel stride = active width",
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
