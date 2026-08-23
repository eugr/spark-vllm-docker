#!/usr/bin/env bash
# Run the outstanding Qwen3.8 tuning experiments back to back, unattended.
#
# Each experiment needs a different serve config, so each one costs a full
# engine restart (~4-8 min on this cluster, longer for the 1M recipe). Doing
# them by hand means babysitting a terminal for an hour; this drives the whole
# matrix and leaves a JSON file per variant to diff.
#
# It expects the cluster to itself. Stop the running unit first:
#
#   sudo systemctl stop vllm@qwen3.8-27b-nvfp4-1m.service
#
# Stopping the CONTAINERS is not enough. The vllm@ template carries
# Restart=on-failure with RestartSec=30, so `launch-cluster.sh stop` against a
# live unit looks like it worked and then systemd relaunches the old recipe
# ~30s later -- the next variant either measures the wrong model or dies on
# "Address already in use". The guard below refuses to start in that state.
#   tools/bench-campaign.sh                 # every variant
#   tools/bench-campaign.sh mtp-triton 1m-triton   # or a subset
#
# Results land in $OUTDIR (default /tmp/bench-campaign) as
# bench-<variant>.json and quality-<variant>.json, plus a run log each.
#
# NOTE: this script has not itself been executed end to end -- cluster access
# was unavailable when it was written. The pieces it calls (run-recipe.sh,
# bench-serving.py, quality-probe.py) have all been run individually against
# this cluster. Watch the first variant through before walking away.

set -uo pipefail

REPO="${REPO:-$HOME/spark-vllm-docker}"
OUTDIR="${OUTDIR:-/tmp/bench-campaign}"
BASE="${BASE:-https://localhost:8000}"
KEYFILE="${KEYFILE:-$HOME/.vllm-api-key}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-1500}"   # 1M recipe compiles for ~3.5 min
RECIPE_DIR="$REPO/recipes"
GEN_DIR="$OUTDIR/recipes"

mkdir -p "$OUTDIR" "$GEN_DIR"

# Each variant is a source recipe, the model id it serves, and a transform that
# rewrites the recipe on stdin. Transforms are functions rather than sed strings
# because one of them inserts a line, and escaping a newline through both the
# shell and sed is a reliable way to produce a silently truncated program.

transform_none() { cat; }

# One measured point to check the cost model's depth prediction. The model says
# k=5 should land ~4% below k=3; if it lands above, the model is wrong.
transform_k5() { sed 's/^  num_speculative_tokens: 3$/  num_speculative_tokens: 5/'; }

# The open 1M question: Triton buys decode but may cost long-context prefill,
# and prefill is what the 1M recipe actually spends its time on.
#
# Both flags are required. With only --attention-backend the drafter re-selects
# FlashInfer and the verify step stays outside CUDA graphs, which would make
# this variant look like a null result for the wrong reason.
transform_triton() {
  awk '
    /^    --reasoning-parser qwen3 \\$/ { print "    --attention-backend TRITON_ATTN \\" }
    /^    --speculative_config\.num_speculative_tokens \{num_speculative_tokens\}$/ {
      print $0 " \\"
      print "    --speculative_config.attention_backend TRITON_ATTN"
      next
    }
    { print }
  '
}

# Aggregate throughput was flat past 4 concurrent streams while the KV pool sat
# nowhere near its limit for realistically-sized requests.
transform_seqs8() { sed 's/^  max_num_seqs: 4$/  max_num_seqs: 8/'; }

# vLLM's boot log offers 93.79 GiB against the 82.98 GiB actually in use.
# The risk here is OOM under load, not correctness.
transform_kv() { sed 's/^  gpu_memory_utilization: 0.82$/  gpu_memory_utilization: 0.88/'; }

# Past the native 262,144-token window the MTP head accepts nothing at all
# (measured: acceptance length 1.00, per-position 0/0/0 at 287,519 tokens,
# against 3.12 at 254,652), so speculation there is three wasted drafter passes
# per step. Decode fell from 23.0 to 8.5 tok/s across that boundary. Since the
# 1M recipe exists precisely to serve contexts beyond 262k, this variant asks
# whether it should carry speculation at all.
transform_nospec() {
  sed -e '/^    --speculative_config\./d' \
      -e 's/^    --tool-call-parser qwen3_xml \\$/    --tool-call-parser qwen3_xml/'
}

# Depth targets per recipe family. The filler runs ~1.64 prompt tokens per unit
# of depth, so 175000 lands near 287k: past the native window on purpose for the
# 1M recipes, and past max_model_len entirely on the 262,144-token ones, where
# the request would simply be rejected.
DEPTHS_262K="1000,60000,140000"
DEPTHS_1M="1000,130000,175000"

# DSpark drafts a block of num_speculative_tokens tokens in ONE parallel pass
# (vllm/v1/worker/gpu/spec_decode/dspark/speculator.py), not k sequential ones
# like the MTP head, and the checkpoint is trained at block_size 7. So the cost
# model that made k=3 win for MTP should not transfer: a 3-wide pass costs
# nearly what a 7-wide one does on a bandwidth-bound device, while capping the
# reward at 3 accepted tokens. Measured here rather than argued.
transform_dspark_k3() { sed 's/^  num_speculative_tokens: 7$/  num_speculative_tokens: 3/'; }

# The ds4f filler runs ~0.87 prompt tokens per unit of depth (measured: depth
# 1000 -> 870 tokens), so 75000 lands just past the model's native 65,536-token
# YaRN window and 300000 sits deep into the scaled region.
DEPTHS_DS4F="1000,75000,300000"

# DeepSeek-V4-Flash speaks fp8_ds_mla KV at 584 B/token and cannot do better;
# these variants vary speculation depth and the attention stack instead.
transform_ds4f_k3() { sed 's/^  num_speculative_tokens: 5$/  num_speculative_tokens: 3/'; }

# The base recipe pairs DSpark k=5 with B12X_MLA_SPARSE. Past the native window
# the drafter may accept nothing, in which case five wasted drafter passes per
# step cost more than they return -- the same shape as the MTP cliff above.
transform_ds4f_nospec() { sed -e '/^      --speculative-config /d'; }

# Third-party DGX Spark writeups run this model with the attention backend on
# AUTO and the V2 model runner off, because their older build rejects both. This
# asks what that costs on our stack rather than assuming it costs nothing.
transform_ds4f_theirs() {
  sed -e '/^      --attention-backend B12X_MLA_SPARSE \\$/d' \
      -e 's/,"attention_backend":"B12X_MLA_SPARSE"//' \
      -e 's/^  VLLM_USE_V2_MODEL_RUNNER: "1"$/  VLLM_USE_V2_MODEL_RUNNER: "0"/'
}

# name : source recipe : served model id : transform function : depth list
variant_spec() {
  case "$1" in
    # Validates PR #7's headline number with the current harness rather than
    # the single-prompt script the 39.5 tok/s figure came from.
    mtp-triton)    echo "qwen3.8-27b-nvfp4|qwen3.8-27b|transform_none|$DEPTHS_262K" ;;
    mtp-triton-k5) echo "qwen3.8-27b-nvfp4|qwen3.8-27b|transform_k5|$DEPTHS_262K" ;;
    1m-triton)     echo "qwen3.8-27b-nvfp4-1m|qwen3.8-27b-1m|transform_triton|$DEPTHS_1M" ;;
    1m-nospec)     echo "qwen3.8-27b-nvfp4-1m|qwen3.8-27b-1m|transform_nospec|$DEPTHS_1M" ;;
    1m-seqs8)      echo "qwen3.8-27b-nvfp4-1m|qwen3.8-27b-1m|transform_seqs8|$DEPTHS_1M" ;;
    1m-kvheadroom) echo "qwen3.8-27b-nvfp4-1m|qwen3.8-27b-1m|transform_kv|$DEPTHS_1M" ;;
    dspark-k7)     echo "qwen3.8-27b-nvfp4-dspark|qwen3.8-27b|transform_none|$DEPTHS_262K" ;;
    dspark-k3)     echo "qwen3.8-27b-nvfp4-dspark|qwen3.8-27b|transform_dspark_k3|$DEPTHS_262K" ;;
    # DeepSeek-V4-Flash. ds4f-base is eugr's recipe untouched, which leaves
    # max_model_len on "auto" and therefore serves a different window on every
    # boot; ds4f-1m is the pinned 1,048,576 variant.
    ds4f-base)     echo "deepseek-v4-flash-0731|deepseek-ai/DeepSeek-V4-Flash-0731|transform_none|$DEPTHS_DS4F" ;;
    ds4f-1m)       echo "deepseek-v4-flash-0731-1m|deepseek-ai/DeepSeek-V4-Flash-0731|transform_none|$DEPTHS_DS4F" ;;
    ds4f-1m-k3)    echo "deepseek-v4-flash-0731-1m|deepseek-ai/DeepSeek-V4-Flash-0731|transform_ds4f_k3|$DEPTHS_DS4F" ;;
    ds4f-1m-nospec) echo "deepseek-v4-flash-0731-1m|deepseek-ai/DeepSeek-V4-Flash-0731|transform_ds4f_nospec|$DEPTHS_DS4F" ;;
    ds4f-1m-theirs) echo "deepseek-v4-flash-0731-1m|deepseek-ai/DeepSeek-V4-Flash-0731|transform_ds4f_theirs|$DEPTHS_DS4F" ;;
    *) return 1 ;;
  esac
}

ALL_VARIANTS="mtp-triton mtp-triton-k5 1m-nospec 1m-triton 1m-seqs8 1m-kvheadroom"
VARIANTS=("${@:-}")
[ -z "${VARIANTS[0]:-}" ] && read -ra VARIANTS <<< "$ALL_VARIANTS"

# A live vllm@ unit will relaunch its own recipe under us; see the note above.
systemd_guard() {
  local active
  active="$(systemctl list-units 'vllm@*' --state=active --no-legend 2>/dev/null | awk '{print $1}')"
  [ -n "$active" ] || return 0
  echo "refusing to run: active vllm unit(s): $active" >&2
  for unit in $active; do echo "  sudo systemctl stop $unit" >&2; done
  exit 1
}
systemd_guard

api_key() { sed -n 's/^VLLM_API_KEY=//p' "$KEYFILE" | tr -d "\"' \n"; }
KEY="$(api_key)"
[ -n "$KEY" ] || { echo "no API key in $KEYFILE" >&2; exit 1; }

server_up() {
  curl -sk -m 5 -o /dev/null -w '%{http_code}' \
    "$BASE/v1/models" -H "Authorization: Bearer $KEY" 2>/dev/null
}

stop_server() {
  ( cd "$REPO" && ./launch-cluster.sh stop >/dev/null 2>&1 )
  for _ in $(seq 1 60); do
    [ "$(server_up)" = "200" ] || return 0
    sleep 5
  done
  echo "  WARN: server still answering after stop" >&2
}

# Refuse to trample a server this script did not start.
if [ "$(server_up)" = "200" ]; then
  echo "A vLLM server is already answering on $BASE." >&2
  echo "Stop it first (sudo systemctl stop vllm@<recipe>.service) so this" >&2
  echo "script is not benchmarking someone else's config." >&2
  exit 1
fi

echo "campaign: ${VARIANTS[*]}"
echo "output:   $OUTDIR"
echo

for v in "${VARIANTS[@]}"; do
  spec="$(variant_spec "$v")" || { echo "unknown variant: $v" >&2; continue; }
  IFS='|' read -r src model xform depths <<< "$spec"

  echo "=============================================================="
  echo "[$v] from $src (serves $model)"
  echo "=============================================================="

  recipe_arg="$src"
  if [ "$xform" != "transform_none" ]; then
    gen="$GEN_DIR/${src}-${v}.yaml"
    "$xform" < "$RECIPE_DIR/$src.yaml" > "$gen"
    # A transform that changes nothing means the recipe text moved under it.
    # Benchmarking the unmodified recipe under the variant's name would quietly
    # produce a wrong answer, so refuse rather than continue.
    if diff -q "$RECIPE_DIR/$src.yaml" "$gen" >/dev/null; then
      echo "  ERROR: $xform changed nothing -- the recipe text moved." >&2
      echo "  Fix the transform before trusting this result. Skipping." >&2
      continue
    fi
    recipe_arg="$gen"
    echo "  generated $gen"
    diff "$RECIPE_DIR/$src.yaml" "$gen" | grep -E '^[<>]' | sed 's/^/    /'
  fi

  log="$OUTDIR/serve-$v.log"
  ( cd "$REPO" && nohup ./run-recipe.sh "$recipe_arg" -e "VLLM_API_KEY=$KEY" \
      > "$log" 2>&1 & )

  # Wait for readiness, but treat a dead launch as dead rather than slow. A
  # crashed engine looks exactly like a long boot to a poll loop that only ever
  # checks for success, so a transient failure costs the whole BOOT_TIMEOUT --
  # 25 minutes of dots for a launch that died in the first two.
  echo -n "  booting"
  ok=""; dead=""
  for _ in $(seq 1 $((BOOT_TIMEOUT / 10))); do
    if [ "$(server_up)" = "200" ]; then ok=1; break; fi
    if grep -qE "WorkerProc failed to start|Engine core initialization failed|raise e from None|EngineDeadError|torch.OutOfMemoryError|Traceback \(most recent call last\)" "$log" 2>/dev/null; then
      dead="engine reported a fatal error"; break
    fi
    # The launcher exiting with no server is also terminal.
    if ! pgrep -f "run-recipe" >/dev/null 2>&1 && [ -s "$log" ]; then
      dead="launcher exited without serving"; break
    fi
    echo -n "."
    sleep 10
  done
  echo

  if [ -z "$ok" ]; then
    if [ -n "$dead" ]; then
      echo "  FAILED: $dead" >&2
    else
      echo "  FAILED to come up within ${BOOT_TIMEOUT}s" >&2
    fi
    # Surface the actual exception, not the last 20 lines of shutdown noise.
    grep -m3 -E "Error|Exception|Traceback" "$log" 2>/dev/null | cut -c1-200 >&2 || true
    echo "  full log: $log" >&2
    stop_server
    continue
  fi

  # Record what the engine actually chose, not what the recipe asked for --
  # vLLM silently downgrades cudagraph_mode when the backend cannot support it.
  grep -hoE "Using [A-Z_]+ attention backend|not supported with spec-decode[^\"]*|GPU KV cache size: [0-9,]+ tokens" \
    "$log" | sort -u > "$OUTDIR/engine-$v.txt" 2>/dev/null
  echo "  engine facts -> $OUTDIR/engine-$v.txt"
  sed 's/^/    /' "$OUTDIR/engine-$v.txt" 2>/dev/null | head -5

  echo "  benchmarking..."
  python3 "$REPO/tools/bench-serving.py" --base "$BASE" --model "$model" \
    --suite all --repeats 3 --max-tokens 256 --concurrency 1,2,4,8 \
    --prefill-sizes 16000,64000 --depths "$depths" --label "$v" \
    --json "$OUTDIR/bench-$v.json" 2>&1 | sed 's/^/    /'

  echo "  quality probe..."
  python3 "$REPO/tools/quality-probe.py" --base "$BASE" --model "$model" \
    --no-think --label "$v" --out "$OUTDIR/quality-$v.json" 2>&1 | sed 's/^/    /'

  echo "  stopping..."
  stop_server
  echo
done

echo "=============================================================="
echo "done. results in $OUTDIR"
echo
echo "compare quality against a baseline with:"
echo "  python3 $REPO/tools/quality-probe.py --compare \\"
echo "      $OUTDIR/quality-mtp-triton.json $OUTDIR/quality-1m-triton.json"
