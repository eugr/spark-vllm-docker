#!/usr/bin/env bash
# Long-context recipe: serve Qwen3.8-Flash-Next with vLLM, NVFP4 weights, and the model's
# own MTP draft head. OpenAI-compatible API on $PORT.
#
#   ./serve.sh                      defaults below - full 262,144 context
#   MTP=0 ./serve.sh                no speculative decoding, for an A/B
#   CTX=32768 ./serve.sh            smaller context, more KV headroom
#   docker logs -f qwen38-flash     watch it load
#
# Tunables:
#   PORT=8000        host port (loopback only - see BIND below)
#   BIND=127.0.0.1   set to 0.0.0.0 to expose on your LAN. Think before you do.
#   CTX=262144       max context. Native maximum; KV pool measured at 641,601 tokens.
#   SEQS=16          concurrent sequences. Past 16, TTFT collapses - see README
#   GPU_MEM=0.85     fraction of the 128 GB pool for weights + KV
#   MTP=3            speculative tokens from the model's MTP head (0 disables)
#   PREWARM=1        stream the n-gram table once at boot so the first request is not cold
#   PREFIX_CACHE=1   reuse KV for repeated prefixes. 1.76x aggregate and less than half the
#                    first-token latency on a shared-prefix workload - see README. Set 0 to
#                    reproduce the published prefill figures, which were all measured
#                    cache-free.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The version of the recipes, not of vLLM. A default changing here moves your numbers.
RECIPE_VERSION="$(cat "$HERE/../../VERSION" 2>/dev/null || echo unknown)"
ROOT="${ROOT:-$HOME/.qwen38fn-longctx}"
SRC="${SRC:-$ROOT/qwen3.8-Flash-DGX}"
NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8000}"
BIND="${BIND:-127.0.0.1}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-16}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-3}"
PREWARM="${PREWARM:-1}"

die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image '$IMAGE' not built. Run ./setup.sh first."

REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
SNAP_HOST="$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | head -1 || true)"
[ -z "$SNAP_HOST" ] && die "checkpoint not found under $REPO_DIR. Run ./setup.sh first."
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$(basename "$SNAP_HOST")"

# The n-gram gather is a CPU op followed by a host->device copy, so it MUST stay outside CUDA
# graphs. Declaring it a splitting op and capturing PIECEWISE is what makes that work; FULL
# capture modes break. --enforce-eager also works and is slower.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

SPEC=""
[ "$MTP" != 0 ] && SPEC="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$NAME" --gpus all --ipc=host --shm-size 16g \
  -p "${BIND}:${PORT}:8000" \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}" -e VLLM_PLE_MMAP_PREWARM="$PREWARM" \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  "$IMAGE" \
  "$SNAP_IN" --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port 8000 --load-format safetensors \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
    $( [ "$PREFIX_CACHE" = 0 ] && echo --no-enable-prefix-caching || echo --enable-prefix-caching ) \
    --enable-chunked-prefill --max-num-batched-tokens 8192 \
    $CC \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    $SPEC

# Built by setup.sh from a moving branch, so say which commit this image actually is.
BUILT_FROM=$(docker image inspect -f '{{index .Config.Labels "de.qwen38fn.upstream-sha"}}' "$IMAGE" 2>/dev/null)
[ -n "$BUILT_FROM" ] && [ "$BUILT_FROM" != "<no value>" ] || BUILT_FROM="unknown (image predates the build label)"

cat <<EOF
>> $NAME starting on ${BIND}:${PORT}, context $CTX, MTP=$MTP, recipe v${RECIPE_VERSION}
>> image built from upstream $BUILT_FROM
>> first boot loads ~83 GiB of weights and takes 12-15 minutes. Watch:
     docker logs -f $NAME
>> ready when a real completion returns 200 - /v1/models answers earlier than that, so:
     until curl -sf -o /dev/null -X POST http://127.0.0.1:${PORT}/v1/chat/completions \\
       -H 'Content-Type: application/json' \\
       -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'; \\
       do sleep 15; done; echo READY
EOF
