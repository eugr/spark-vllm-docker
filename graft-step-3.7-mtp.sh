#!/bin/bash
# graft-step-3.7-mtp.sh
#
# Graft BF16 MTP (next-n predict) weights into the cached Step-3.7-Flash-NVFP4
# checkpoint so MTP speculative decoding works. The official NVFP4 export ships
# no MTP weights (ModelOpt strips layers 45-47 and truncates per-layer config
# lists during quantization), so the step-3.7-flash-nvfp4-mtp recipe needs them
# grafted back first.
#
# Usage (run on EACH Spark in the cluster, after downloading the model):
#     ./hf-download.sh stepfun-ai/Step-3.7-Flash-NVFP4 -c
#     ./graft-step-3.7-mtp.sh
#     ./run-recipe.sh step-3.7-flash-nvfp4-mtp
#
# Idempotent (skips if already grafted). Extraction runs inside the vllm-node
# container (needs torch + safetensors + huggingface_hub). Downloads ~5-7 GB
# (one shard of the original stepfun-ai/Step-3.7-Flash) the first time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HUB="$HF_HOME/hub"
IMAGE="${IMAGE:-vllm-node}"
REPO_DIR="$HUB/models--stepfun-ai--Step-3.7-Flash-NVFP4"

if [ ! -d "$REPO_DIR/snapshots" ]; then
  echo "Error: Step-3.7-Flash-NVFP4 not found in HF cache."
  echo "Run first: ./hf-download.sh stepfun-ai/Step-3.7-Flash-NVFP4 -c"
  exit 1
fi

SNAP_HOST="$(ls -d "$REPO_DIR"/snapshots/*/ | head -1)"
SNAP_NAME="$(basename "$SNAP_HOST")"
SNAP_CTR="/root/.cache/huggingface/hub/models--stepfun-ai--Step-3.7-Flash-NVFP4/snapshots/$SNAP_NAME"

echo "[graft] snapshot: $SNAP_HOST"
echo "[graft] running extraction inside $IMAGE ..."
docker run --rm --network host \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$SCRIPT_DIR/mods/step-3.7-flash:/mod:ro" \
  --entrypoint python3 "$IMAGE" /mod/graft_mtp.py "$SNAP_CTR"

echo "[graft] done. Serve with: ./run-recipe.sh step-3.7-flash-nvfp4-mtp"
echo "[graft] (cluster: run this script on each Spark, or copy the snapshot's"
echo "        model-mtp.safetensors + model.safetensors.index.json + config.json to peers.)"
