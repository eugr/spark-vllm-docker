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

# Resolve the snapshot referenced by refs/main (the current revision) so every
# cluster node grafts the same one; fall back to the newest snapshot.
REF_FILE="$REPO_DIR/refs/main"
if [ -f "$REF_FILE" ] && [ -d "$REPO_DIR/snapshots/$(cat "$REF_FILE")" ]; then
  SNAP_HOST="$REPO_DIR/snapshots/$(cat "$REF_FILE")/"
else
  SNAP_HOST="$(ls -dt "$REPO_DIR"/snapshots/*/ 2>/dev/null | head -1)"
fi
if [ -z "${SNAP_HOST:-}" ] || [ ! -d "$SNAP_HOST" ]; then
  echo "Error: no snapshot found under $REPO_DIR/snapshots"; exit 1
fi
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
