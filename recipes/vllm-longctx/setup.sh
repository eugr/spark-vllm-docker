#!/usr/bin/env bash
# Long-context recipe: build the vLLM image and fetch the NVFP4 checkpoint.
#
#   ./setup.sh
#
# What this does, and why it is a fetch rather than a vendored copy:
#
# The container that makes this work is blazux/qwen3.8-Flash-DGX (Apache-2.0). Its contribution
# is a patch that serves the 51.2B n-gram table from disk instead of keeping it resident, which
# is the only reason a 122 GiB checkpoint fits next to a usable KV cache on one box. That patch
# is their work and stays in their repository; this recipe clones it, builds it, and then adds
# only the serving configuration we measured. See ../../CREDITS.md.
set -euo pipefail

ROOT="${ROOT:-$HOME/.qwen38fn-longctx}"
SRC="${SRC:-$ROOT/qwen3.8-Flash-DGX}"
UPSTREAM="${UPSTREAM:-https://github.com/blazux/qwen3.8-Flash-DGX.git}"
UPSTREAM_REF="${UPSTREAM_REF:-main}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v git    >/dev/null || die "git not found"
command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1  || die "cannot talk to the docker daemon (permissions? is it running?)"

free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "${free_gb:-0}" -lt 140 ] && cat <<EOF
WARNING: only ${free_gb} GB free under $HOME.
         The checkpoint is ~126 GB and the image ~21 GB. Expect this to fail.
EOF

mkdir -p "$ROOT" "$HF_CACHE"

if [ ! -d "$SRC/.git" ]; then
  log "cloning the upstream container recipe (blazux/qwen3.8-Flash-DGX, Apache-2.0)"
  git clone "$UPSTREAM" "$SRC"
fi
git -C "$SRC" fetch --depth 50 origin "$UPSTREAM_REF" --force
git -C "$SRC" checkout -q "$UPSTREAM_REF"
git -C "$SRC" pull -q --ff-only origin "$UPSTREAM_REF" || true
UPSTREAM_SHA=$(git -C "$SRC" rev-parse --short HEAD)
log "upstream at $UPSTREAM_SHA"

# Stamp the upstream commit into the image. UPSTREAM_REF defaults to a moving branch, so two
# people running this a week apart get materially different servers - prefix-caching behaviour
# among them - and neither can tell from the outside which one they have. A label makes the
# build self-describing, so a bug report against this recipe can name a commit instead of a day.
log "building $IMAGE (pulls a multi-GB base image on first run)"
docker build -t "$IMAGE" \
  --label "de.qwen38fn.upstream-repo=$UPSTREAM" \
  --label "de.qwen38fn.upstream-ref=$UPSTREAM_REF" \
  --label "de.qwen38fn.upstream-sha=$UPSTREAM_SHA" \
  "$SRC"

log "fetching $MODEL into $HF_CACHE (~126 GB, resumable)"
# HF_HUB_DISABLE_XET=1: the Xet backend stalls on some Spark setups - it reports files as
# fetched while writing no blobs. Plain HTTPS saturates the link and is reliable.
docker run --rm --name qwen38fn-dl \
  -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 \
  -v "$HF_CACHE:/hf" --entrypoint bash "$IMAGE" \
  -c "hf download '$MODEL' --max-workers 8"

log "done. start it with:  ./serve.sh"
