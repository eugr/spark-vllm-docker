#!/bin/bash
# Install Ray into the container so vLLM can use the Ray distributed executor.
#
# The upstream qwen38-flash-next image (vllm/vllm-openai:qwen38-flash-next)
# ships WITHOUT Ray, but launch-cluster.sh's --ray path runs `ray start --head`
# / `ray start --address=...` inside each container and vLLM needs
# --distributed-executor-backend ray. This mod adds the `ray` package to the
# container's site-packages before those run (launch-cluster applies mods at
# container start, ahead of start_ray_head/start_ray_worker and the vLLM launch
# script).
#
# Note: this does NOT change NCCL transport selection. If the multi-node hang
# is on the RoCE fabric (NCCL_IB_DISABLE=0 + HCA in get_env_flags), Ray alone
# will not fix it -- pin NCCL to Ethernet (NCCL_IB_DISABLE=1) instead, see the
# qwen recipe's env: notes.
set -euo pipefail

echo "[install-ray] Installing ray into the container..."
if python3 -c "import ray" 2>/dev/null; then
    echo "[install-ray] ray already present: $(python3 -c 'import ray; print(ray.__version__)')"
else
    pip install --no-cache-dir ray
fi

python3 -c "import ray; print('[install-ray] ray', ray.__version__, 'imports OK')"
