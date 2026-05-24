#!/bin/bash
# Guarded ARK Kimi K2.6 native INT4 launch script.

export HF_DATASETS_OFFLINE="1"
export HF_HUB_DISABLE_TELEMETRY="1"
export HF_HUB_OFFLINE="1"
export NCCL_IB_DISABLE="0"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_ROCE_VERSION_NUM="2"
export NCCL_IB_TIMEOUT="22"
export NCCL_IB_RETRY_CNT="13"
export NCCL_IB_QPS_PER_CONNECTION="1"
export NCCL_IB_MERGE_NICS="1"
export NCCL_IB_SPLIT_DATA_ON_HCAS="1"
export NCCL_IB_PCI_RELAXED_ORDERING="1"
export NCCL_IB_TC="106"
export NCCL_IB_SL="0"
export NCCL_IB_ADAPTIVE_ROUTING="0"
export NCCL_CROSS_NIC="1"
export NCCL_NET_MERGE_LEVEL="NODE"
export NCCL_NET_GDR_LEVEL="0"
export NCCL_NET_GDR_C2C="1"
export NCCL_PXN_C2C="1"
export NCCL_CUMEM_ENABLE="0"
export NCCL_NVLS_ENABLE="0"
export NCCL_ALGO="Ring"
export NCCL_BUFFSIZE="4194304"
export NCCL_NTHREADS="512"
export NCCL_MIN_NCHANNELS="16"
export NCCL_IGNORE_CPU_AFFINITY="1"
export NCCL_LAZY_CONNECT="1"
export NCCL_SOCKET_RETRY_CNT="50"
export NCCL_SOCKET_RETRY_SLEEP_MSEC="150"
export NCCL_NET_PLUGIN="none"
export NCCL_DEBUG="INFO"
export NCCL_MIN_CTAS="32"
export NCCL_SOCKET_FAMILY="AF_INET"
export RAY_CGRAPH_get_timeout="900"
export RAY_USAGE_STATS_ENABLED="0"
export TRANSFORMERS_OFFLINE="1"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
export VLLM_DISABLE_MOE_WNA16_CUDA="1"
export VLLM_DISABLE_WNA16_MARLIN_MOE="1"
export VLLM_NO_USAGE_STATS="1"
export VLLM_USE_FLASHINFER_MOE_FP4="1"

vllm serve moonshotai/Kimi-K2.6 \
  --served-model-name moonshotai/Kimi-K2.6 \
  --kv-cache-dtype fp8 \
  --safetensors-load-strategy lazy \
  --mm-encoder-tp-mode data \
  --attention-backend TRITON_MLA \
  --skip-mm-profiling \
  --compilation_config.pass_config.fuse_allreduce_rms true \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --gpu-memory-utilization 0.76 \
  --max-model-len 16384 \
  --max-num-seqs 1 \
  --host 0.0.0.0 \
  --port 30020 \
  --tensor-parallel-size 8 \
  --distributed-executor-backend ray
