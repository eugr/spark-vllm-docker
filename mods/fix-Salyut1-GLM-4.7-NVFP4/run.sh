#!/bin/bash
set -e
patch -p1 -d / < glm4_moe.patch
# Arity fix: on newer vLLM (e.g. 0.23.1rc1.dev*), glm4_moe.py routes the scalar
# k/v scale through the stacked-param path and calls weight_loader(param, weight, shard_id)
# -- but KVCacheScaleParameter.weight_loader is a staticmethod taking (param, weight).
# shard_id is irrelevant for a scalar scale (per-head scales route elsewhere per its
# docstring), so absorb and ignore the extra positional.
sed -i 's/def weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor)/def weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args)/' \
  /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/kv_cache.py
