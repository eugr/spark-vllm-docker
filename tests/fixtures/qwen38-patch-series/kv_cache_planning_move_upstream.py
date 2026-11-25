# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from 12b005eefa06f97eeebfae3e14bcac5fa3b81197:
# vllm/v1/core/kv_cache_utils.py after #61183 moved cache planning out.
# The hasher body is elided; its return and the next function are unchanged.


def get_request_block_hasher():
    return request_block_hasher


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """Get the maximum concurrency for the given KV cache configuration.

    A request at max_model_len consumes whole blocks from each group's block
    table — cdiv(per-request bytes, page bytes) of the group's spec — and all
    device groups draw those block ids from one shared pool, so the per-request
    total is the sum over groups. The memory/page ratio is identical whether
    a group carries an aggregated UniformTypeKVCacheSpecs (worker config) or
    a representative per-layer spec (scheduler config), so both capacity
    call sites agree.

    Host groups use a separate pool; the smaller concurrency limit applies.
    """
    num_blocks_per_request = 0
    host_blocks_per_request = 0
    for group in kv_cache_config.kv_cache_groups:
        required = cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        if group.host_resident:
            host_blocks_per_request += required
        else:
            num_blocks_per_request += required
    limits = [kv_cache_config.num_blocks / num_blocks_per_request]
    if host_blocks_per_request:
        assert kv_cache_config.hisparse_host_num_blocks is not None
        limits.append(
            kv_cache_config.hisparse_host_num_blocks / host_blocks_per_request
        )
    return min(limits)
