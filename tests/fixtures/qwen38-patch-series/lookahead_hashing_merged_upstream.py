# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# vllm/v1/core/kv_cache_utils.py at fe5fe8ba5fb9412ac6deafd192ae8b7c466b96c5.
# Upstream #61589 owns integer lookahead hashing; the preset extends connector gating.

def resolve_cache_hit_alignment_tokens(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
    scheduler_block_size: int,
    hash_block_size: int,
) -> int:
    """Token granularity at which prefix-cache hits land.

    Mirrors the alignment the hybrid KV cache coordinator pushes to its
    managers: fine-grained hits land on ``hash_block_size``, all others on
    ``scheduler_block_size``.
    """
    groups = kv_cache_config.kv_cache_groups
    if not vllm_config.cache_config.enable_prefix_caching or len(groups) <= 1:
        return scheduler_block_size
    manager_classes = []
    for group in groups:
        manager_cls = KVCacheSpecRegistry.get_manager_class(
            group.kv_cache_spec, group.role
        )
        assert manager_cls is not None
        manager_classes.append(manager_cls)
    if partial_hash_hits_enabled(
        groups,
        hash_block_size,
        vllm_config.parallel_config.decode_context_parallel_size,
        manager_classes,
    ):
        return hash_block_size
    return scheduler_block_size


def get_request_block_hasher(
    hash_block_size: int,
    caching_hash_fn: Callable[[Any], bytes],
    num_hash_lookahead_tokens: int = 0,
) -> Callable[[Request], list[BlockHash]]:
    """Returns a function which computes the list of un-computed block hashes
    of a request.

    Hashes are computed at ``hash_block_size`` granularity and chained over the
    full prefix, so each hash uniquely fingerprints the prefix ending at its
    boundary. Coarser group block sizes and partial-cache boundaries reuse
    these hashes directly (see ``BlockHashListWithBlockSize``).

    With ``num_hash_lookahead_tokens`` (L) > 0, each hash also covers the L
    tokens after its block, which the drafter's KV at the block's last position
    depends on, so it is emitted only once those tokens exist. The block tokens
    stay the hashed tokens; the extra keys become ``(E, *tokens)``, where ``E``
    covers the block and those tokens.
    """

    def request_block_hasher(request: Request) -> list[BlockHash]:
        start_token_idx = len(request.block_hashes) * hash_block_size
        num_tokens = request.num_tokens - num_hash_lookahead_tokens

        if start_token_idx + hash_block_size > num_tokens:
            # Early stop when there no new full blocks created.
            return []

        curr_mm_idx = 0
        mm_features = request.mm_features
        if start_token_idx > 0 and mm_features:
            last_mm_pos = mm_features[-1].mm_position
            if last_mm_pos.offset + last_mm_pos.length > start_token_idx:
                curr_mm_idx, _ = get_mm_features_in_window(
                    mm_features,
                    start_token_idx,
                    start_token_idx + hash_block_size,
                )

        prev_block_hash_value = (
            request.block_hashes[-1] if request.block_hashes else None
        )
        new_block_hashes: list[BlockHash] = []
        while True:
            end_token_idx = start_token_idx + hash_block_size
            if end_token_idx > num_tokens:
                # We only hash full blocks
                break

            # MM and LoRA requests need extra keys for block-hash computation.
            if num_hash_lookahead_tokens:
                hash_end_token_idx = end_token_idx + num_hash_lookahead_tokens
                extra_keys, curr_mm_idx = generate_lookahead_block_hash_extra_keys(
                    request,
                    start_token_idx,
                    end_token_idx,
                    hash_end_token_idx,
                    curr_mm_idx,
                )
                if num_hash_lookahead_tokens == 1:
                    extra_keys = (extra_keys, request.all_token_ids[end_token_idx])
                else:
                    extra_keys = (
                        extra_keys,
                        *request.all_token_ids[end_token_idx:hash_end_token_idx],
                    )
            else:
                extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                    request, start_token_idx, end_token_idx, curr_mm_idx
                )

            # Compute the hash of the current block
            block_tokens = request.all_token_ids[start_token_idx:end_token_idx]
            block_hash = hash_block_tokens(
                caching_hash_fn, prev_block_hash_value, block_tokens, extra_keys
            )

            new_block_hashes.append(block_hash)
            start_token_idx += hash_block_size
            prev_block_hash_value = block_hash

        return new_block_hashes

    return request_block_hasher


def generate_lookahead_block_hash_extra_keys(
    request: Request,
    start_token_idx: int,
    end_token_idx: int,
    hash_end_token_idx: int,
    start_mm_idx: int,
) -> tuple[tuple[Any, ...] | None, int]:
    """Generate extra keys for the block and its lookahead tokens.

    Rewind the MM index to retain inputs overlapping the next block.
    """
    extra_keys, mm_idx = generate_block_hash_extra_keys(
        request, start_token_idx, hash_end_token_idx, start_mm_idx
    )
    while mm_idx > 0:
        mm_position = request.mm_features[mm_idx - 1].mm_position
        if mm_position.offset + mm_position.length <= end_token_idx:
            break
        mm_idx -= 1
    return extra_keys, mm_idx

