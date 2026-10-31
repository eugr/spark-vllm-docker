# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from fe5fe8ba5fb9412ac6deafd192ae8b7c466b96c5:
# vllm/v1/core/kv_cache_coordinator.py

def replay_boundaries(
    num_prompt_tokens: int, scheduler_block_size: int, use_eagle: bool
) -> tuple[int, ...]:
    """Positions a later request replaying a prompt can resume at.

    A hit is the shortest across all groups, so every group retains state
    at each position; EAGLE groups also keep the block above, which they
    match and drop back from (see ``reachable_block_mask``).

    Two positions are reachable: a resend of the identical prompt is capped
    at ``num_tokens - 1`` (its last token is recomputed for logits), a
    longer sibling matches the final aligned block. They differ only on a
    block-aligned prompt, where retaining just the higher one collapses the
    resend's hit to 0. The alignment is the scheduler block size, not the
    finer hash granularity, which would over-estimate the reach.
    """
    if not use_eagle:
        return (num_prompt_tokens - 1,)
    block = scheduler_block_size
    resend = (num_prompt_tokens - 1) // block * block
    extension = num_prompt_tokens // block * block
    return tuple(sorted({max(resend - block, 0), max(extension - block, 0)}))

class KVCacheCoordinator:
    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request.

        Args:
            request: The request.
            num_computed_tokens: The total number of tokens
                that need to be cached
                (including tokens that are already cached).

        """
        boundaries = replay_boundaries(
            request.num_prompt_tokens,
            self.scheduler_block_size,
            use_eagle=bool(self.drop_eagle_group_ids),
        )
        for group_id, manager in enumerate(self.single_type_managers):
            if not manager.enable_caching:
                continue
            manager.cache_blocks(
                request,
                self.get_num_cacheable_tokens(num_computed_tokens, group_id),
                retention_interval=self.retention_interval,
                replay_boundaries=boundaries,
            )

