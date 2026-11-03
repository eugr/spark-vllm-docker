# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from dc4dd5a2217bc757232e9fa4d8a8b59e207ca068:
# vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py

class MooncakeStoreCoordinator:
    def _reachable_masks(
        self,
        aligned_token_len: int,
        start_token: int,
        *,
        retention_interval: int | None,
        num_prompt_tokens: int | None,
        exclude_mamba: bool = False,
    ) -> tuple[list[bool] | None, ...]:
        mask_alignment = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.lcm_block_size
        )
        assert aligned_token_len % mask_alignment == 0, (
            f"aligned_token_len ({aligned_token_len}) must be a multiple of "
            f"{mask_alignment}"
        )
        reachable_boundaries = (
            ()
            if num_prompt_tokens is None
            else replay_boundaries(
                num_prompt_tokens,
                self.lcm_block_size,
                use_eagle=bool(self.eagle_group_ids),
            )
        )
        masks: list[list[bool] | None] = []
        for g_idx, g in enumerate(self.kv_cache_groups):
            spec = _coverage_spec(g.kv_cache_spec)
            end_block = aligned_token_len // spec.block_size
            start_block = min(end_block, max(0, cdiv(start_token, spec.block_size)))
            if exclude_mamba and isinstance(spec, MambaSpec):
                masks.append([False] * (end_block - start_block))
                continue
            manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
            assert manager_cls is not None
            use_eagle = g_idx in self.eagle_group_ids
            mask = manager_cls.reachable_block_mask(
                start_block=start_block,
                end_block=end_block,
                alignment_tokens=self.lcm_block_size,
                kv_cache_spec=spec,
                use_eagle=use_eagle,
                retention_interval=retention_interval,
                reachable_boundaries=reachable_boundaries,
                # Worker initialization has already resolved DCP in this spec.
                dcp_world_size=1,
            )
            if mask is not None:
                assert len(mask) == end_block - start_block
            masks.append(mask)
        return tuple(masks)

