# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from 564ac0de1b228568cafcbfbe8ccaccc82d6023eb.

class EngineCore:
    def init_hash(self, vllm_config, kv_connector, hash_block_size):
        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        # Lets frontends hash prompt blocks; see EngineCoreRequest.block_hashes.
        self.prefix_hash_config: PrefixHashConfig | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                hash_block_size,
                caching_hash_fn,
                vllm_config.num_hash_lookahead_tokens,
            )
            self.prefix_hash_config = PrefixHashConfig(
                hash_algo=vllm_config.cache_config.prefix_caching_hash_algo,
                hash_block_size=hash_block_size,
                none_hash=kv_cache_utils.NONE_HASH,
                num_lookahead_tokens=vllm_config.num_hash_lookahead_tokens,
            )

        self.step_fn = (
            None
        )
