# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from 5c8ae7c74aaa99d1f7fbeea942fd7282e3413810:
# vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py

class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    _cache_hit_source = CacheHitSource.P2P

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        # A decode pulls the unhashed blocks of every transfer group, so the
        # recurrent-state slot arrives even when the attention prefix is local.
        # Roles that prefill locally keep the lookup that finds the shared
        # prefix boundary.
        return self._pulls_mamba_state

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id
        self._pulls_mamba_state = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
            and any(
                isinstance(group.kv_cache_spec, MambaSpec)
                for group in kv_cache_config.transfer_groups
            )
        )

        if role == KVConnectorRole.SCHEDULER:
            assert kv_cache_config is not None, (
                "kv_cache_config is required for SCHEDULER role"
            )
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

