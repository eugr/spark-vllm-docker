# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from ad457ad7440bf0da7a7a8edd67ba9e66a9dec590:
# vllm/v1/worker/gpu_worker.py

class Worker:
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        with (
            self._maybe_get_memory_pool_context(tag="weights"),
            set_current_vllm_config(self.vllm_config),
            # 20 MiB is the minimum PyTorch allows for max_split_size_mb.
            self._scoped_allocator_max_split(max_split_size_mb=20),
            self._external_weight_charges.track(),
        ):
            self.model_runner.load_model(load_dummy_weights=load_dummy_weights)

        if self._external_weight_charges.fallback_bytes:
            self.requested_memory = self._resolve_requested_memory()
            logger.info(
                "Updated engine memory budget after disk fallback: %s GiB",
                format_gib(self.requested_memory),
            )

        if has_ec_transfer():
            get_ec_transfer().start_worker_services()

        if self.vllm_config.weight_transfer_config is not None:
            self.weight_transfer_engine = WeightTransferEngineFactory.create_engine(
                self.vllm_config.weight_transfer_config,
                self.vllm_config,
                self.device,
                self.model_runner.get_model(),
            )

        # Preserve parallel weight loading, then use serving's thread count
        # for profiling and compilation so Dynamo's global-state guards remain
        # valid when requests arrive.
        set_torch_threads_for_runtime()
