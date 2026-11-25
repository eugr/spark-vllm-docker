# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpts from 790be385c8a69f45ec9a7a85ef00836d1684060f:
# vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py

class AutoRegressiveSpeculator(DraftModelSpeculator):
    def _prefill_capture_slot_mappings(self) -> torch.Tensor | None:
        return None

    def _configure_decode_metadata_update(self) -> None:
        if self.num_speculative_steps == 1:
            self.supports_decode_metadata_update = False
            return
        super()._configure_decode_metadata_update()

    def on_prefill_begin(self, num_reqs: int) -> None: ...

