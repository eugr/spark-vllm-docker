# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpts from fe341a53dda00472c281a2d1f4bd34a74e7eae6e:
# vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py

class TargetDependentARSpeculator(DraftModelSpeculator):
    @property
    def advance_draft_positions(self) -> bool:
        """Whether to increment positions and seq_lens between draft steps.

        True for Eagle/standard MTP (each step produces new KV).
        False for Gemma4 MTP (Q-only, shares target KV, constant positions).
        """
        return True

    def _configure_decode_metadata_update(self) -> None:
        # With one speculative step there are no draft decode steps to update.
        if self.num_speculative_steps == 1:
            self.supports_decode_metadata_update = False
            return
        super()._configure_decode_metadata_update()

