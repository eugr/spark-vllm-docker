# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpts from upstream b409a4f0d787fdddb75ec2e8ddb6dfcfd36235b7 (with preceding series entries).
# vllm/v1/worker/gpu/model_runner.py

class GPUModelRunner:
    def _get_num_drafted_and_verified(
        self, input_batch: InputBatch
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Drafted and verified draft tokens per request.

        Must run before the drafter overwrites the draft tokens being verified.
        """
        if self.speculator is None or input_batch.num_draft_tokens_per_req is None:
            return None, None
        draft_tokens = self.req_states.draft_tokens[input_batch.idx_mapping]
        # -1 marks a draft slot without a draft token.
        num_drafted = (draft_tokens != -1).sum(dim=-1)
        num_bonus_tokens = self.model_state.num_new_sampled_tokens_per_step
        num_in_forward = input_batch.cu_num_logits.diff() - num_bonus_tokens
        num_verified = torch.minimum(num_in_forward, num_drafted)
        return num_drafted, num_verified

    def next_method(self):
        pass
