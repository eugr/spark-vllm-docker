# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream propose() tail from 790be385c8a69f45ec9a7a85ef00836d1684060f:
# vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py

class TargetDependentARSpeculator:
    def propose_tail(self, input_batch, num_rejected, num_reqs,
                     dummy_run, skip_attn_for_dummy_run, dp_sync_state, is_profile,
                     num_speculative_tokens, prefill_batch_desc, attn_metadata=None,
                     slot_mappings=None, num_tokens_across_dp=None, mm_inputs=None):
        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:
            # Replay the full graph for draft prefill.
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(prefill_batch_desc)
        else:
            # The target model's attention metadata and slot mappings
            # can directly be used for draft prefill, because of the
            # identical batch shape and KV cache layout.
            self._prefill(
                num_reqs,
                prefill_batch_desc.num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=prefill_batch_desc.cg_mode,
                mm_inputs=mm_inputs,
                skip_sample=num_speculative_tokens == 0,
            )
        self.on_prefill_end(num_reqs)

        if num_speculative_tokens <= 1:
            if num_speculative_tokens == 0:
                self.draft_tokens[:num_reqs].fill_(-1)
            return self.draft_tokens[:num_reqs]

        if self.pcp_manager is not None and not dummy_run:
            self.block_tables.gather_block_tables(
                input_batch.idx_mapping, num_reqs_padded=num_reqs
            )

        self._run_multi_step_decode(
            input_batch,
            num_rejected,
            num_reqs,
            dummy_run and skip_attn_for_dummy_run,
            dp_sync_state,
            input_batch.seq_lens_cpu_upper_bound,
            is_profile,
            num_speculative_tokens,
        )

        return self.draft_tokens[:num_reqs]

