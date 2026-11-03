# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpt from vllm/v1/attention/backends/mamba_attn.py
# at vllm-project/vllm@c7a760751977e4e3396d586956d8767a9bc44a82.

class BaseMambaAttentionMetadataBuilder:
    def __init__(self):
        self._init_reorder_batch_threshold(1, self.use_spec_decode)
        if self.use_spec_decode:
            self.supports_update_block_table = False

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        pass

    def stage(self, metadata, state_indices_tensor_d, num_accepted_tokens, query_start_loc_d):
        if (
            metadata.num_prefills == 0
            and metadata.num_decodes <= self.decode_cudagraph_max_bs
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        ):
            padded_bs = metadata.num_reqs
            self.state_indices_tensor_d[: metadata.num_decodes].copy_(
                state_indices_tensor_d, non_blocking=True
            )
            state_indices_tensor_d = self.state_indices_tensor_d[:padded_bs]
            state_indices_tensor_d[metadata.num_decodes :] = NULL_BLOCK_ID

            if self.use_spec_decode and num_accepted_tokens is not None:
                assert query_start_loc_d is not None
                query_start_loc_d = query_start_loc_d[: padded_bs + 1]
                self.decode_num_accepted_tokens[: metadata.num_decodes].copy_(
                    num_accepted_tokens, non_blocking=True
                )
                num_accepted_tokens = self.decode_num_accepted_tokens[:padded_bs]
                num_accepted_tokens[metadata.num_decodes :] = (
                    1  # pad with 1st slot index
                )

        return state_indices_tensor_d, num_accepted_tokens
