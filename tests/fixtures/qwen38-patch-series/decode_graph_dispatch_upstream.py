# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpt from vllm/v1/worker/gpu/model_runner.py
# at vllm-project/vllm@c7a760751977e4e3396d586956d8767a9bc44a82.

class Runner:
    def prepare_dispatch(self, scheduler_output, dummy_run):
        # Get batch descriptor and sync across DP ranks.
        num_reqs = len(scheduler_output.num_scheduled_tokens)
        num_toks = scheduler_output.total_num_scheduled_tokens
        max_query_len: int | None = max(scheduler_output.num_scheduled_tokens.values())
        batch_req_state, uniform_tok_count = self.gather_batch_req_state(
            scheduler_output, dummy_run
        )
        if batch_req_state is not None:
            num_toks = batch_req_state.num_tokens
            if not batch_req_state.decode_graph_eligible:
                # A query-length bound alone would admit a short prefill.
                max_query_len = None
            if self.pcp_manager is not None:
                num_toks = self.pcp_manager.get_num_tokens_for_dispatch(
                    batch_req_state.num_scheduled_tokens,
                    batch_req_state.is_prefilling_np,
                )

        return num_toks, max_query_len
