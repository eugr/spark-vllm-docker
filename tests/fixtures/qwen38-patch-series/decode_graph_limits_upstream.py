# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpts from 790be385c8a69f45ec9a7a85ef00836d1684060f:
# vllm/config/vllm.py

class VllmConfig:
    def get_decode_cudagraph_query_lens(self, decode_query_len: int) -> dict[int, int]:
        """Map each uniform decode query length of a target-shaped runner, whose
        widest is `decode_query_len`, to the most requests its batches have.

        Dynamic speculative decoding drafts as many tokens as its schedule gives
        the batch size, so a query length only runs up to the largest batch size
        that drafts its number of tokens.
        """
        max_num_seqs = self.scheduler_config.max_num_seqs
        speculative_config = self.speculative_config
        num_new_sampled_tokens = decode_query_len - self.num_speculative_tokens
        if (
            speculative_config is None
            or not speculative_config.uses_dynamic_speculative_decoding()
            or num_new_sampled_tokens <= 0
        ):
            return {decode_query_len: max_num_seqs}
        from vllm.v1.spec_decode.dynamic.utils import build_dynamic_sd_schedule_lookup

        dense_schedule = build_dynamic_sd_schedule_lookup(
            speculative_config.num_speculative_tokens_per_batch_size,
            vllm_max_batch_size=max_num_seqs,
            vllm_num_speculative_tokens=self.num_speculative_tokens,
        )
        # By ascending batch size, so each query length keeps its widest batch.
        return {
            k + num_new_sampled_tokens: batch_size
            for batch_size, k in enumerate(dense_schedule)
            if batch_size > 0
        }


# vllm/config/compilation.py
def decode_cudagraph_capture_shapes(
    sizes: Iterable[int], query_lens: Mapping[int, int], max_tokens: int
) -> list[tuple[int, int]]:
    """The `(num_reqs, query_len)` uniform decode graphs for the request counts
    `sizes`, for each query length up to the most requests it maps to and
    `max_tokens` tokens. A batch pads to the smallest graph that holds it, so
    the first size above a limit is captured at the limit and larger ones are
    unreachable."""
    sizes = sorted(set(sizes))
    shapes: set[tuple[int, int]] = set()
    for q, max_reqs in query_lens.items():
        limit = min(max_reqs, max_tokens // q)
        reachable = sizes[: bisect.bisect_left(sizes, limit) + 1]
        shapes.update((min(n, limit), q) for n in reachable if limit >= 1)
    return sorted(shapes, key=lambda shape: (shape[1], shape[0]))
