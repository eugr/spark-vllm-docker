# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpt from vLLM 90309f6b742cdb1d6305778b81063918cf61be81.
# Keep the current watermark, sharded sampler, and greedy dispatch intact.
class DraftModelSpeculator:
    def sample_draft(
        self,
        hidden_states: torch.Tensor,
        sample_src_positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
        spec_step_idx: int = 0,
        num_steps: int = 1,
    ) -> torch.Tensor:
        if draft_logits is None and self.use_local_argmax_reduction:
            return self.get_draft_top_tokens(hidden_states, spec_step_idx)

        logits = self.compute_draft_logits(hidden_states, spec_step_idx)
        if draft_logits is not None and self.draft_watermarker is not None:
            sampled = self.draft_watermarker.sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                sample_src_positions,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
                num_steps=num_steps,
            )
        elif draft_logits is not None:
            sampled = gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                sample_src_positions,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=draft_logits,
                logits_cache_col=draft_step,
                use_fp64=self.use_fp64_gumbel,
                logits_cache_num_shards=self.draft_logits_num_shards,
                logits_cache_shard_idx=self.draft_logits_shard_idx,
            )
        else:
            sampled = logits.argmax(dim=-1)
        self._maybe_predict_acceptance(logits, idx_mapping, draft_step)
        return sampled
