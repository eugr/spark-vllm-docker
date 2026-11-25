# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpt from vllm/v1/core/sched/scheduler.py
# at vllm-project/vllm@ad457ad7440bf0da7a7a8edd67ba9e66a9dec590.

class Scheduler:
    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """
        # Keep all output tokens but the final sampled one.
        keep_end = session.num_tokens - 1
        assert keep_end >= session.num_computed_tokens
        del session._all_token_ids[keep_end:]
        del session.block_hashes[keep_end // self.hash_block_size :]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        extend_from_slice(
            session.prompt_token_ids,
            session._all_token_ids,
            session.num_prompt_tokens,
            keep_end,
        )

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        session.prefix_cache_hit_limit = session.get_prefix_cache_hit_limit()
        session.max_tokens = update.max_tokens
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)
