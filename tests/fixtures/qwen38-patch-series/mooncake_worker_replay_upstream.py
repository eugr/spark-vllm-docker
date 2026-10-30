# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from dc4dd5a2217bc757232e9fa4d8a8b59e207ca068:
# vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py

class KVCacheStoreSendingThread:
    def mask(self, req_meta, checkpoint_len, save_start):
        if True:
            replay_boundary = req_meta.prefill_end_tokens
            if (
                self.kv_role == "kv_consumer"
                and replay_boundary is not None
                and checkpoint_len > replay_boundary
            ):
                replay_boundary = checkpoint_len + 1
            # Successful saves advance by complete scheduler-aligned intervals.
            store_masks = self.coord.store_mask(
                checkpoint_len,
                save_start,
                num_prompt_tokens=replay_boundary,
            )

            starts: list[int] = []
            ends: list[int] = []
            return store_masks
