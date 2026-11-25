# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState, dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
    SpeculatorCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


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

    def _run_multi_step_decode(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        dp_sync_state: DPSyncState | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
        is_profile: bool,
        num_speculative_steps: int,
    ) -> None:
        decode_batch_sync, num_batch_tokens = None, num_reqs
        if dp_sync_state is not None:
            # Padded dummy batches can exceed the requests the DP sync covers.
            num_reqs = min(num_reqs, dp_sync_state.num_reqs)
            decode_batch_sync, num_batch_tokens = self._build_uniform_batch_dp_sync(
                dp_sync_state, num_reqs
            )
        decode_batch_desc, decode_batch_sync = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=1,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync_state=decode_batch_sync,
            parallel_config=self.vllm_config.parallel_config,
        )
        if self.decode_cudagraph_manager is not None:
            decode_batch_desc = self.decode_cudagraph_manager.specialize_spec_tokens(
                decode_batch_desc, num_speculative_steps
            )

        self._prepare_decode_inputs(
            input_batch, num_rejected, num_reqs, skip_attn, decode_batch_desc
        )

        num_tokens_across_dp = (
            decode_batch_sync.num_tokens_across_dp
            if decode_batch_sync is not None
            else None
        )

        self.on_multi_step_decode_begin(num_reqs)
        multi_step_decode_fn = (
            self._run_fused_multi_step_decode
            if self.supports_decode_metadata_update
            else self._multi_step_decode
        )
        multi_step_decode_fn(
            num_reqs,
            skip_attn,
            decode_batch_desc,
            num_tokens_across_dp,
            seq_lens_cpu_upper_bound,
            num_speculative_steps,
        )
        self.on_multi_step_decode_end(num_reqs)

    def _multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_speculative_steps: int,
    ) -> None:
        positions = self.input_buffers.positions[:num_reqs]
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]
        idx_mapping = self.idx_mapping[:num_reqs]

        attn_metadata = None
        slot_mappings_by_layer = None
        for step in range(1, num_speculative_steps):
            if not skip_attn and (self.advance_draft_positions or step == 1):
                slot_mappings = self.block_tables.compute_slot_mappings(
                    idx_mapping,
                    query_start_loc,
                    positions,
                    batch_desc.num_tokens,
                )
                slot_mappings_by_layer = build_slot_mappings_by_layer(
                    slot_mappings, self.kv_cache_config
                )
                attn_metadata = self._build_uniform_attn_metadata(
                    num_reqs=num_reqs,
                    batch_desc=batch_desc,
                    num_query_per_req=1,
                    seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                    step=step,
                )

            self.current_draft_step.fill_(step)
            if batch_desc.cg_mode == CUDAGraphMode.FULL:
                assert self.decode_cudagraph_manager is not None
                self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            else:
                self._decode(
                    num_reqs,
                    batch_desc.num_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp,
                    batch_desc.cg_mode,
                    num_speculative_steps,
                )

    def _run_fused_multi_step_decode(
        self,
        num_reqs: int,
        skip_attn: bool,
        batch_desc: BatchExecutionDescriptor,
        num_tokens_across_dp: torch.Tensor | None,
        seq_lens_cpu_upper_bound: torch.Tensor,
        num_speculative_steps: int,
    ) -> None:
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            self._map_kernel_block_tables()
            assert self.decode_cudagraph_manager is not None
            self.decode_cudagraph_manager.run_fullgraph(batch_desc)
            return

        attn_metadata = None
        slot_mappings_by_layer = None
        if not skip_attn:
            slot_mappings = self.block_tables.compute_slot_mappings(
                self.idx_mapping[:num_reqs],
                self.input_buffers.query_start_loc[: num_reqs + 1],
                self.input_buffers.positions[:num_reqs],
                batch_desc.num_tokens,
            )
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings, self.kv_cache_config
            )
            attn_metadata = self._build_uniform_attn_metadata(
                batch_desc=batch_desc,
                num_reqs=num_reqs,
                num_query_per_req=1,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=1,
            )

        self._fused_multi_step_decode(
            num_reqs,
            batch_desc.num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp,
            batch_desc.cg_mode,
            num_speculative_steps,
        )

