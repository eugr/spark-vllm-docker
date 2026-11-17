# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental DSpark draft model for DeepSeek V4 Flash.

This module follows the reference implementation shipped with
DeepSeek-V4-Flash-DSpark. It intentionally keeps the draft-side DSpark
attention cache internal to the draft model instead of registering more vLLM
KV-cache layers; DSpark uses a small sliding window over target features and
draft block tokens, which is different from the normal MTP cache contract.
"""

from __future__ import annotations

import os
import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import HCHeadOp
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.models.deepseek_v4.common.ops import fused_inv_rope_fp8_quant
from vllm.platforms import current_platform
from vllm.v1.spec_decode.dspark import (
    map_dspark_stacked_param_name,
    unpack_mhc_pre_outputs,
)

from .dspark_kernels import (
    dspark_markov_argmax,
    dspark_quant_dequant_nope,
    dspark_sparse_attention,
)
from .model import (
    DeepseekV4MoE,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)


# Kill switch for A/B diagnosis: DSPARK_SLOT_CLAMP=0 reverts the protective
# clamp (detect+log only, return the index unchanged) so operators can verify
# the clamp is load-bearing on their rig. Default on = the fix. Read once at
# import (fixed per container start).
_SLOT_CLAMP_ENABLED = os.environ.get("DSPARK_SLOT_CLAMP", "1") != "0"


def _guard_slot_index(
    slot_index: torch.Tensor | None, num_rows: int, site: str
) -> torch.Tensor | None:
    """Bounds-check DSpark ring-buffer slot ids before a KV gather.

    A stale/out-of-range ``slot_index`` (observed after request condensation at
    long context) gathering row >= ``num_rows`` triggers a device-side
    ``indexSelectSmallIndex`` assert (``srcIndex < srcSelectDimSize``) that kills
    the worker. Clamp into range on-device so the gather degrades to a rejected
    speculation instead of a crash: this is the draft path only, so a clamped
    (wrong) slot yields a bad draft token that the target model simply rejects
    at verification — it cannot corrupt output.

    The clamp is graph-safe (pure device op, baked into the captured graph and
    active at replay). The loud host-side value log requires a device sync,
    which is illegal mid CUDA-graph-capture, so it only runs on eager paths
    (``is_current_stream_capturing()`` is a driver query, no sync).
    """
    if slot_index is None or slot_index.numel() == 0:
        return slot_index
    clamped = slot_index.clamp(0, num_rows - 1)
    if not torch.cuda.is_current_stream_capturing():
        hi = int(slot_index.max())
        lo = int(slot_index.min())
        if hi >= num_rows or lo < 0:
            logger.error(
                "DSPARK_STALE_SLOT_INDEX site=%s num_rows=%d min=%d max=%d "
                "slot_index=%s",
                site,
                num_rows,
                lo,
                hi,
                slot_index.tolist(),
            )
    return clamped if _SLOT_CLAMP_ENABLED else slot_index


_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


def _read_bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _linear_no_bias(
    linear: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    out = linear(x)
    if isinstance(out, tuple):
        y, bias = out
        assert bias is None
        return y
    return out


def _vocab_parallel_argmax(
    local_logits: torch.Tensor,
    lm_head: VocabParallelEmbedding,
) -> torch.Tensor:
    """Return global greedy token ids from local vocab-parallel logits."""
    num_pad = lm_head.shard_indices.num_org_vocab_padding
    if num_pad > 0:
        local_logits[..., -num_pad:] = -float("inf")

    local_max_vals, local_max_indices = local_logits.max(dim=-1)
    global_indices = local_max_indices + lm_head.shard_indices.org_vocab_start_index
    return _vocab_parallel_argmax_from_local(local_max_vals, global_indices)


def _vocab_parallel_argmax_from_local(
    local_max_vals: torch.Tensor,
    global_indices: torch.Tensor,
) -> torch.Tensor:
    """Return global token ids from per-rank local top-1 candidates."""

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return global_indices.to(torch.long)

    local_pair = torch.stack(
        [local_max_vals.float(), global_indices.float()],
        dim=-1,
    )
    gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
    gathered = gathered.view(local_max_vals.shape[0], tp_size, 2)
    max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
    return top_tokens.squeeze(-1).to(torch.long)


def _vocab_parallel_markov_argmax(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2: ParallelLMHead,
    lm_head: VocabParallelEmbedding,
) -> torch.Tensor:
    """Return global greedy ids for base logits plus DSpark Markov bias."""

    num_pad = lm_head.shard_indices.num_org_vocab_padding
    local_max_vals, local_max_indices = dspark_markov_argmax(
        base_logits,
        markov_embed,
        markov_w2.weight,
        num_pad=num_pad,
    )
    global_indices = local_max_indices + lm_head.shard_indices.org_vocab_start_index
    return _vocab_parallel_argmax_from_local(local_max_vals, global_indices)


class DeepSeekV4DSparkAttention(nn.Module):
    """DSpark sparse MLA attention with an internal main-token KV window."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        self.dtype = vllm_config.model_config.dtype
        self.n_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        self.block_size = config.dspark_block_size
        self.eps = config.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5
        self._reference_kv_quant_dequant = _read_bool_env(
            "VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT"
        )
        cap = current_platform.get_device_capability()
        assert cap is not None, "DSpark attention requires a CUDA device"
        self._einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
        self._tma_aligned_scales = cap.major >= 10

        self.attn_sink = nn.Parameter(
            torch.full((self.n_local_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        rope_parameters = config.rope_parameters
        rope_parameters["rope_theta"] = config.rope_theta
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        rope_parameters["mscale"] = 0
        rope_parameters["mscale_all_dim"] = 0
        rope_parameters["is_deepseek_v4"] = True
        rope_parameters["rope_dim"] = self.rope_head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.register_buffer(
            "main_kv_cache",
            torch.zeros(
                max_batch_size,
                self.window_size,
                self.head_dim,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        self.register_buffer(
            "sparse_scores",
            torch.empty(
                max_batch_size,
                self.block_size,
                self.n_local_heads,
                self.window_size + self.block_size,
                dtype=torch.float32,
                device=current_platform.device_type,
            ),
            persistent=False,
        )
        if self._reference_kv_quant_dequant:
            logger.info(
                "DSpark reference KV FP8 quant-dequant enabled for no-RoPE "
                "dimensions in %s.",
                prefix,
            )

    def _maybe_quant_dequant_kv(self, kv: torch.Tensor) -> torch.Tensor:
        if self._reference_kv_quant_dequant:
            return dspark_quant_dequant_nope(
                kv,
                rope_dim=self.rope_head_dim,
                group_size=64,
            )
        return kv

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        qra_kv = _linear_no_bias(self.fused_wqa_wkv, hidden_states)
        _, kv = qra_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        kv = self.kv_norm(kv)
        kv, _ = self.rotary_emb(positions, kv.unsqueeze(1), None)
        kv = kv.squeeze(1)
        return self._maybe_quant_dequant_kv(kv)

    def _store_main_kv_ragged(
        self,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
        query_start_loc: list[int],
        num_rejected_tokens: torch.Tensor | None,
        slot_index: list[int] | torch.Tensor | None,
    ) -> None:
        """Scatter ragged (mixed prefill+decode) per-request rows.

        ``main_x`` is flat [total_rows, hidden_size] and ``main_positions`` is
        flat [total_rows]; ``query_start_loc`` gives per-request offsets. Each
        request's rows are scattered into ITS slot's ring buffer using absolute
        positions (``positions % window``), so no rectangular view / uniform
        length is required. Runs eager (mixed steps are never cudagraphed).
        """
        flat_positions = main_positions.reshape(-1)
        flat_kv = self._project_kv(main_x, flat_positions)
        batch_size = len(query_start_loc) - 1
        rejected = None
        if num_rejected_tokens is not None:
            rejected = num_rejected_tokens.to(
                device=main_x.device,
                dtype=torch.long,
                non_blocking=True,
            ).view(batch_size)
        for i in range(batch_size):
            start = int(query_start_loc[i])
            end = int(query_start_loc[i + 1])
            seg_kv = flat_kv[start:end]
            seg_positions = flat_positions[start:end].to(torch.long)
            if seg_kv.shape[0] > self.window_size:
                seg_kv = seg_kv[-self.window_size :]
                seg_positions = seg_positions[-self.window_size :]
            seg_len = seg_kv.shape[0]
            if seg_len == 0:
                continue
            slots = seg_positions.remainder(self.window_size)
            row = i if slot_index is None else int(slot_index[i])
            num_rows = self.main_kv_cache.shape[0]
            if slot_index is not None and not 0 <= row < num_rows:
                logger.error(
                    "DSPARK_STALE_SLOT_INDEX site=ragged row=%d num_rows=%d i=%d",
                    row,
                    num_rows,
                    i,
                )
                if _SLOT_CLAMP_ENABLED:
                    row = min(max(row, 0), num_rows - 1)
            cache_row = self.main_kv_cache[row]
            values = seg_kv
            if rejected is not None:
                valid_len = (seg_len - rejected[i]).clamp(min=1, max=seg_len)
                token_offsets = torch.arange(
                    seg_len,
                    device=seg_kv.device,
                    dtype=torch.long,
                )
                valid_mask = (token_offsets < valid_len).unsqueeze(-1)
                old_values = cache_row.index_select(0, slots)
                values = torch.where(valid_mask, seg_kv, old_values)
            cache_row.index_copy_(0, slots, values)

    def store_main_kv(
        self,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
        num_rejected_tokens: torch.Tensor | None = None,
        slot_index: torch.Tensor | None = None,
        query_start_loc: list[int] | None = None,
    ) -> None:
        if query_start_loc is not None:
            self._store_main_kv_ragged(
                main_x,
                main_positions,
                query_start_loc,
                num_rejected_tokens,
                slot_index,
            )
            return
        if main_x.shape[1] > self.window_size:
            main_x = main_x[:, -self.window_size :]
            main_positions = main_positions[:, -self.window_size :]
        batch_size, seq_len, _ = main_x.shape
        flat_kv = self._project_kv(
            main_x.reshape(batch_size * seq_len, self.hidden_size),
            main_positions.reshape(batch_size * seq_len),
        ).view(batch_size, seq_len, self.head_dim)
        slots = main_positions.to(torch.long).remainder(self.window_size)
        slots_index = slots.unsqueeze(-1).expand(-1, -1, self.head_dim)
        values = flat_kv
        # ``slot_index is None`` keeps the original single-stream behaviour:
        # operate in place on the leading ``batch_size`` rows (byte-for-byte
        # identical). Otherwise gather the per-request slot rows, update them,
        # and scatter them back, so the stateful sliding-window KV follows the
        # request id rather than the (condensable) batch-row position.
        if slot_index is None:
            cache_rows = self.main_kv_cache[:batch_size]
        else:
            slot_index = _guard_slot_index(
                slot_index, self.main_kv_cache.shape[0], "store_main_kv"
            )
            cache_rows = self.main_kv_cache.index_select(0, slot_index)
        if num_rejected_tokens is not None:
            rejected = num_rejected_tokens.to(
                device=main_x.device,
                dtype=torch.long,
                non_blocking=True,
            ).view(batch_size)
            valid_lengths = (seq_len - rejected).clamp(min=1, max=seq_len)
            token_offsets = torch.arange(
                seq_len,
                device=main_x.device,
                dtype=torch.long,
            ).view(1, seq_len)
            valid_mask = token_offsets < valid_lengths.view(batch_size, 1)
            old_values = cache_rows.gather(1, slots_index)
            values = torch.where(valid_mask.unsqueeze(-1), flat_kv, old_values)
        cache_rows.scatter_(1, slots_index, values)
        if slot_index is not None:
            self.main_kv_cache.index_copy_(0, slot_index, cache_rows)

    def _project_q_and_draft_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qra_kv = _linear_no_bias(self.fused_wqa_wkv, hidden_states)
        qra, kv = qra_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qra = self.q_norm(qra)
        q = _linear_no_bias(self.wq_b, qra).view(-1, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        kv = self.kv_norm(kv)
        q, _ = self.rotary_emb(positions, q, None)
        kv, _ = self.rotary_emb(positions, kv.unsqueeze(1), None)
        kv = kv.squeeze(1)
        return q, self._maybe_quant_dequant_kv(kv)

    def forward_dspark(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        batch_size: int,
        block_size: int,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
        store_main_kv: bool = True,
        slot_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if store_main_kv:
            self.store_main_kv(main_x, main_positions, slot_index=slot_index)

        q, draft_kv = self._project_q_and_draft_kv(hidden_states, positions)
        q = q.view(batch_size, block_size, self.n_local_heads, self.head_dim)
        draft_kv = draft_kv.view(batch_size, block_size, self.head_dim)
        current_positions = main_positions[:, -1]
        valid_main_lengths = torch.minimum(
            current_positions + 1,
            torch.full_like(current_positions, self.window_size),
        )

        # The kernel indexes contiguous batch rows 0..B-1. ``slot_index is
        # None`` passes the full persistent cache (rows 0..B-1 == identity);
        # otherwise gather the per-request slot rows into a contiguous
        # [B, window, head_dim] view in the same order as ``valid_main_lengths``.
        if slot_index is None:
            main_kv_cache = self.main_kv_cache
        else:
            slot_index = _guard_slot_index(
                slot_index, self.main_kv_cache.shape[0], "forward_dspark"
            )
            main_kv_cache = self.main_kv_cache.index_select(0, slot_index)

        out = dspark_sparse_attention(
            q,
            draft_kv,
            main_kv_cache,
            valid_main_lengths,
            self.attn_sink,
            self.softmax_scale,
            self.sparse_scores[:batch_size, :block_size],
        ).to(self.dtype)
        out_fp8, out_scale = fused_inv_rope_fp8_quant(
            out,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=self._tma_aligned_scales,
        )
        projected = torch.empty(
            (batch_size * block_size, self.n_local_groups, self.o_lora_rank),
            dtype=self.dtype,
            device=out.device,
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(
            out_fp8,
            out_scale,
            self.wo_a.weight,
            self.wo_a.weight_scale_inv,
            projected,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )
        return _linear_no_bias(self.wo_b, projected.flatten(1).to(self.dtype))


class DeepSeekV4DSparkMarkovHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, *, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self._replicated_w1 = _read_bool_env("VLLM_DSPARK_REPLICATE_MARKOV_W1")
        if self._replicated_w1:
            self.markov_w1 = nn.Embedding(
                config.vocab_size,
                config.dspark_markov_rank,
                dtype=vllm_config.model_config.dtype,
            )
            self.markov_w1.weight.requires_grad_(False)
            logger.info(
                "DSpark replicated Markov W1 enabled for %s. This removes "
                "the per-position vocab-parallel embedding all-reduce.",
                prefix,
            )
        else:
            self.markov_w1 = VocabParallelEmbedding(
                config.vocab_size,
                config.dspark_markov_rank,
                prefix=f"{prefix}.markov_w1",
            )
        self.markov_w2 = ParallelLMHead(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=f"{prefix}.markov_w2",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        markov_embed = self.markov_w1(token_ids)
        markov_logits = self.logits_processor(self.markov_w2, markov_embed)
        return markov_logits, markov_embed

    def forward_local(
        self,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        markov_embed = self.markov_w1(token_ids)
        markov_logits = self.markov_w2.quant_method.apply(
            self.markov_w2,
            markov_embed,
            bias=None,
        )
        return markov_logits, markov_embed


class DeepSeekV4DSparkConfidenceHead(nn.Module):
    def __init__(self, vllm_config: VllmConfig, *, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.proj = ReplicatedLinear(
            config.hidden_size + config.dspark_markov_rank,
            1,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embed: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([hidden_states, markov_embed], dim=-1)
        return _linear_no_bias(self.proj, features.float()).squeeze(-1)


class DeepSeekV4DSparkLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        stage_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.stage_id = stage_id
        self.hidden_size = config.hidden_size
        self.dtype = vllm_config.model_config.dtype
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * self.hidden_size
        self.block_size = config.dspark_block_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_head_op = HCHeadOp()

        if stage_id == 0:
            self.main_proj = ReplicatedLinear(
                config.hidden_size * len(config.dspark_target_layer_ids),
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                return_bias=False,
                prefix=f"{prefix}.main_proj",
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attn = DeepSeekV4DSparkAttention(
            vllm_config,
            prefix=f"{prefix}.attn",
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        # Reuse the target decoder's MHC kernels/parameters.
        from vllm.model_executor.layers.mhc import MHCPostOp, MHCPreOp

        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        self.hc_attn_fn = nn.Parameter(
            torch.empty((mix_hc, self.hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty((mix_hc, self.hc_dim), dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32),
            requires_grad=False,
        )
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()

        if stage_id == config.dspark_num_draft_layers - 1:
            self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.markov_head = DeepSeekV4DSparkMarkovHead(
                vllm_config, prefix=f"{prefix}.markov_head"
            )
            self.confidence_head = DeepSeekV4DSparkConfidenceHead(
                vllm_config, prefix=f"{prefix}.confidence_head"
            )
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_head_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32),
                requires_grad=False,
            )

    def project_main(self, main_hidden: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "main_proj")
        main_x = _linear_no_bias(self.main_proj, main_hidden)
        return self.main_norm(main_x)

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
        )

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        return self.mhc_post(x, residual, post, comb)

    def forward_dspark(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        batch_size: int,
        block_size: int,
        main_x: torch.Tensor,
        main_positions: torch.Tensor,
        store_main_kv: bool = True,
        slot_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        attn_in, post, comb = unpack_mhc_pre_outputs(
            self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        )
        attn_in = self.attn_norm(attn_in).to(self.dtype)
        if attn_in.ndim == 3:
            attn_in = attn_in.mean(dim=1).to(self.dtype)
        attn_out = self.attn.forward_dspark(
            attn_in,
            positions,
            batch_size=batch_size,
            block_size=block_size,
            main_x=main_x,
            main_positions=main_positions,
            store_main_kv=store_main_kv,
            slot_index=slot_index,
        )
        x = self.hc_post(attn_out.to(self.dtype), residual, post, comb).to(self.dtype)

        residual = x
        ffn_in, post, comb = unpack_mhc_pre_outputs(
            self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        )
        ffn_in = self.ffn_norm(ffn_in).to(self.dtype)
        ffn_out = self.ffn(ffn_in, input_ids)
        return self.hc_post(ffn_out.to(self.dtype), residual, post, comb).to(self.dtype)

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.hc_head_op(
            x,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )


class DeepSeekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.num_draft_layers = config.dspark_num_draft_layers
        self._local_argmax = _read_bool_env("VLLM_DSPARK_LOCAL_ARGMAX")
        self._fused_markov_argmax = _read_bool_env(
            "VLLM_DSPARK_FUSED_MARKOV_ARGMAX"
        )
        self.dspark_start_layer_idx = max(
            config.num_hidden_layers,
            max(getattr(config, "dspark_target_layer_ids", [-1])) + 1,
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.stage_layer_keys = [
            str(self.dspark_start_layer_idx + stage_id)
            for stage_id in range(self.num_draft_layers)
        ]
        self.layers = nn.ModuleDict(
            {
                layer_key: DeepSeekV4DSparkLayer(
                    vllm_config,
                    stage_id=stage_id,
                    prefix=maybe_prefix(prefix, f"layers.{layer_key}"),
                )
                for stage_id, layer_key in enumerate(self.stage_layer_keys)
            }
        )
        if self._local_argmax:
            logger.info(
                "DSpark local vocab-parallel argmax is enabled. This is "
                "experimental and may add per-position synchronization overhead."
            )
        if self._fused_markov_argmax:
            logger.info(
                "DSpark fused Markov argmax is enabled. This keeps the paper's "
                "low-rank Markov bias but avoids materializing Markov logits "
                "on the greedy no-confidence draft path."
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_main(self, main_hidden: torch.Tensor) -> torch.Tensor:
        first = self.layers[self.stage_layer_keys[0]]
        return first.project_main(main_hidden)

    def prefill_main(
        self,
        main_hidden: torch.Tensor,
        main_positions: torch.Tensor,
        num_rejected_tokens: torch.Tensor | None = None,
        slot_index: torch.Tensor | None = None,
        query_start_loc: list[int] | None = None,
    ) -> None:
        main_x = self.project_main(main_hidden.reshape(-1, main_hidden.shape[-1]))
        if query_start_loc is None:
            # Rectangular [B, seq, hidden_size] fast-path (uniform / static).
            main_x = main_x.view(*main_hidden.shape[:-1], self.config.hidden_size)
        # else: keep flat [total_rows, hidden_size] for the ragged path.
        for layer in self.layers.values():
            layer.attn.store_main_kv(
                main_x,
                main_positions,
                num_rejected_tokens=num_rejected_tokens,
                slot_index=slot_index,
                query_start_loc=query_start_loc,
            )

    def draft(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        main_positions: torch.Tensor,
        lm_head: ParallelLMHead,
        logits_processor: LogitsProcessor,
        *,
        return_logits: bool = True,
        return_confidence: bool = True,
        store_main_kv: bool = True,
        slot_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input_ids.shape[0]
        block_size = self.block_size
        main_positions = main_positions.view(batch_size, 1)
        main_x = self.project_main(main_hidden).view(
            batch_size, 1, self.config.hidden_size
        )

        draft_input_ids = input_ids.new_full(
            (batch_size, block_size), self.noise_token_id
        )
        draft_input_ids[:, 0] = input_ids
        x = self.embed_tokens(draft_input_ids).view(
            batch_size * block_size, self.config.hidden_size
        )
        x = x.unsqueeze(1).repeat(1, self.config.hc_mult, 1)

        # DeepSpec trains/evaluates DSpark with the anchor token itself at the
        # first draft position: anchor + [0, gamma). The hidden state at that
        # anchor position predicts the next token.
        offsets = torch.arange(
            0,
            block_size,
            dtype=main_positions.dtype,
            device=main_positions.device,
        )
        draft_positions = (main_positions[:, -1:] + offsets).reshape(-1)
        flat_draft_input_ids = draft_input_ids.reshape(-1)
        for layer in self.layers.values():
            x = layer.forward_dspark(
                x,
                draft_positions,
                flat_draft_input_ids,
                batch_size=batch_size,
                block_size=block_size,
                main_x=main_x,
                main_positions=main_positions,
                store_main_kv=store_main_kv,
                slot_index=slot_index,
            )

        final_layer = self.layers[self.stage_layer_keys[-1]]
        dense = final_layer.forward_head(x).view(
            batch_size, block_size, self.config.hidden_size
        )
        normed = final_layer.norm(dense.reshape(batch_size * block_size, -1))

        if not return_logits and getattr(self, "_local_argmax", False):
            local_logits = lm_head.quant_method.apply(
                lm_head,
                normed,
                bias=None,
            ).view(batch_size, block_size, -1)
            output_ids = input_ids.new_empty(batch_size, block_size + 1)
            output_ids[:, 0] = input_ids
            markov_embeds = [] if return_confidence else None
            for pos in range(block_size):
                if markov_embeds is not None:
                    markov_logits, markov_embed = (
                        final_layer.markov_head.forward_local(output_ids[:, pos])
                    )
                    markov_embeds.append(markov_embed)
                    step_logits = local_logits[:, pos] + markov_logits
                    output_ids[:, pos + 1] = _vocab_parallel_argmax(
                        step_logits,
                        lm_head,
                    )
                elif getattr(self, "_fused_markov_argmax", False):
                    markov_embed = final_layer.markov_head.markov_w1(output_ids[:, pos])
                    output_ids[:, pos + 1] = _vocab_parallel_markov_argmax(
                        local_logits[:, pos],
                        markov_embed,
                        final_layer.markov_head.markov_w2,
                        lm_head,
                    )
                else:
                    markov_logits, _ = final_layer.markov_head.forward_local(
                        output_ids[:, pos]
                    )
                    step_logits = local_logits[:, pos] + markov_logits
                    output_ids[:, pos + 1] = _vocab_parallel_argmax(
                        step_logits,
                        lm_head,
                    )
            logits = normed.new_empty((0, 0, 0))
            if return_confidence:
                assert markov_embeds is not None
                markov_embed = torch.stack(markov_embeds, dim=1)
                confidence = final_layer.confidence_head(dense, markov_embed).sigmoid()
            else:
                confidence = dense.new_empty((batch_size, 0), dtype=torch.float32)
            return output_ids[:, 1:], logits, confidence

        logits = logits_processor(lm_head, normed).view(
            batch_size, block_size, self.config.vocab_size
        )

        output_ids = input_ids.new_empty(batch_size, block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = [] if return_confidence else None
        for pos in range(block_size):
            markov_logits, markov_embed = final_layer.markov_head(output_ids[:, pos])
            logits[:, pos].add_(markov_logits)
            if markov_embeds is not None:
                markov_embeds.append(markov_embed)
            output_ids[:, pos + 1] = logits[:, pos].argmax(dim=-1)

        if return_confidence:
            assert markov_embeds is not None
            markov_embed = torch.stack(markov_embeds, dim=1)
            confidence = final_layer.confidence_head(dense, markov_embed).sigmoid()
        else:
            confidence = dense.new_empty((batch_size, 0), dtype=torch.float32)
        if not return_logits:
            logits = logits.new_empty((0, 0, 0))
        return output_ids[:, 1:], logits, confidence

    def finalize_mega_moe_weights(self) -> None:
        for layer in self.layers.values():
            if layer.ffn.use_mega_moe:
                layer.ffn.finalize_mega_moe_weights()


class DeepSeekV4DSpark(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.model = DeepSeekV4DSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self._last_confidence: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def prefill_main(
        self,
        main_hidden: torch.Tensor,
        main_positions: torch.Tensor,
        num_rejected_tokens: torch.Tensor | None = None,
        slot_index: torch.Tensor | None = None,
        query_start_loc: list[int] | None = None,
    ) -> None:
        self.model.prefill_main(
            main_hidden,
            main_positions,
            num_rejected_tokens=num_rejected_tokens,
            slot_index=slot_index,
            query_start_loc=query_start_loc,
        )

    def draft(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        main_positions: torch.Tensor,
        *,
        store_main_kv: bool = True,
        slot_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        draft_ids, _logits, confidence = self.model.draft(
            input_ids,
            main_hidden,
            main_positions,
            self.lm_head,
            self.logits_processor,
            store_main_kv=store_main_kv,
            slot_index=slot_index,
        )
        self._last_confidence = confidence
        return draft_ids

    def draft_with_confidence(
        self,
        input_ids: torch.Tensor,
        main_hidden: torch.Tensor,
        main_positions: torch.Tensor,
        *,
        return_logits: bool = True,
        return_confidence: bool = True,
        store_main_kv: bool = True,
        slot_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        draft_ids, logits, confidence = self.model.draft(
            input_ids,
            main_hidden,
            main_positions,
            self.lm_head,
            self.logits_processor,
            return_logits=return_logits,
            return_confidence=return_confidence,
            store_main_kv=store_main_kv,
            slot_index=slot_index,
        )
        return draft_ids, logits, confidence

    def take_last_confidence(self) -> torch.Tensor | None:
        confidence = self._last_confidence
        self._last_confidence = None
        return confidence

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        first_layer = next(iter(self.model.layers.values()))
        if first_layer.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = FusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )
        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            if not name.startswith("mtp."):
                continue
            stage_id = int(name.split(".", 2)[1])
            if stage_id >= self.config.dspark_num_draft_layers:
                continue
            virtual_layer_id = self.model.dspark_start_layer_idx + stage_id
            name = name.replace(
                f"mtp.{stage_id}.", f"model.layers.{virtual_layer_id}.", 1
            )
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")

            if ".attn.attn_sink" in name:
                param = params_dict[name]
                param.data.copy_(loaded_weight[head_rank_start:head_rank_end])
                loaded_params.add(name)
                continue
            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            mapped_stacked = map_dspark_stacked_param_name(name)
            if mapped_stacked is not None:
                mapped_name, shard_id = mapped_stacked
                param = params_dict.get(mapped_name)
                if param is None:
                    raise KeyError(mapped_name)
                else:
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(mapped_name)
                continue

            if ".experts." in name:
                if (
                    "weight_scale" in name
                    and loaded_weight.dtype == torch.float8_e8m0fnu
                ):
                    loaded_weight = loaded_weight.view(torch.uint8)
                for mapping in expert_mapping:
                    param_name, weight_name, expert_id, expert_shard_id = mapping
                    if weight_name not in name:
                        continue
                    mapped_name = name.replace(weight_name, param_name)
                    param = params_dict[mapped_name]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        mapped_name,
                        shard_id=expert_shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        loaded_params.add(mapped_name)
                        break
                continue

            param = params_dict.get(name)
            if param is None:
                logger.debug("Skipping unknown DSpark weight %s", name)
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        self.model.finalize_mega_moe_weights()
        return loaded_params
