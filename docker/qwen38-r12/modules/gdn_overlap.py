# SPDX-License-Identifier: Apache-2.0
"""Bit-exact overlap of independent Qwen3.8 GDN input projections."""

from __future__ import annotations

import logging
import os

import torch
from einops import rearrange
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention as _BaseQwenGatedDeltaNetAttention,
    _encode_layer_name,
)


logger = logging.getLogger(__name__)


class QwenGatedDeltaNetAttention(_BaseQwenGatedDeltaNetAttention):
    """Run the original QKVZ and BA GEMMs on joined CUDA streams."""

    def forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        side = getattr(self, "_qwen38_gdn_projection_stream", None)
        max_tokens = int(os.getenv("QWEN38_GDN_OVERLAP_MAX_TOKENS", "3"))
        if side is None or hidden_states.size(0) > max_tokens:
            return super().forward_cuda(hidden_states)

        num_tokens = hidden_states.size(0)
        current = torch.cuda.current_stream(hidden_states.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            ba, _ = self.in_proj_ba(hidden_states)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        current.wait_stream(side)

        use_fused_gdn_decode = (
            self.enable_fused_gdn_decode
            and hidden_states.dtype == torch.bfloat16
            and self.norm.weight.dtype in (torch.bfloat16, torch.float32)
        )
        if use_fused_gdn_decode:
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.vllm.qwen_gdn_attention_core_fused_norm_packed(
                mixed_qkvz,
                ba,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
            )
            output, _ = self.out_proj(core_attn_out.flatten(-2))
            return output

        if self.gqa_interleaved_layout:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda value: rearrange(value, "l p d -> l (p d)"),
                (query, key, value),
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)

        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b.contiguous(),
            a.contiguous(),
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )
        return self._output_projection(core_attn_out, z)


@torch.no_grad()
def prepare_gdn_projection_overlap(root: nn.Module) -> int:
    """Attach one shared side stream to eligible target GDN modules."""

    if os.getenv("QWEN38_GDN_PROJECTION_OVERLAP", "1") != "1":
        logger.info("Qwen3.8 GDN projection overlap is disabled")
        return 0

    eligible: list[QwenGatedDeltaNetAttention] = []
    for child in root.modules():
        if not isinstance(child, QwenGatedDeltaNetAttention):
            continue
        if child.tp_size != 1:
            raise NotImplementedError("GDN projection overlap currently requires TP1")
        if not (
            type(child.in_proj_qkvz.quant_method) is UnquantizedLinearMethod
            and type(child.in_proj_ba.quant_method) is UnquantizedLinearMethod
            and child.in_proj_qkvz.weight.dtype == torch.bfloat16
            and child.in_proj_ba.weight.dtype == torch.bfloat16
        ):
            raise NotImplementedError("GDN projection overlap requires BF16 projections")
        eligible.append(child)

    if not eligible:
        raise RuntimeError("no eligible Qwen3.8 GDN modules were found")
    device = eligible[0].in_proj_qkvz.weight.device
    side = torch.cuda.Stream(device=device)
    for child in eligible:
        child._qwen38_gdn_projection_stream = side
    logger.info(
        "Enabled bit-exact Qwen3.8 GDN projection overlap for %d target modules",
        len(eligible),
    )
    return len(eligible)


__all__ = ["QwenGatedDeltaNetAttention", "prepare_gdn_projection_overlap"]
