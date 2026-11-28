"""Runtime Humming quantization & kernel helpers for Inkling 2 Mini on SM12x."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.linear import LinearMethodBase


_SHARED_LOCKS: dict[torch.device, torch.Tensor] = {}


def _get_locks(device: torch.device) -> torch.Tensor:
    locks = _SHARED_LOCKS.get(device)
    if locks is None:
        locks = torch.zeros(2048, dtype=torch.int32, device=device)
        _SHARED_LOCKS[device] = locks
    return locks


class HummingDenseOp:
    """Online-quantized BF16->(MXFP8|NVFP4) dense W{4,8}A16 Humming linear op."""

    def __init__(
        self,
        weight_bf16: torch.Tensor,
        mode: str,
        bias: torch.Tensor | None = None,
        *,
        use_stream_k: bool = True,
        pad_n_to: int = 256,
        pad_k_to: int = 128,
        slice_n: bool = True,
    ) -> None:
        from vllm.utils.humming import (
            GemmType,
            HummingInputSchema,
            HummingWeightSchema,
            WeightScale2Type,
            WeightScaleType,
            dtypes,
            get_heuristics_config,
            prepare_and_transform_humming_tensors,
            quantize_weight,
        )

        if mode not in ("mxfp8", "nvfp4"):
            raise ValueError(f"Unsupported Humming quantization mode: {mode!r}")
        if weight_bf16.ndim != 2:
            raise ValueError(f"Expected 2D weight tensor, got shape {weight_bf16.shape}")

        orig_n, orig_k = weight_bf16.shape
        self.orig_n = orig_n
        self.orig_k = orig_k
        self.mode = mode
        device = weight_bf16.device

        pad_n = ((orig_n + pad_n_to - 1) // pad_n_to) * pad_n_to
        pad_k = ((orig_k + pad_k_to - 1) // pad_k_to) * pad_k_to
        w = weight_bf16.detach()
        if pad_n != orig_n or pad_k != orig_k:
            w_pad = torch.zeros((pad_n, pad_k), dtype=w.dtype, device=device)
            w_pad[:orig_n, :orig_k].copy_(w)
            w = w_pad
        else:
            w = w.contiguous()

        if mode == "mxfp8":
            qw, ws, _, _ = quantize_weight(
                w,
                dtype=dtypes.float8e4m3,
                scale_dtype=dtypes.float8e8m0,
                group_size=32,
                pack=True,
            )
            schema = HummingWeightSchema(
                b_dtype=dtypes.float8e4m3,
                bs_dtype=dtypes.float8e8m0,
                weight_scale_group_size=32,
            )
            raw_tensors: dict[str, torch.Tensor] = {
                "weight": qw,
                "weight_scale": ws,
            }
        else:
            qw, ws, _, ws2 = quantize_weight(
                w,
                dtype=dtypes.float4e2m1,
                scale_dtype=dtypes.float8e4m3,
                group_size=16,
                weight_scale_2_type="channel",
                pack=True,
            )
            schema = HummingWeightSchema(
                b_dtype=dtypes.float4e2m1,
                bs_dtype=dtypes.float8e4m3,
                weight_scale_group_size=16,
                weight_scale_type=WeightScaleType.GROUP,
                weight_scale_2_type=WeightScale2Type.CHANNEL,
            )
            assert ws2 is not None
            raw_tensors = {
                "weight": qw,
                "weight_scale": ws,
                "weight_scale_2": ws2.to(torch.bfloat16),
            }

        if bias is not None:
            b = bias.detach().to(torch.bfloat16)
            if pad_n != orig_n:
                b_pad = torch.zeros((pad_n,), dtype=torch.bfloat16, device=device)
                b_pad[:orig_n].copy_(b)
                b = b_pad
            raw_tensors["bias"] = b

        cfg, tensors = prepare_and_transform_humming_tensors(
            tensors=raw_tensors,
            shape_n=pad_n,
            shape_k=pad_k,
            pad_n_to_multiple=pad_n_to,
            pad_k_to_multiple=pad_k_to,
            input_schema=HummingInputSchema(),
            weight_schema=schema,
            torch_dtype=torch.bfloat16,
            device=device,
        )
        tuning = get_heuristics_config(
            layer_config=cfg,
            use_f16_accum=False,
            use_batch_invariant=False,
            gemm_type=GemmType.DENSE,
        )
        if not use_stream_k:
            for _, _, kc in tuning:
                kc["use_stream_k"] = False
        self.config = cfg
        self.tensors = tensors
        self.tuning_str = json.dumps(tuning)
        self.compute_config = json.dumps(
            {
                "use_batch_invariant": False,
                "use_f16_accum": False,
                "gemm_type": GemmType.DENSE.value,
            }
        )
        self.locks = _get_locks(device)
        self.pad_n = pad_n
        self.pad_k = pad_k
        self.slice_n = slice_n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        if self.pad_k != self.orig_k:
            x_pad = torch.zeros(
                (x2.shape[0], self.pad_k), dtype=torch.bfloat16, device=x2.device
            )
            x_pad[:, : self.orig_k].copy_(x2)
            x2 = x_pad
        out = humming_forward(
            self.config,
            inputs=x2,
            weight=self.tensors["weight"],
            weight_scale=self.tensors.get("weight_scale"),
            weight_scale_2=self.tensors.get("weight_scale_2"),
            bias=self.tensors.get("bias"),
            locks=self.locks,
            compute_config=self.compute_config,
            tuning_config=self.tuning_str,
        )
        if self.slice_n and self.pad_n != self.orig_n:
            out = out[:, : self.orig_n].contiguous()
            return out.reshape(*lead, self.orig_n)
        return out.reshape(*lead, self.pad_n)


class HummingOnlineLinearMethod(LinearMethodBase):
    """Drop-in vLLM LinearMethodBase backed by HummingDenseOp."""

    def __init__(self, op: HummingDenseOp) -> None:
        self.op = op

    def create_weights(self, *args: Any, **kwargs: Any) -> None:
        pass

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.op(x)


class HummingDraftLMHead(nn.Module):
    """Quantized draft-only LM head wrapper compatible with LogitsProcessor."""

    def __init__(self, target_lm_head: nn.Module, vocab_size: int, mode: str) -> None:
        super().__init__()
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=0,
            org_vocab_end_index=vocab_size,
            num_org_vocab_padding=0,
        )
        self.tp_size = getattr(target_lm_head, "tp_size", 1)
        self.org_vocab_size = vocab_size
        self.padding_size = 0
        self.bias = None
        w = target_lm_head.weight.data[:vocab_size]
        # Keep padded columns contiguous so radix_topk does not copy;
        # LogitsProcessor._mask_outside_vocab masks [vocab_size:pad_n] in-place.
        self.op = HummingDenseOp(w, mode=mode, use_stream_k=True, slice_n=False)
        self.quant_method = HummingOnlineLinearMethod(self.op)


def quantize_dflash_model_inplace(
    dflash_model: nn.Module, target_lm_head: nn.Module | None
) -> None:
    """Apply env-selected online Humming quantization to DFlash2 draft model."""
    lmhead_mode = os.environ.get("INKLING2_OPT_DFLASH_LMHEAD", "").strip().lower()
    backbone_mode = os.environ.get("INKLING2_OPT_DFLASH_BACKBONE", "").strip().lower()
    if not lmhead_mode and not backbone_mode:
        return

    t0 = time.perf_counter()
    if lmhead_mode and target_lm_head is not None:
        vocab_size = min(
            int(getattr(dflash_model.candidate_logits_processor, "vocab_size", 200058)),
            target_lm_head.weight.shape[0],
        )
        dflash_model.lm_head = HummingDraftLMHead(
            target_lm_head, vocab_size=vocab_size, mode=lmhead_mode
        )
        if hasattr(dflash_model, "candidate_logits_processor"):
            dflash_model.candidate_logits_processor.head_dtype = None
        print(
            f"[inkling2-perf-opt] Quantized DFlash2 draft lm_head ({vocab_size}x"
            f"{target_lm_head.weight.shape[1]}) -> Humming {lmhead_mode}",
            flush=True,
        )

    if backbone_mode:
        draft_inner = dflash_model.model
        count = 0
        targets: list[nn.Module] = []
        for layer in draft_inner.layers:
            targets.extend(
                [
                    layer.self_attn.qkv_proj,
                    layer.self_attn.o_proj,
                    layer.mlp.gate_up_proj,
                    layer.mlp.down_proj,
                ]
            )
        if hasattr(draft_inner, "fc") and draft_inner.fc is not None:
            targets.append(draft_inner.fc)

        for mod in targets:
            if not hasattr(mod, "weight") or mod.weight.numel() == 0:
                continue
            bias = getattr(mod, "bias", None)
            op = HummingDenseOp(
                mod.weight.data, mode=backbone_mode, bias=bias, use_stream_k=True
            )
            mod.quant_method = HummingOnlineLinearMethod(op)
            mod.weight.data = mod.weight.data.new_empty((0,))
            if bias is not None:
                mod.bias = None
            count += 1

        fused_kv = getattr(draft_inner, "_fused_kv_weight", None)
        if fused_kv is not None and fused_kv.numel() > 0:
            fused_bias = getattr(draft_inner, "_fused_kv_bias", None)
            draft_inner._humming_fused_kv_op = HummingDenseOp(
                fused_kv, mode=backbone_mode, bias=fused_bias, use_stream_k=True
            )
            draft_inner._fused_kv_weight = fused_kv.new_empty((0,))
            count += 1

        torch.cuda.empty_cache()
        print(
            f"[inkling2-perf-opt] Quantized {count} DFlash2 backbone linears -> "
            f"Humming {backbone_mode} in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )
