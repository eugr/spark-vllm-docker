"""Inkling-only adapter for the vendored SM12 FlashAttention 4 bundle.

The downstream bundle exposes preallocated output support in its internal
forward entry point, but not in the public ``flash_attn_varlen_func`` wrapper.
Current vLLM Inkling passes ``out=`` during both warmup and inference, so this
adapter preserves that contract without changing the bundled kernel sources.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm.third_party.inkling_sm120_fa4.interface import _flash_attn_fwd


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    min_seqlen_k: int | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    gather_kv_indices: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: torch.Tensor | None = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    score_mod: Callable[..., Any] | None = None,
    aux_tensors: list[torch.Tensor] | None = None,
    return_lse: bool = False,
    dropout_p: float = 0.0,
    dropout_seed: int | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the vendored paged-KV kernel with vLLM's Inkling call contract."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise RuntimeError(
            "The Inkling SM12 paged-KV mod is inference-only; autograd is unsupported."
        )
    if deterministic:
        raise ValueError(
            "deterministic=True is not supported by the Inkling SM12 inference adapter."
        )

    # The bundled SM12 interface and its published validation explicitly use
    # single-split decode. Clamp here as a second line of defense in case a
    # future Inkling caller bypasses inkling_fa4_num_splits().
    num_splits = 1

    return _flash_attn_fwd(
        q=q,
        k=k,
        v=v,
        qv=qv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        min_seqlen_k=min_seqlen_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        softcap=softcap,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        learnable_sink=learnable_sink,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        return_lse=return_lse,
        out=out,
        gather_kv_indices=gather_kv_indices,
        dropout_p=dropout_p,
        dropout_seed=dropout_seed,
    )
