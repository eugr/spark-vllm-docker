# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

_NEG_INF = -3.4028234663852886e38
_DSPARK_SCORE_K_BLOCK = 8
_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_DSPARK_MARKOV_V_BLOCK = 256
_DSPARK_MARKOV_R_BLOCK = 32
_DSPARK_HC_POST_H_BLOCK = 64


@triton.jit
def _dspark_markov_block_argmax_kernel(
    base_logits_ptr,
    markov_embed_ptr,
    markov_w2_ptr,
    block_vals_ptr,
    block_indices_ptr,
    batch_size,
    local_vocab_size,
    markov_rank: tl.constexpr,
    num_pad,
    base_stride_b,
    base_stride_v,
    embed_stride_b,
    embed_stride_r,
    w2_stride_v,
    w2_stride_r,
    out_stride_b,
    out_stride_block,
    V_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1).to(tl.int64)

    offs_v = pid_block * V_BLOCK + tl.arange(0, V_BLOCK)
    valid_vocab = offs_v < local_vocab_size
    padded_vocab = offs_v >= (local_vocab_size - num_pad)
    valid_vocab = valid_vocab & ~padded_vocab

    acc = tl.zeros((V_BLOCK,), dtype=tl.float32)
    for r_start in tl.static_range(0, markov_rank, R_BLOCK):
        offs_r = r_start + tl.arange(0, R_BLOCK)
        embed = tl.load(
            markov_embed_ptr + pid_b * embed_stride_b + offs_r * embed_stride_r,
            mask=offs_r < markov_rank,
            other=0.0,
        ).to(tl.float32)
        w2 = tl.load(
            markov_w2_ptr
            + offs_v[None, :] * w2_stride_v
            + offs_r[:, None] * w2_stride_r,
            mask=(offs_v[None, :] < local_vocab_size) & (offs_r[:, None] < markov_rank),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w2 * embed[:, None], axis=0)

    base = tl.load(
        base_logits_ptr + pid_b * base_stride_b + offs_v * base_stride_v,
        mask=offs_v < local_vocab_size,
        other=NEG_INF,
    ).to(tl.float32)
    scores = tl.where(valid_vocab, base + acc, NEG_INF)
    max_val = tl.max(scores, axis=0)
    local_idx = tl.argmax(scores, axis=0)
    token_idx = pid_block * V_BLOCK + local_idx

    tl.store(
        block_vals_ptr + pid_b * out_stride_b + pid_block * out_stride_block,
        max_val,
        mask=pid_b < batch_size,
    )
    tl.store(
        block_indices_ptr + pid_b * out_stride_b + pid_block * out_stride_block,
        token_idx,
        mask=pid_b < batch_size,
    )


@triton.jit
def _dspark_quant_dequant_nope_kernel(
    kv_ptr,
    num_rows,
    kv_stride_row,
    kv_stride_d,
    NOPE_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)

    offsets = group * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    mask = (row < num_rows) & (offsets < NOPE_DIM)
    vals = tl.load(
        kv_ptr + row * kv_stride_row + offsets * kv_stride_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    abs_vals = tl.where(offsets < NOPE_DIM, tl.abs(vals), 0.0)
    amax = tl.maximum(tl.max(abs_vals, axis=0), EPS)
    scale = tl.math.exp2(tl.ceil(tl.log2(amax * (1.0 / FP8_MAX))))
    quantized = tl.clamp(vals * (1.0 / scale), -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
    dequantized = quantized.to(tl.float32) * scale
    tl.store(
        kv_ptr + row * kv_stride_row + offsets * kv_stride_d,
        dequantized,
        mask=mask,
    )


@triton.jit
def _dspark_sparse_scores_kernel(
    q_ptr,
    draft_kv_ptr,
    main_kv_ptr,
    valid_main_lengths_ptr,
    scores_ptr,
    softmax_scale: tl.constexpr,
    q_stride_b,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    draft_stride_b,
    draft_stride_k,
    draft_stride_d,
    main_stride_b,
    main_stride_k,
    main_stride_d,
    scores_stride_b,
    scores_stride_q,
    scores_stride_h,
    scores_stride_k,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    KV_TOKENS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_bqh = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)

    h = pid_bqh % NUM_HEADS
    tmp = pid_bqh // NUM_HEADS
    q_idx = tmp % BLOCK_SIZE
    batch_idx = tmp // BLOCK_SIZE

    offs_k = pid_k * K_BLOCK + tl.arange(0, K_BLOCK)
    valid_main_len = tl.load(valid_main_lengths_ptr + batch_idx).to(tl.int64)
    is_main = offs_k < WINDOW_SIZE
    is_draft = (offs_k >= WINDOW_SIZE) & (offs_k < KV_TOKENS)
    is_valid = (is_main & (offs_k < valid_main_len)) | is_draft

    acc = tl.zeros((K_BLOCK,), dtype=tl.float32)
    for d_start in tl.static_range(0, HEAD_DIM, D_BLOCK):
        offs_d = d_start + tl.arange(0, D_BLOCK)
        q_vals = tl.load(
            q_ptr
            + batch_idx * q_stride_b
            + q_idx * q_stride_q
            + h * q_stride_h
            + offs_d * q_stride_d
        ).to(tl.float32)

        main_vals = tl.load(
            main_kv_ptr
            + batch_idx * main_stride_b
            + offs_k[:, None] * main_stride_k
            + offs_d[None, :] * main_stride_d,
            mask=(offs_k[:, None] < WINDOW_SIZE),
            other=0.0,
        )
        draft_k = offs_k - WINDOW_SIZE
        draft_vals = tl.load(
            draft_kv_ptr
            + batch_idx * draft_stride_b
            + draft_k[:, None] * draft_stride_k
            + offs_d[None, :] * draft_stride_d,
            mask=(draft_k[:, None] >= 0) & (draft_k[:, None] < BLOCK_SIZE),
            other=0.0,
        )
        kv_vals = tl.where(is_main[:, None], main_vals, draft_vals).to(tl.float32)
        acc += tl.sum(kv_vals * q_vals[None, :], axis=1)

    scores = acc * softmax_scale
    scores = tl.where(is_valid, scores, NEG_INF)
    tl.store(
        scores_ptr
        + batch_idx * scores_stride_b
        + q_idx * scores_stride_q
        + h * scores_stride_h
        + offs_k * scores_stride_k,
        scores,
        mask=offs_k < KV_TOKENS,
    )


@triton.jit
def _dspark_sparse_out_kernel(
    scores_ptr,
    draft_kv_ptr,
    main_kv_ptr,
    attn_sink_ptr,
    out_ptr,
    draft_stride_b,
    draft_stride_k,
    draft_stride_d,
    main_stride_b,
    main_stride_k,
    main_stride_d,
    scores_stride_b,
    scores_stride_q,
    scores_stride_h,
    scores_stride_k,
    out_stride_b,
    out_stride_q,
    out_stride_h,
    out_stride_d,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    KV_TOKENS: tl.constexpr,
    K_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
    NEG_INF: tl.constexpr,
):
    pid_bqh = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1).to(tl.int64)

    h = pid_bqh % NUM_HEADS
    tmp = pid_bqh // NUM_HEADS
    q_idx = tmp % BLOCK_SIZE
    batch_idx = tmp // BLOCK_SIZE

    offs_k = tl.arange(0, K_BLOCK)
    scores = tl.load(
        scores_ptr
        + batch_idx * scores_stride_b
        + q_idx * scores_stride_q
        + h * scores_stride_h
        + offs_k * scores_stride_k,
        mask=offs_k < KV_TOKENS,
        other=NEG_INF,
    ).to(tl.float32)

    sink = tl.load(attn_sink_ptr + h).to(tl.float32)
    normalizer = tl.maximum(tl.max(scores, axis=0), sink)
    weights = tl.exp(scores - normalizer)
    denom = tl.sum(weights, axis=0) + tl.exp(sink - normalizer)

    offs_d = pid_d * D_BLOCK + tl.arange(0, D_BLOCK)
    main_vals = tl.load(
        main_kv_ptr
        + batch_idx * main_stride_b
        + offs_k[:, None] * main_stride_k
        + offs_d[None, :] * main_stride_d,
        mask=(offs_k[:, None] < WINDOW_SIZE) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    draft_k = offs_k - WINDOW_SIZE
    draft_vals = tl.load(
        draft_kv_ptr
        + batch_idx * draft_stride_b
        + draft_k[:, None] * draft_stride_k
        + offs_d[None, :] * draft_stride_d,
        mask=(
            (draft_k[:, None] >= 0)
            & (draft_k[:, None] < BLOCK_SIZE)
            & (offs_d[None, :] < HEAD_DIM)
        ),
        other=0.0,
    )
    vals = tl.where((offs_k < WINDOW_SIZE)[:, None], main_vals, draft_vals).to(
        tl.float32
    )
    out = tl.sum(weights[:, None] * vals, axis=0) / denom
    tl.store(
        out_ptr
        + batch_idx * out_stride_b
        + q_idx * out_stride_q
        + h * out_stride_h
        + offs_d * out_stride_d,
        out,
        mask=offs_d < HEAD_DIM,
    )


@triton.jit
def _dspark_hc_post_mean_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    hc_mult: tl.constexpr,
    x_stride_t,
    x_stride_h,
    residual_stride_t,
    residual_stride_c,
    residual_stride_h,
    post_stride_t,
    post_stride_c,
    comb_stride_t,
    comb_stride_i,
    comb_stride_j,
    out_stride_t,
    out_stride_h,
    H_BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    h_start = tl.program_id(1).to(tl.int64) * H_BLOCK
    offs_h = h_start + tl.arange(0, H_BLOCK)
    mask_h = (token < num_tokens) & (offs_h < hidden_size)

    inv_hc = 1.0 / hc_mult
    post_sum = tl.full((), 0.0, dtype=tl.float32)
    for j in tl.static_range(0, hc_mult):
        post_sum += tl.load(
            post_ptr + token * post_stride_t + j * post_stride_c,
            mask=token < num_tokens,
            other=0.0,
        ).to(tl.float32)

    x_vals = tl.load(
        x_ptr + token * x_stride_t + offs_h * x_stride_h,
        mask=mask_h,
        other=0.0,
    ).to(tl.float32)
    acc = x_vals * post_sum * inv_hc

    for i in tl.static_range(0, hc_mult):
        comb_sum = tl.full((), 0.0, dtype=tl.float32)
        for j in tl.static_range(0, hc_mult):
            comb_sum += tl.load(
                comb_ptr
                + token * comb_stride_t
                + i * comb_stride_i
                + j * comb_stride_j,
                mask=token < num_tokens,
                other=0.0,
            ).to(tl.float32)
        residual_vals = tl.load(
            residual_ptr
            + token * residual_stride_t
            + i * residual_stride_c
            + offs_h * residual_stride_h,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        acc += residual_vals * comb_sum * inv_hc

    tl.store(
        out_ptr + token * out_stride_t + offs_h * out_stride_h,
        acc,
        mask=mask_h,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _post_strides(post: torch.Tensor) -> tuple[int, int]:
    if post.ndim == 2:
        return post.stride(0), post.stride(1)
    if post.ndim == 3 and post.shape[-1] == 1:
        return post.stride(0), post.stride(1)
    raise ValueError(f"Unsupported DSpark post mix shape: {tuple(post.shape)}")


def _post_2d(post: torch.Tensor) -> torch.Tensor:
    if post.ndim == 2:
        return post
    if post.ndim == 3 and post.shape[-1] == 1:
        return post.squeeze(-1)
    raise ValueError(f"Unsupported DSpark post mix shape: {tuple(post.shape)}")


def dspark_hc_post_mean_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Reference ``MHCPostOp(...).mean(dim=1)`` without materializing streams."""

    post_2d = _post_2d(post).to(torch.float32)
    residual_f = residual.to(torch.float32)
    comb_f = comb.to(torch.float32)
    x_f = x.to(torch.float32)
    hc_mult = residual.shape[1]
    post_term = post_2d.sum(dim=1, keepdim=True) * (1.0 / hc_mult) * x_f
    residual_weights = comb_f.sum(dim=-1) * (1.0 / hc_mult)
    residual_term = torch.sum(residual_weights.unsqueeze(-1) * residual_f, dim=1)
    return (post_term + residual_term).to(residual.dtype)


def dspark_hc_post_mean(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the DSpark target feature ``MHCPostOp(...).mean(dim=1)``.

    This is used by deferred DSpark target-layer capture. The target forward
    still uses the fused post/pre path, while this kernel writes only the
    reduced hidden feature required by the DSpark drafter.
    """

    if out is None:
        out = torch.empty(
            (x.shape[0], x.shape[-1]),
            dtype=residual.dtype,
            device=x.device,
        )

    if (
        not x.is_cuda
        or not residual.is_cuda
        or not post.is_cuda
        or not comb.is_cuda
        or not out.is_cuda
        or not HAS_TRITON
    ):
        out.copy_(dspark_hc_post_mean_torch(x, residual, post, comb))
        return out

    if residual.ndim != 3 or x.ndim != 2 or comb.ndim != 3:
        out.copy_(dspark_hc_post_mean_torch(x, residual, post, comb))
        return out
    if x.shape[0] != residual.shape[0] or comb.shape[:2] != residual.shape[:2]:
        out.copy_(dspark_hc_post_mean_torch(x, residual, post, comb))
        return out
    if comb.shape[2] != residual.shape[1] or x.shape[1] != residual.shape[2]:
        out.copy_(dspark_hc_post_mean_torch(x, residual, post, comb))
        return out
    if out.shape != x.shape:
        out.copy_(dspark_hc_post_mean_torch(x, residual, post, comb))
        return out

    num_tokens, hidden_size = x.shape
    hc_mult = residual.shape[1]
    if hc_mult <= 0 or hidden_size <= 0:
        return out

    post_stride_t, post_stride_c = _post_strides(post)
    grid = (num_tokens, triton.cdiv(hidden_size, _DSPARK_HC_POST_H_BLOCK))
    _dspark_hc_post_mean_kernel[grid](
        x,
        residual,
        post,
        comb,
        out,
        num_tokens,
        hidden_size,
        hc_mult,
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        post_stride_t,
        post_stride_c,
        comb.stride(0),
        comb.stride(1),
        comb.stride(2),
        out.stride(0),
        out.stride(1),
        H_BLOCK=_DSPARK_HC_POST_H_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return out


def dspark_markov_argmax_torch(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2_weight: torch.Tensor,
    *,
    num_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference local top-1 for ``base_logits + W1[token] @ W2``."""

    scores = base_logits.float() + torch.matmul(
        markov_embed.float(), markov_w2_weight.float().t()
    )
    if num_pad > 0:
        scores[..., -num_pad:] = -float("inf")
    local_max_vals, local_max_indices = scores.max(dim=-1)
    return local_max_vals, local_max_indices.to(torch.long)


def dspark_markov_argmax(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2_weight: torch.Tensor,
    *,
    num_pad: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused local top-1 for the DSpark Markov-head greedy step.

    This computes only the local maximum of the Markov-corrected logits instead
    of materializing the full Markov-logit vector. The caller still performs the
    tensor-parallel global top-1 reduction.
    """

    if (
        not base_logits.is_cuda
        or not markov_embed.is_cuda
        or not markov_w2_weight.is_cuda
        or not HAS_TRITON
    ):
        return dspark_markov_argmax_torch(
            base_logits,
            markov_embed,
            markov_w2_weight,
            num_pad=num_pad,
        )

    if base_logits.ndim != 2 or markov_embed.ndim != 2 or markov_w2_weight.ndim != 2:
        return dspark_markov_argmax_torch(
            base_logits,
            markov_embed,
            markov_w2_weight,
            num_pad=num_pad,
        )

    batch_size, local_vocab_size = base_logits.shape
    markov_rank = markov_embed.shape[-1]
    if markov_w2_weight.shape != (local_vocab_size, markov_rank):
        return dspark_markov_argmax_torch(
            base_logits,
            markov_embed,
            markov_w2_weight,
            num_pad=num_pad,
        )
    if markov_rank % _DSPARK_MARKOV_R_BLOCK != 0:
        return dspark_markov_argmax_torch(
            base_logits,
            markov_embed,
            markov_w2_weight,
            num_pad=num_pad,
        )

    num_blocks = triton.cdiv(local_vocab_size, _DSPARK_MARKOV_V_BLOCK)
    block_vals = torch.empty(
        (batch_size, num_blocks),
        device=base_logits.device,
        dtype=torch.float32,
    )
    block_indices = torch.empty(
        (batch_size, num_blocks),
        device=base_logits.device,
        dtype=torch.int64,
    )
    _dspark_markov_block_argmax_kernel[(batch_size, num_blocks)](
        base_logits,
        markov_embed,
        markov_w2_weight,
        block_vals,
        block_indices,
        batch_size,
        local_vocab_size,
        markov_rank,
        int(num_pad),
        base_logits.stride(0),
        base_logits.stride(1),
        markov_embed.stride(0),
        markov_embed.stride(1),
        markov_w2_weight.stride(0),
        markov_w2_weight.stride(1),
        block_vals.stride(0),
        block_vals.stride(1),
        V_BLOCK=_DSPARK_MARKOV_V_BLOCK,
        R_BLOCK=_DSPARK_MARKOV_R_BLOCK,
        NEG_INF=_NEG_INF,
        num_warps=8,
        num_stages=4,
    )
    max_block = block_vals.argmax(dim=-1, keepdim=True)
    local_max_vals = block_vals.gather(dim=-1, index=max_block).squeeze(-1)
    local_max_indices = block_indices.gather(dim=-1, index=max_block).squeeze(-1)
    return local_max_vals, local_max_indices


def dspark_quant_dequant_nope_torch(
    kv: torch.Tensor,
    rope_dim: int,
    group_size: int = 64,
) -> torch.Tensor:
    """Reference in-place FP8 quant-dequant for DSpark no-RoPE KV dims."""

    head_dim = kv.shape[-1]
    nope_dim = head_dim - rope_dim
    assert nope_dim >= 0
    if nope_dim == 0:
        return kv
    assert nope_dim % group_size == 0

    nope = kv[..., :nope_dim]
    original_shape = nope.shape
    groups = nope.reshape(-1, nope_dim // group_size, group_size).float()
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp_min_(1.0e-4)
    scale = torch.pow(
        torch.full((), 2.0, device=kv.device, dtype=torch.float32),
        torch.ceil(torch.log2(amax / _FP8_E4M3_MAX)),
    )
    quantized = torch.clamp(
        groups / scale,
        min=-_FP8_E4M3_MAX,
        max=_FP8_E4M3_MAX,
    ).to(torch.float8_e4m3fn)
    nope.copy_((quantized.float() * scale).reshape(original_shape).to(kv.dtype))
    return kv


def dspark_quant_dequant_nope(
    kv: torch.Tensor,
    rope_dim: int,
    group_size: int = 64,
) -> torch.Tensor:
    """In-place reference-parity FP8 quant-dequant for no-RoPE KV dims.

    The released DeepSeek V4 Flash path applies `act_quant(..., inplace=True)`
    after KV norm and RoPE, but only to the non-RoPE dimensions. This helper
    mirrors that QAT simulation while keeping the RoPE slice in bf16.
    """

    head_dim = kv.shape[-1]
    nope_dim = head_dim - rope_dim
    assert nope_dim >= 0
    if nope_dim == 0:
        return kv
    assert nope_dim % group_size == 0

    if not kv.is_cuda or not HAS_TRITON:
        return dspark_quant_dequant_nope_torch(kv, rope_dim, group_size)

    if not kv.is_contiguous():
        return dspark_quant_dequant_nope_torch(kv, rope_dim, group_size)

    flat = kv.view(-1, head_dim)
    grid = (flat.shape[0], triton.cdiv(nope_dim, group_size))
    _dspark_quant_dequant_nope_kernel[grid](
        flat,
        flat.shape[0],
        flat.stride(0),
        flat.stride(1),
        NOPE_DIM=nope_dim,
        GROUP_SIZE=group_size,
        FP8_MAX=_FP8_E4M3_MAX,
        EPS=1.0e-4,
        num_warps=2,
        num_stages=4,
    )
    return kv


def dspark_sparse_attention_torch(
    q: torch.Tensor,
    draft_kv: torch.Tensor,
    main_kv_cache: torch.Tensor,
    valid_main_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference DSpark sparse attention matching the CUDA kernel contract."""

    batch_size, block_size, num_heads, head_dim = q.shape
    window_size = main_kv_cache.shape[1]
    main_kv = main_kv_cache[:batch_size]
    kv = torch.cat([main_kv, draft_kv], dim=1)
    kv_tokens = window_size + block_size

    kv_idx = torch.arange(kv_tokens, device=q.device)
    valid_main = kv_idx.unsqueeze(0) < valid_main_lengths.to(torch.long).unsqueeze(1)
    valid = torch.where(
        kv_idx.unsqueeze(0) < window_size,
        valid_main,
        torch.ones((batch_size, kv_tokens), dtype=torch.bool, device=q.device),
    )

    scores = torch.einsum("bqhd,bkd->bqhk", q.float(), kv.float())
    scores.mul_(softmax_scale)
    scores.masked_fill_(~valid[:, None, None, :], _NEG_INF)
    normalizer = torch.maximum(
        scores.max(dim=-1, keepdim=True).values,
        attn_sink[:num_heads].view(1, 1, num_heads, 1),
    )
    weights = torch.exp(scores - normalizer)
    denom = weights.sum(dim=-1, keepdim=True) + torch.exp(
        attn_sink[:num_heads].view(1, 1, num_heads, 1) - normalizer
    )
    out = torch.einsum("bqhk,bkd->bqhd", weights.to(kv.dtype), kv) / denom.to(kv.dtype)
    return out.reshape(batch_size * block_size, num_heads, head_dim)


def dspark_sparse_attention(
    q: torch.Tensor,
    draft_kv: torch.Tensor,
    main_kv_cache: torch.Tensor,
    valid_main_lengths: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    scores_buffer: torch.Tensor,
) -> torch.Tensor:
    """Run DSpark sparse attention with a Triton CUDA kernel when available."""

    if not q.is_cuda or not HAS_TRITON:
        return dspark_sparse_attention_torch(
            q,
            draft_kv,
            main_kv_cache,
            valid_main_lengths,
            attn_sink,
            softmax_scale,
        )

    batch_size, block_size, num_heads, head_dim = q.shape
    window_size = main_kv_cache.shape[1]
    kv_tokens = window_size + block_size
    assert scores_buffer.shape[:4] == (batch_size, block_size, num_heads, kv_tokens)
    assert head_dim % 64 == 0

    scores = scores_buffer
    out = torch.empty_like(q)
    k_score_block = _DSPARK_SCORE_K_BLOCK
    k_out_block = _next_power_of_2(kv_tokens)
    d_score_block = 64
    d_out_block = 32

    grid_scores = (
        batch_size * block_size * num_heads,
        triton.cdiv(kv_tokens, k_score_block),
    )
    _dspark_sparse_scores_kernel[grid_scores](
        q,
        draft_kv,
        main_kv_cache,
        valid_main_lengths,
        scores,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        draft_kv.stride(0),
        draft_kv.stride(1),
        draft_kv.stride(2),
        main_kv_cache.stride(0),
        main_kv_cache.stride(1),
        main_kv_cache.stride(2),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        BLOCK_SIZE=block_size,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        WINDOW_SIZE=window_size,
        KV_TOKENS=kv_tokens,
        K_BLOCK=k_score_block,
        D_BLOCK=d_score_block,
        NEG_INF=_NEG_INF,
        num_warps=2,
        num_stages=4,
    )

    grid_out = (
        batch_size * block_size * num_heads,
        triton.cdiv(head_dim, d_out_block),
    )
    _dspark_sparse_out_kernel[grid_out](
        scores,
        draft_kv,
        main_kv_cache,
        attn_sink,
        out,
        draft_kv.stride(0),
        draft_kv.stride(1),
        draft_kv.stride(2),
        main_kv_cache.stride(0),
        main_kv_cache.stride(1),
        main_kv_cache.stride(2),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_SIZE=block_size,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        WINDOW_SIZE=window_size,
        KV_TOKENS=kv_tokens,
        K_BLOCK=k_out_block,
        D_BLOCK=d_out_block,
        NEG_INF=_NEG_INF,
        num_warps=8,
        num_stages=4,
    )
    return out.reshape(batch_size * block_size, num_heads, head_dim)
