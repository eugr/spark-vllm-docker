# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 (consumer Blackwell) sparse-MLA impl for DeepSeek-V4.

Counterpart to :class:`DeepseekV4FlashMLASparseImpl` (Hopper / SM10x). The
forward path is driven by flashinfer's :class:`BatchSparseMLAPagedAttention
Wrapper` — the same wrapper used by the V32-family SPARSE_MLA_SM120 backend —
which auto-dispatches decode (num_tokens <= 64) and prefill internally and
accepts the SWA + compressed-indexer dual cache through its ``extra_kv_cache``
parameter. Decode scratch is borrowed from vLLM's shared workspace so large
C128A contexts do not allocate per-layer split-K buffers.

Selected by ``_select_v4_sparse_impl()`` in :mod:`vllm.models.deepseek_v4
.attention` when the runtime compute capability is SM120; the
flashinfer wrapper itself lives on the layer (``layer._sparse_mla_wrapper``)
only for its reusable LSE buffer; split-K decode scratch is supplied per call.
"""

import os
from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
    DeepseekV4SparseMLAAttentionImpl,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseMetadata
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.attention import DeepseekV4MLAAttention
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)


_DECODE_MAX_TOKENS = 64
_DECODE_SPLIT_TILE = 64
_C128A_TOPK_ALIGNMENT = 128


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def _max_decode_workspace_tokens(max_num_batched_tokens: int) -> int:
    return min(int(max_num_batched_tokens), _DECODE_MAX_TOKENS)


def _c128a_max_compressed(max_model_len: int, compress_ratio: int) -> int:
    return (
        _cdiv(
            _cdiv(max_model_len, compress_ratio),
            _C128A_TOPK_ALIGNMENT,
        )
        * _C128A_TOPK_ALIGNMENT
    )


def _env_enabled(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in ("0", "false", "no", "off", "")


def _use_b12x_compressed_mla() -> bool:
    return _env_enabled("VLLM_DSV4_B12X_COMPRESSED_MLA")


def _extra_topk_capacity(layer: "DeepseekV4MLAAttention") -> int:
    if layer.compress_ratio <= 1:
        return 0
    if layer.compress_ratio == 4:
        assert layer.topk_indices_buffer is not None
        return int(layer.topk_indices_buffer.shape[-1])
    if layer.compress_ratio == 128:
        return _c128a_max_compressed(layer.max_model_len, layer.compress_ratio)
    raise ValueError(
        f"Unsupported compress_ratio={layer.compress_ratio}; "
        "expected 1, 4, or 128."
    )


def _get_decode_scratch(
    num_tokens: int,
    num_heads: int,
    d_v: int,
    topk: int,
    extra_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_splits = _decode_num_splits(topk, extra_topk)
    mid_out, mid_lse = current_workspace_manager().get_simultaneous(
        ((num_tokens, num_heads, num_splits, d_v), torch.bfloat16),
        ((num_tokens, num_heads, num_splits), torch.float32),
    )
    return mid_out, mid_lse


def _b12x_index_matrix(indices: torch.Tensor | None) -> torch.Tensor | None:
    if indices is None:
        return None
    if indices.ndim == 3:
        assert indices.shape[1] == 1
        return indices.squeeze(1)
    return indices


def _get_b12x_decode_workspace(
    layer: "DeepseekV4MLAAttention",
    *,
    extra_topk: int,
):
    from b12x.attention.workspace import B12XAttentionWorkspace

    total_topk = int(layer.window_size) + int(extra_topk)
    max_rows = _max_decode_workspace_tokens(layer.max_num_batched_tokens)
    max_chunks = _decode_num_splits(layer.window_size, extra_topk)

    workspace = getattr(layer, "_b12x_compressed_mla_workspace", None)
    if (
        workspace is None
        or int(workspace.topk) < total_topk
        or int(workspace.max_total_q) < max_rows
        or int(workspace.max_chunks_per_row) < max_chunks
        or int(workspace.num_q_heads) != int(layer.padded_heads)
    ):
        device = layer.attn_sink.device
        if device.type != "cuda":
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        workspace = B12XAttentionWorkspace(
            mode="decode",
            device=device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=int(layer.padded_heads),
            head_dim=512,
            v_head_dim=512,
            topk=total_topk,
            max_total_q=max_rows,
            max_batch=max_rows,
            max_page_table_width=total_topk,
            max_paged_q_rows=max_rows,
            page_size=int(layer.swa_cache_layer.block_size),
            padded_heads=int(layer.padded_heads),
            max_chunks_per_row=max_chunks,
        )
        workspace.kv_chunk_size_ptr = torch.empty(
            (1,), dtype=torch.int32, device=device
        )
        workspace.num_chunks_ptr = torch.empty((1,), dtype=torch.int32, device=device)
        layer._b12x_compressed_mla_workspace = workspace
        logger.info_once(
            "DeepSeek V4 SM120 b12x compressed MLA decode enabled "
            "(topk=%d, max_rows=%d, max_chunks=%d).",
            total_topk,
            max_rows,
            max_chunks,
        )
    return workspace


class DeepseekV4SM120SparseBackend(DeepseekV4FlashMLASparseBackend):
    """SM120 variant. Geometry is identical to the FlashMLA parent (same KV
    layout, head size, block size); the only thing that changes is the impl
    class returned by ``get_impl_cls``."""

    @staticmethod
    def get_name() -> str:
        return "DSV4_SPARSE_MLA_SM120"

    @staticmethod
    def get_impl_cls() -> type["DeepseekV4SM120SparseImpl"]:
        return DeepseekV4SM120SparseImpl


class DeepseekV4SM120SparseImpl(DeepseekV4SparseMLAAttentionImpl):
    """SM120 flashinfer-wrapper-driven sparse-MLA impl for DeepseekV4.

    The wrapper auto-dispatches decode (num_tokens <= 64) and prefill on
    num_tokens, so this impl issues a single ``wrapper.run`` per chunk —
    no separate prefill kernel call, no plan() step.
    """

    backend_cls: ClassVar[type[AttentionBackend]] = DeepseekV4SM120SparseBackend

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 SM120 sparse MLA does not support {num_heads} heads "
            "(kernel requires h_q in {16, 32, 64, 128})."
        )

    @classmethod
    def forward_mqa(  # type: ignore[override]
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            cls._reserve_decode_workspace(layer)
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(layer.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(layer.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = layer.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation; layer.kv_cache may be empty after profiling cleanup.
        self_kv_cache = layer.kv_cache if not swa_only else None
        swa_kv_cache = layer.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            cls._forward_prefill(
                layer=layer,
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            cls._forward_decode(
                layer=layer,
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    @classmethod
    def _reserve_decode_workspace(cls, layer: "DeepseekV4MLAAttention") -> None:
        extra_topk = _extra_topk_capacity(layer)
        _get_decode_scratch(
            _max_decode_workspace_tokens(layer.max_num_batched_tokens),
            layer.padded_heads,
            512,
            layer.window_size,
            extra_topk,
        )
        if _use_b12x_compressed_mla():
            _get_b12x_decode_workspace(layer, extra_topk=extra_topk)

    @classmethod
    def _forward_decode(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // layer.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if layer.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert layer.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    layer.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                    kv_cache.shape[0],
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        extra_topk = topk_indices.shape[-1] if topk_indices is not None else 0
        mid_out, mid_lse = _get_decode_scratch(
            num_decode_tokens,
            q.shape[1],
            output.shape[-1],
            swa_indices.shape[-1],
            extra_topk,
        )

        # Treat queries in the same seq as independent queries (attended
        # purely by the generated indices). q arrives pre-padded to
        # layer.padded_heads by the outer wrapper.
        if _use_b12x_compressed_mla():
            from b12x.attention.mla.compressed_api import (
                compressed_mla_decode_forward,
            )

            workspace = _get_b12x_decode_workspace(layer, extra_topk=extra_topk)
            workspace.tmp_output = mid_out
            workspace.tmp_lse = mid_lse
            workspace.output_buffer = output
            result = compressed_mla_decode_forward(
                q_all=q,
                swa_k_cache=layer.swa_cache_layer.kv_cache,
                swa_indices=_b12x_index_matrix(swa_indices),
                swa_topk_lengths=swa_lens,
                workspace=workspace,
                sm_scale=layer.scale,
                swa_page_size=swa_metadata.block_size,
                indexed_k_cache=kv_cache,
                indexed_indices=_b12x_index_matrix(topk_indices),
                indexed_topk_lengths=topk_lens,
                indexed_page_size=block_size if kv_cache is not None else None,
                attn_sink=layer.attn_sink,
                expected_num_q_heads=q.shape[1],
                backend="sm120_unified",
            )
            if result.data_ptr() != output.data_ptr():
                output.copy_(result)
            return

        q = q.unsqueeze(1)
        swa_cache = layer.swa_cache_layer.kv_cache.unsqueeze(-2)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        assert layer._sparse_mla_wrapper is not None, (
            "DeepseekV4SM120SparseImpl requires layer._sparse_mla_wrapper; "
            "the flashinfer wrapper must be constructed in the layer __init__."
        )
        layer._sparse_mla_wrapper.run(
            q=q,
            kv_cache=swa_cache,
            indices=swa_indices,
            output=output,
            sm_scale=layer.scale,
            topk_length=swa_lens,
            attn_sink=layer.attn_sink,
            extra_kv_cache=kv_cache if not swa_only else None,
            extra_indices=topk_indices,
            extra_topk_length=topk_lens,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    @classmethod
    def _forward_prefill(
        cls,
        layer: "DeepseekV4MLAAttention",
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        # `_dummy_run` passes synthetic non-None attn_metadata for swa-only
        # layers during cudagraph capture, so check compress_ratio directly.
        swa_only = layer.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif layer.compress_ratio == 4:
            assert layer.topk_indices_buffer is not None
            local_topk_indices = layer.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            # C128A: pre-computed during metadata build.
            assert attn_metadata is not None
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_topk_indices: torch.Tensor | None = None
        extra_topk_lens: torch.Tensor | None = None
        if local_topk_indices is not None:
            assert attn_metadata is not None
            assert swa_metadata.token_to_req_indices is not None
            assert swa_metadata.is_valid_token is not None
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            # FlashInfer prefill expects physical KV slots; keep padding rows
            # masked through the metadata validity mask.
            block_size = attn_metadata.block_size // layer.compress_ratio
            extra_topk_indices, extra_topk_lens = compute_global_topk_indices_and_lens(
                local_topk_indices,
                swa_metadata.token_to_req_indices[prefill_token_slice],
                attn_metadata.block_table,
                block_size,
                swa_metadata.is_valid_token[prefill_token_slice],
                compressed_k_cache.shape[0],
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None
        assert layer._sparse_mla_wrapper is not None

        # unsqueeze(-2) adds the h_kv=1 axis without copying.
        swa_kv_paged = swa_k_cache.unsqueeze(-2)
        if swa_only:
            extra_kv_paged = None
        else:
            assert compressed_k_cache is not None
            extra_kv_paged = compressed_k_cache.unsqueeze(-2)

        num_chunks = (
            num_prefills + cls.PREFILL_CHUNK_SIZE - 1
        ) // cls.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * cls.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + cls.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_indices_chunk = (
                extra_topk_indices[query_start:query_end]
                if extra_topk_indices is not None
                else None
            )
            extra_topk_length_chunk = (
                extra_topk_lens[query_start:query_end]
                if extra_topk_lens is not None
                else None
            )
            chunk_tokens = query_end - query_start
            mid_out = None
            mid_lse = None
            if chunk_tokens <= _DECODE_MAX_TOKENS:
                extra_topk = (
                    extra_indices_chunk.shape[-1]
                    if extra_indices_chunk is not None
                    else 0
                )
                mid_out, mid_lse = _get_decode_scratch(
                    chunk_tokens,
                    q.shape[1],
                    output.shape[-1],
                    swa_metadata.prefill_swa_indices.shape[-1],
                    extra_topk,
                )

            layer._sparse_mla_wrapper.run(
                q=q[query_start:query_end],
                kv_cache=swa_kv_paged,
                indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                output=output[query_start:query_end],
                sm_scale=layer.scale,
                topk_length=swa_metadata.prefill_swa_lens[query_start:query_end],
                attn_sink=layer.attn_sink,
                extra_kv_cache=extra_kv_paged,
                extra_indices=extra_indices_chunk,
                extra_topk_length=extra_topk_length_chunk,
                mid_out=mid_out,
                mid_lse=mid_lse,
            )
