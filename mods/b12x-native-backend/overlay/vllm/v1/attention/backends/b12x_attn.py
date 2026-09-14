# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X paged GQA attention backend.

This is the regular, non-MLA counterpart to the existing B12X sparse-MLA
backend. The integration follows the same eager PLAN -> BIND -> KERNEL shape:
vLLM owns one uint8 scratch buffer borrowed from ``current_workspace_manager()``,
and b12x only receives plain scratch views from ``plan.bind(...)``. There are no
b12x workspaces, arenas, cached workspace pools, or allocator-owned buffers on
the vLLM path.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_kv_cache_layout,
    set_kv_cache_layout,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_B12X_SUPPORTED_PAGE_SIZES = (64, 128)
_B12X_PREFERRED_PAGE_SIZE = 128
_MIN_PAGED_TILE_Q = 16
_B12X_FP8_KV_CACHE_DTYPES = ("fp8", "fp8_e4m3")
_B12X_SUPPORTED_KV_CACHE_DTYPES = (
    "auto",
    "float16",
    "bfloat16",
    *_B12X_FP8_KV_CACHE_DTYPES,
)


def _uses_b12x_dflash_attention(spec_config: Any) -> bool:
    return bool(
        spec_config is not None
        and getattr(spec_config, "method", None) == "dflash"
        and getattr(spec_config, "attention_backend", None)
        == AttentionBackendEnum.B12X_ATTN
    )


def _max_page_table_width(
    max_model_len: int,
    block_size: int,
    max_num_batched_tokens: int,
    mamba_cache_mode: str,
) -> int:
    width = max(cdiv(max(max_model_len, 1), block_size), 1)
    if mamba_cache_mode == "align":
        # Hybrid cache setup can enlarge the storage block after attention
        # layers are initialized. Its expansion into kernel-sized blocks adds
        # at most one storage block of trailing page-table capacity.
        width += cdiv(max_num_batched_tokens, block_size)
    return width


def _kv_page_size(key_cache: torch.Tensor, value_cache: torch.Tensor) -> int:
    """Return the static kernel page geometry negotiated by vLLM.

    The KV manager can split the configured storage block into a smaller
    kernel page when another backend shares its cache group.  This notably
    happens when a B12X target shares a DFlash cache group with FlashInfer.
    Cache shapes are fixed before graph capture, so this is not a live-length
    policy decision.
    """
    if key_cache.ndim < 2 or value_cache.ndim < 2:
        raise ValueError(
            "B12X_ATTN expects paged K/V caches with a page dimension, got "
            f"{tuple(key_cache.shape)} and {tuple(value_cache.shape)}."
        )
    key_page_size = int(key_cache.shape[1])
    value_page_size = int(value_cache.shape[1])
    if key_page_size != value_page_size:
        raise ValueError(
            "B12X_ATTN requires matching K/V page sizes, got "
            f"{key_page_size} and {value_page_size}."
        )
    if key_page_size not in _B12X_SUPPORTED_PAGE_SIZES:
        raise ValueError(
            "B12X_ATTN requires runtime page size in "
            f"{_B12X_SUPPORTED_PAGE_SIZES}, got {key_page_size}."
        )
    return key_page_size


@triton.jit(do_not_specialize=["num_reqs"])
def _b12x_noncausal_cu_lens_kernel(
    seq_lens_ptr,
    cu_seqlens_q_src_ptr,
    cu_seqlens_q_dst_ptr,
    cu_seqlens_k_dst_ptr,
    num_reqs,
    MAX_BATCH: tl.constexpr,
    KV_WINDOW: tl.constexpr,
):
    q_acc = tl.full((), 0, tl.int32)
    k_acc = tl.full((), 0, tl.int32)
    for req_idx in tl.static_range(0, MAX_BATCH):
        tl.store(cu_seqlens_q_dst_ptr + req_idx, q_acc)
        tl.store(cu_seqlens_k_dst_ptr + req_idx, k_acc)
        active = req_idx < num_reqs
        seq_len = tl.load(seq_lens_ptr + req_idx, mask=active, other=0)
        q_start = tl.load(cu_seqlens_q_src_ptr + req_idx, mask=active, other=0)
        q_end = tl.load(cu_seqlens_q_src_ptr + req_idx + 1, mask=active, other=0)
        q_len = tl.where(active, q_end - q_start, 0)
        k_len = tl.where(active, tl.minimum(seq_len, KV_WINDOW), 0)
        q_acc += q_len
        k_acc += k_len
    tl.store(cu_seqlens_q_dst_ptr + MAX_BATCH, q_acc)
    tl.store(cu_seqlens_k_dst_ptr + MAX_BATCH, k_acc)


@triton.jit(do_not_specialize=["num_reqs"])
def _b12x_gather_paged_kv_to_contiguous_kernel(
    k_cache_ptr,
    v_cache_ptr,
    out_k_ptr,
    out_v_ptr,
    block_table_ptr,
    seq_lens_ptr,
    cu_seqlens_k_ptr,
    k_scale_ptr,
    v_scale_ptr,
    num_reqs,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    K_CACHE_STRIDE_0: tl.constexpr,
    K_CACHE_STRIDE_1: tl.constexpr,
    K_CACHE_STRIDE_2: tl.constexpr,
    K_CACHE_STRIDE_3: tl.constexpr,
    V_CACHE_STRIDE_0: tl.constexpr,
    V_CACHE_STRIDE_1: tl.constexpr,
    V_CACHE_STRIDE_2: tl.constexpr,
    V_CACHE_STRIDE_3: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_WINDOW: tl.constexpr,
    HAS_FP8_KV: tl.constexpr,
    K_SCALE_NUMEL: tl.constexpr,
    V_SCALE_NUMEL: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req_idx = tl.program_id(0)
    tok_block = tl.program_id(1)
    head_dim_block = tl.program_id(2)

    num_d_blocks = tl.cdiv(HEAD_DIM, BLOCK_D)
    head_idx = head_dim_block // num_d_blocks
    dim_block = head_dim_block - head_idx * num_d_blocks

    token_offsets = tok_block * BLOCK_T + tl.arange(0, BLOCK_T)
    dim_offsets = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    active_req = req_idx < num_reqs

    seq_len = tl.load(seq_lens_ptr + req_idx, mask=active_req, other=0)
    gather_len = tl.minimum(seq_len, KV_WINDOW)
    start_token = seq_len - gather_len
    logical_tokens = start_token + token_offsets
    valid_tokens = active_req & (token_offsets < gather_len)

    page_offsets = logical_tokens // PAGE_SIZE
    page_slots = logical_tokens - page_offsets * PAGE_SIZE
    block_ids = tl.load(
        block_table_ptr + req_idx * BLOCK_TABLE_STRIDE + page_offsets,
        mask=valid_tokens,
        other=0,
    )
    out_start = tl.load(cu_seqlens_k_ptr + req_idx, mask=active_req, other=0)
    out_tokens = out_start + token_offsets

    k_cache_offsets = (
        block_ids[:, None] * K_CACHE_STRIDE_0
        + page_slots[:, None] * K_CACHE_STRIDE_1
        + head_idx * K_CACHE_STRIDE_2
        + dim_offsets[None, :] * K_CACHE_STRIDE_3
    )
    v_cache_offsets = (
        block_ids[:, None] * V_CACHE_STRIDE_0
        + page_slots[:, None] * V_CACHE_STRIDE_1
        + head_idx * V_CACHE_STRIDE_2
        + dim_offsets[None, :] * V_CACHE_STRIDE_3
    )
    out_offsets = (
        out_tokens[:, None] * NUM_KV_HEADS * HEAD_DIM
        + head_idx * HEAD_DIM
        + dim_offsets[None, :]
    )
    mask = valid_tokens[:, None] & (dim_offsets[None, :] < HEAD_DIM)

    k_values = tl.load(k_cache_ptr + k_cache_offsets, mask=mask, other=0.0)
    v_values = tl.load(v_cache_ptr + v_cache_offsets, mask=mask, other=0.0)
    if HAS_FP8_KV:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
        if K_SCALE_NUMEL == NUM_KV_HEADS:
            k_scale = tl.load(k_scale_ptr + head_idx)
        if V_SCALE_NUMEL == NUM_KV_HEADS:
            v_scale = tl.load(v_scale_ptr + head_idx)
        k_values = k_values.to(tl.float32) * k_scale
        v_values = v_values.to(tl.float32) * v_scale

    tl.store(out_k_ptr + out_offsets, k_values, mask=mask)
    tl.store(out_v_ptr + out_offsets, v_values, mask=mask)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, value, default)
        return default
    if parsed <= 0:
        logger.warning("Ignoring non-positive %s=%r; using %d", name, value, default)
        return default
    return parsed


def _env_optional_storage_limit(name: str, *, allow_zero: bool) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using b12x's planned capacity",
            name,
            value,
        )
        return None
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        logger.warning(
            "Ignoring %s=%r below the minimum %d; using b12x's planned capacity",
            name,
            value,
            minimum,
        )
        return None
    return parsed


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    logger.warning("Ignoring invalid %s=%r; using %s", name, value, default)
    return default


def _disable_cutlass_memory_debug_snapshot_if_off() -> None:
    if _env_flag("CUTLASS_DSL_CUDA_MEMORY_DEBUG", False):
        return
    try:
        from cutlass.base_dsl.runtime import cuda as cuda_helpers
    except Exception:
        return
    if getattr(cuda_helpers, "_vllm_b12x_memory_debug_snapshot_patched", False):
        return

    def _empty_memory_debug_snapshot() -> dict[str, int | None]:
        return {
            "free": None,
            "total": None,
            "used": None,
            "torch_allocated": None,
            "torch_reserved": None,
            "external": None,
            "device": None,
        }

    cuda_helpers._memory_debug_snapshot = _empty_memory_debug_snapshot
    cuda_helpers._vllm_b12x_memory_debug_snapshot_patched = True


def _capture_alloc_forbidden() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _ensure_i32_contiguous(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        if _capture_alloc_forbidden():
            raise RuntimeError(
                f"B12X_ATTN would convert {name} to int32 during CUDA graph "
                "capture. Prepare int32 metadata before capture."
            )
        tensor = tensor.to(torch.int32)
    if not tensor.is_contiguous():
        if _capture_alloc_forbidden():
            raise RuntimeError(
                f"B12X_ATTN would make {name} contiguous during CUDA graph "
                "capture. Prepare contiguous metadata before capture."
            )
        tensor = tensor.contiguous()
    return tensor


def _dtype_from_cache_config(
    kv_cache_dtype: str,
    vllm_config: VllmConfig,
) -> torch.dtype:
    if kv_cache_dtype == "float16":
        return torch.float16
    if kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if kv_cache_dtype in _B12X_FP8_KV_CACHE_DTYPES:
        return current_platform.fp8_dtype()
    if kv_cache_dtype != "auto":
        raise NotImplementedError(
            "B12X_ATTN currently supports only auto, float16, bfloat16, "
            "fp8, and fp8_e4m3 "
            f"KV cache dtypes; got {kv_cache_dtype!r}."
        )
    return vllm_config.model_config.dtype


def _is_b12x_fp8_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in _B12X_FP8_KV_CACHE_DTYPES


class B12XPagedAttentionBackend(AttentionBackend):
    """Opt-in b12x paged attention backend for regular/GQA decoder layers."""

    # None means use the regular two-plane KV layout. Per-layer dynamic
    # subclasses set this to use the packed DiffKV-style layout.
    head_size_v: ClassVar[int | None] = None
    _impl_cls: ClassVar[type[B12XPagedAttentionImpl] | None] = None

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        return "B12X_ATTN"

    @classmethod
    def get_impl_cls(cls) -> type[B12XPagedAttentionImpl]:
        return cls._impl_cls or B12XPagedAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[B12XPagedMetadataBuilder]:
        return B12XPagedMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return list(_B12X_SUPPORTED_PAGE_SIZES)

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or int(block_size) in _B12X_SUPPORTED_PAGE_SIZES

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if int(default_block_size) in _B12X_SUPPORTED_PAGE_SIZES:
            return int(default_block_size)
        return _B12X_PREFERRED_PAGE_SIZE

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 192, 256]

    @classmethod
    def _uses_packed_kv_cache(cls) -> bool:
        return cls.head_size_v is not None

    @classmethod
    def _get_head_size_v(cls, head_size: int) -> int:
        return int(head_size if cls.head_size_v is None else cls.head_size_v)

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Consumer Blackwell SM120 / SM121. The b12x paged kernels also gate
        # internally, but keep vLLM selection fail-fast and explicit.
        return capability.major == 12

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if (
            kv_cache_dtype is not None
            and is_quantized_kv_cache(kv_cache_dtype)
            and not _is_b12x_fp8_kv_cache(kv_cache_dtype)
        ):
            return (
                "B12X_ATTN currently supports only fp8/fp8_e4m3 quantized "
                "KV cache dtypes"
            )
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        if parallel_config.decode_context_parallel_size > 1:
            return "B12X_ATTN does not yet support decode context parallelism"
        if parallel_config.prefill_context_parallel_size > 1:
            return "B12X_ATTN does not yet support prefill context parallelism"
        return None

    @classmethod
    def get_kv_cache_shape(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size not in _B12X_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                "B12X_ATTN requires block_size in "
                f"{_B12X_SUPPORTED_PAGE_SIZES}, got {block_size}."
            )
        if cache_dtype_str not in _B12X_SUPPORTED_KV_CACHE_DTYPES:
            raise ValueError(
                "B12X_ATTN currently supports only auto, float16, bfloat16, "
                "fp8, and fp8_e4m3 "
                f"KV cache dtypes; got {cache_dtype_str!r}."
            )
        if cls._uses_packed_kv_cache():
            head_size_v = cls._get_head_size_v(head_size)
            return (
                num_blocks,
                block_size,
                num_kv_heads,
                head_size + head_size_v,
            )
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def get_kv_cache_stride_order(
        cls,
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout != "NHD":
            raise ValueError(
                f"B12X_ATTN requires NHD KV cache layout; got {cache_layout!r}."
            )
        if cls._uses_packed_kv_cache():
            if include_num_layers_dimension:
                return (1, 0, 2, 3, 4)
            return (0, 1, 2, 3)
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        return "NHD"


@dataclass
class B12XPagedMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class B12XPagedMetadataBuilder(AttentionMetadataBuilder[B12XPagedMetadata]):
    """Metadata builder for B12X_ATTN.

    Decode and uniform speculative-verifier batches use preplanned graph
    buckets. Extend/prefill remains eager and does not affect uniform decode
    graph eligibility.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        del vllm_config, kv_cache_spec
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> B12XPagedMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        return B12XPagedMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            max_seq_len=cm.max_seq_len,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            causal=cm.causal,
        )

    def update_block_table(
        self,
        metadata: B12XPagedMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> B12XPagedMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata


class B12XPagedAttentionImpl(AttentionImpl[B12XPagedMetadata]):
    """b12x paged GQA attention implementation."""

    can_return_lse_for_decode: bool = False
    head_size_v: ClassVar[int | None] = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("B12X_ATTN does not support ALiBi.")
        if logits_soft_cap not in (None, 0):
            raise NotImplementedError("B12X_ATTN does not support logits soft cap.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "B12X_ATTN currently supports decoder self-attention only."
            )
        if is_quantized_kv_cache(kv_cache_dtype) and not _is_b12x_fp8_kv_cache(
            kv_cache_dtype
        ):
            raise NotImplementedError(
                "B12X_ATTN currently supports only fp8/fp8_e4m3 quantized "
                "KV cache dtypes."
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError("B12X_ATTN requires q heads divisible by kv heads.")

        expected_scale = head_size**-0.5
        if not math.isclose(float(scale), expected_scale, rel_tol=1e-5, abs_tol=1e-7):
            raise NotImplementedError(
                "B12X_ATTN currently requires canonical softmax scale "
                f"head_dim**-0.5={expected_scale}, got {scale}."
            )
        if self.total_cp_world_size > 1:
            raise NotImplementedError(
                "B12X_ATTN does not yet support decode/prefill context parallelism."
            )

        self.num_heads = int(num_heads)
        self.head_size = int(head_size)
        class_head_size_v = type(self).head_size_v
        self.output_head_size = int(
            self.head_size if class_head_size_v is None else class_head_size_v
        )
        self._uses_packed_kv_cache = type(self).head_size_v is not None
        self.scale = float(scale)
        self.num_kv_heads = int(num_kv_heads)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.window_left = -1 if sliding_window is None else int(sliding_window) - 1

        self.sinks = sinks
        if self.sinks is not None and (
            self.sinks.ndim != 1 or int(self.sinks.shape[0]) != self.num_heads
        ):
            raise ValueError(
                "B12X_ATTN sinks must have shape "
                f"[{self.num_heads}], got {tuple(self.sinks.shape)}."
            )
        self._sinks_cache: dict[tuple[Any, ...], torch.Tensor] = {}

        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        spec_config = vllm_config.speculative_config
        default_block_size = int(cache_config.block_size)
        if default_block_size not in _B12X_SUPPORTED_PAGE_SIZES:
            raise ValueError(
                "B12X_ATTN requires --block-size in "
                f"{_B12X_SUPPORTED_PAGE_SIZES}, got "
                f"{cache_config.block_size}."
            )

        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.dtype = torch.get_default_dtype()
        if self.dtype not in (torch.float16, torch.bfloat16):
            self.dtype = model_config.dtype
        self.kv_torch_dtype = _dtype_from_cache_config(kv_cache_dtype, vllm_config)
        max_batched = int(scheduler_config.max_num_batched_tokens)
        max_num_seqs = int(scheduler_config.max_num_seqs)
        max_model_len = int(model_config.max_model_len)
        self._max_num_seqs = max_num_seqs
        max_page_table_widths = {
            page_size: _max_page_table_width(
                max_model_len,
                page_size,
                max_batched,
                cache_config.mamba_cache_mode,
            )
            for page_size in _B12X_SUPPORTED_PAGE_SIZES
        }

        # Extend dispatch may depend on the static Q tensor capacity, but never
        # on live per-request lengths. Keep a small set of capacity buckets so
        # short/tail prefills do not replay the maximum 8K CTA grid.
        self._extend_q_capacities = tuple(
            sorted(
                {
                    min(max_batched, q_capacity)
                    for q_capacity in (128, 512, 1024, 2048, 4096, max_batched)
                    if q_capacity > 0
                }
            )
        )

        def _extend_work_items(
            page_size: int,
            q_capacity: int,
            batch_size: int,
        ) -> int:
            capacity = plan_extend_graph_capacity(
                device=self.device,
                q_dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.output_head_size,
                page_size=page_size,
                batch=batch_size,
                total_q_capacity=q_capacity,
                max_cache_page_count=max_page_table_widths[page_size],
                window_left=self.window_left,
            )
            return _env_int(
                "VLLM_B12X_PAGED_EXTEND_MAX_WORK_ITEMS",
                capacity.max_work_items,
            )

        _disable_cutlass_memory_debug_snapshot_if_off()

        from b12x.attention.paged import (
            Caps as B12XPagedAttentionScratchCaps,
        )
        from b12x.attention.paged import (
            compile as compile_paged_attention,
        )
        from b12x.attention.paged import (
            decode_graph_capacity as plan_decode_graph_capacity,
        )
        from b12x.attention.paged import (
            decode_graph_scratch_envelope as plan_decode_graph_scratch_envelope,
        )
        from b12x.attention.paged import (
            extend_graph_capacity as plan_extend_graph_capacity,
        )
        from b12x.attention.paged import (
            plan as plan_paged_attention_scratch,
        )
        from b12x.attention.paged import (
            run as paged_attention_forward,
        )
        from b12x.attention.paged import (
            verify_graph_capacity as plan_verify_graph_capacity,
        )

        self._compile_paged_attention = compile_paged_attention
        self._paged_attention_forward = paged_attention_forward

        def _make_plan(
            page_size: int,
            mode: str,
            max_total_q: int,
            max_batch: int,
            max_work_items: int,
            max_partial_rows: int,
            use_cuda_graph: bool,
            num_cache_pages: int,
            copy_runtime_metadata: bool,
        ) -> Any:
            return plan_paged_attention_scratch(
                B12XPagedAttentionScratchCaps(
                    device=self.device,
                    mode=mode,
                    dtype=self.dtype,
                    kv_dtype=self.kv_torch_dtype,
                    num_q_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    head_dim_qk=self.head_size,
                    head_dim_vo=self.output_head_size,
                    page_size=page_size,
                    max_total_q=max_total_q,
                    max_batch=max_batch,
                    max_page_table_width=max_page_table_widths[page_size],
                    max_work_items=max_work_items,
                    max_partial_rows=max_partial_rows,
                    # Shape-only planning tensor; runtime cache shape is
                    # validated by head/page geometry, not page count.
                    num_cache_pages=num_cache_pages,
                    use_cuda_graph=use_cuda_graph,
                    copy_runtime_metadata=copy_runtime_metadata,
                )
            )

        capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
        decode_plan_sizes = {
            int(size) for size in capture_sizes if 0 < int(size) <= max_num_seqs
        }
        decode_plan_sizes.add(max_num_seqs)
        if os.getenv("VLLM_B12X_PAGED_DECODE_MAX_CHUNKS_PER_REQ"):
            logger.warning_once(
                "VLLM_B12X_PAGED_DECODE_MAX_CHUNKS_PER_REQ is ignored; "
                "b12x owns decode graph chunk policy. Use the fixed "
                "work/partial capacity controls only to constrain storage."
            )
        decode_work_items_limit = _env_optional_storage_limit(
            "VLLM_B12X_PAGED_DECODE_MAX_WORK_ITEMS",
            allow_zero=False,
        )
        decode_partial_rows_limit = _env_optional_storage_limit(
            "VLLM_B12X_PAGED_DECODE_MAX_PARTIAL_ROWS",
            allow_zero=True,
        )

        def _create_decode_plan(page_size: int, batch_size: int) -> Any:
            max_page_table_width = max_page_table_widths[page_size]
            capacity = plan_decode_graph_capacity(
                device=self.device,
                q_dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.output_head_size,
                page_size=page_size,
                batch=batch_size,
                max_cache_page_count=max_page_table_width,
                window_left=self.window_left,
                max_work_items=decode_work_items_limit,
                max_partial_rows=decode_partial_rows_limit,
            )
            plan = _make_plan(
                page_size,
                "decode",
                batch_size,
                batch_size,
                capacity.max_work_items,
                capacity.max_partial_rows,
                True,
                max_page_table_width,
                True,
            )
            plan.prepare_decode_graph_replay_state(
                batch=batch_size,
                total_q_capacity=batch_size,
                max_page_table_width=max_page_table_width,
                max_cache_page_count=max_page_table_width,
                window_left=self.window_left,
            )
            return plan

        self._create_decode_plan = _create_decode_plan
        self._verify_q_per_req = 0
        if spec_config is not None:
            self._verify_q_per_req = 1 + int(
                getattr(spec_config, "num_speculative_tokens", None) or 0
            )
        if self._verify_q_per_req <= 1:
            self._verify_q_per_req = 0

        def _create_verify_plan(page_size: int, batch_size: int) -> Any:
            if self._verify_q_per_req <= 1:
                raise RuntimeError(
                    "B12X_ATTN verifier plan requested without speculation"
                )
            max_page_table_width = max_page_table_widths[page_size]
            total_q = batch_size * self._verify_q_per_req
            capacity = plan_verify_graph_capacity(
                device=self.device,
                q_dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.output_head_size,
                page_size=page_size,
                batch=batch_size,
                query_len=self._verify_q_per_req,
                max_cache_page_count=max_page_table_width,
                window_left=self.window_left,
            )
            plan = _make_plan(
                page_size,
                "verify",
                total_q,
                batch_size,
                capacity.max_work_items,
                capacity.max_partial_rows,
                True,
                max_page_table_width,
                True,
            )
            page_ids = torch.arange(
                max_page_table_width,
                dtype=torch.int32,
                device=self.device,
            )
            max_page_table = page_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
            max_cache_seqlens = torch.full(
                (batch_size,),
                capacity.representative_cache_seqlen,
                dtype=torch.int32,
                device=self.device,
            )
            max_cu_seqlens_q = torch.arange(
                0,
                total_q + 1,
                self._verify_q_per_req,
                dtype=torch.int32,
                device=self.device,
            )
            plan.prepare_graph_replay_state(
                page_table=max_page_table,
                cache_seqlens=max_cache_seqlens,
                cu_seqlens_q=max_cu_seqlens_q,
                active_total_q=total_q,
                window_left=self.window_left,
            )
            return plan

        self._create_verify_plan = _create_verify_plan

        def _create_extend_plan(
            page_size: int,
            batch_size: int,
            q_capacity: int,
        ) -> Any:
            """Prepare a fixed-capacity extend plan without reading live lengths."""
            max_page_table_width = max_page_table_widths[page_size]
            plan = _make_plan(
                page_size,
                "extend",
                q_capacity,
                batch_size,
                _extend_work_items(page_size, q_capacity, batch_size),
                0,
                True,
                max_page_table_width,
                False,
            )
            page_ids = torch.arange(
                max_page_table_width,
                dtype=torch.int32,
                device=self.device,
            )
            max_page_table = page_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
            max_cache_seqlens = torch.full(
                (batch_size,),
                min(max_model_len, max_page_table_width * page_size),
                dtype=torch.int32,
                device=self.device,
            )
            # Put one row in every request except the last, which owns the
            # remainder. This represents the full total-Q capacity while the
            # replay kernel remains responsible for packing arbitrary live
            # per-request lengths from device cu_seqlens_q.
            max_cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=torch.int32,
                device=self.device,
            )
            max_cu_seqlens_q[-1] = q_capacity
            plan.prepare_graph_replay_state(
                page_table=max_page_table,
                cache_seqlens=max_cache_seqlens,
                cu_seqlens_q=max_cu_seqlens_q,
                active_total_q=q_capacity,
                window_left=self.window_left,
            )
            return plan

        self._create_extend_plan = _create_extend_plan
        decode_scratch_envelopes = {
            page_size: plan_decode_graph_scratch_envelope(
                device=self.device,
                q_dtype=self.dtype,
                kv_dtype=self.kv_torch_dtype,
                num_q_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_size,
                head_dim_vo=self.output_head_size,
                page_size=page_size,
                max_batch=max_num_seqs,
                max_page_table_width=max_page_table_widths[page_size],
                max_cache_page_count=max_page_table_widths[page_size],
                window_left=self.window_left,
                max_work_items=decode_work_items_limit,
                max_partial_rows=decode_partial_rows_limit,
                copy_runtime_metadata=True,
            )
            for page_size in _B12X_SUPPORTED_PAGE_SIZES
        }
        self._decode_plans: dict[tuple[int, int], Any] = {}
        self._verify_plans: dict[tuple[int, int], Any] = {}
        self._extend_plans: dict[tuple[int, int, int], Any] = {}
        for page_size in _B12X_SUPPORTED_PAGE_SIZES:
            for batch_size in sorted(decode_plan_sizes):
                self._decode_plans[page_size, batch_size] = self._create_decode_plan(
                    page_size, batch_size
                )
            if self._verify_q_per_req > 1:
                for batch_size in range(1, max_num_seqs + 1):
                    self._verify_plans[page_size, batch_size] = (
                        self._create_verify_plan(page_size, batch_size)
                    )
            for batch_size in range(1, max_num_seqs + 1):
                for q_capacity in self._extend_q_capacities:
                    self._extend_plans[page_size, batch_size, q_capacity] = (
                        self._create_extend_plan(
                            page_size,
                            batch_size,
                            q_capacity,
                        )
                    )
        self._scratch_nbytes = max(
            *(int(envelope.nbytes) for envelope in decode_scratch_envelopes.values()),
            *(int(plan.layout.nbytes) for plan in self._verify_plans.values()),
            *(int(plan.layout.nbytes) for plan in self._extend_plans.values()),
        )

        current_workspace_manager().get_simultaneous(
            ((self._scratch_nbytes,), torch.uint8),
        )

        self._contig_noncausal_enabled = bool(
            _uses_b12x_dflash_attention(spec_config)
            and self.window_left != -1
            and self.output_head_size == self.head_size
        )
        self._contig_q_per_req = 1
        self._contig_max_batch = 1
        self._contig_max_q_rows = 1
        self._contig_max_kv_window = 1
        self._contig_max_kv_rows = 1
        self._contig_scratch_nbytes = 1
        self._contig_scratch_plan: Any | None = None
        self._contig_attention_forward: Any | None = None
        self._contig_window_size: tuple[int, int] | None = None
        if self._contig_noncausal_enabled:
            self._contig_q_per_req = 1 + int(
                getattr(spec_config, "num_speculative_tokens", None) or 0
            )
            self._contig_max_batch = max_num_seqs
            self._contig_max_q_rows = max(
                self._contig_max_batch * self._contig_q_per_req,
                1,
            )
            max_seq_cap = max_model_len + self._contig_q_per_req
            self._contig_max_kv_window = max(
                min(max_seq_cap, self.window_left + self._contig_q_per_req),
                1,
            )
            self._contig_max_kv_rows = max(
                self._contig_max_batch * self._contig_max_kv_window,
                1,
            )
            (
                q_buf,
                k_buf,
                v_buf,
                cu_q,
                cu_k,
            ) = current_workspace_manager().get_simultaneous(
                *self._contig_workspace_specs(include_scratch=False)
            )
            from b12x.attention.varlen import (
                create_plan as create_varlen_attention_plan,
            )
            from b12x.attention.varlen import (
                plan as plan_varlen_attention_scratch,
            )
            from b12x.attention.varlen import (
                run as b12x_varlen_attention_forward,
            )

            plan_sinks = (
                torch.empty((self.num_heads,), dtype=torch.float32, device=self.device)
                if self.sinks is not None
                else None
            )
            self._contig_window_size = (self.window_left, self.window_left)
            contig_plan = create_varlen_attention_plan(
                q_buf,
                k_buf,
                v_buf,
                cu_q,
                cu_k,
                max_seqlen_q=self._contig_q_per_req,
                max_seqlen_k=self._contig_max_kv_window,
                causal=False,
                window_size=self._contig_window_size,
                attention_sink_bias=plan_sinks,
            )
            self._contig_scratch_plan = plan_varlen_attention_scratch(contig_plan)
            self._contig_attention_forward = b12x_varlen_attention_forward
            scratch_spec = self._contig_scratch_plan.scratch_specs()[0]
            self._contig_scratch_nbytes = int(scratch_spec.shape[0])
            current_workspace_manager().get_simultaneous(
                *self._contig_workspace_specs(include_scratch=True)
            )

        self.supports_quant_query_input = False
        register_cutedsl_warmup_provider(self)

        logger.info_once(
            "Using B12X_ATTN with q_heads=%d kv_heads=%d head_dim_qk=%d "
            "head_dim_vo=%d window_left=%d planned_page_sizes=%s "
            "verify_q_per_req=%d extend_q_capacities=%s scratch=%d bytes.",
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.output_head_size,
            self.window_left,
            _B12X_SUPPORTED_PAGE_SIZES,
            self._verify_q_per_req,
            self._extend_q_capacities,
            self._scratch_nbytes,
        )

    def _compile_paged_extend_entry(self, page_size: int) -> None:
        """Compile fixed-capacity paged-prefill entries without a live plan."""
        q_rows = 64
        q = torch.zeros(
            (q_rows, self.num_heads, self.head_size),
            dtype=self.dtype,
            device=self.device,
        )
        output = torch.zeros(
            (q_rows, self.num_heads, self.output_head_size),
            dtype=self.dtype,
            device=self.device,
        )
        if self._uses_packed_kv_cache:
            kv_cache = torch.zeros(
                (
                    1,
                    page_size,
                    self.num_kv_heads,
                    self.head_size + self.output_head_size,
                ),
                dtype=self.kv_torch_dtype,
                device=self.device,
            )
        else:
            kv_cache = torch.zeros(
                (
                    1,
                    2,
                    page_size,
                    self.num_kv_heads,
                    self.head_size,
                ),
                dtype=self.kv_torch_dtype,
                device=self.device,
            )
        key_cache, value_cache = self._kv_cache_views(kv_cache)
        sinks = self._prepare_sinks(self.sinks, self.device)
        (scratch_storage,) = current_workspace_manager().get_simultaneous(
            ((self._scratch_nbytes,), torch.uint8),
        )
        for batch_size in range(1, self._max_num_seqs + 1):
            plan = self._extend_plans[
                page_size,
                batch_size,
                self._extend_q_capacities[0],
            ]
            page_table = torch.zeros(
                (batch_size, plan.caps.max_page_table_width),
                dtype=torch.int32,
                device=self.device,
            )
            cache_seqlens = torch.full(
                (batch_size,), page_size, dtype=torch.int32, device=self.device
            )
            cu_seqlens_q = torch.arange(
                0,
                batch_size + 1,
                dtype=torch.int32,
                device=self.device,
            )
            cu_seqlens_q[-1] = q_rows
            k_descale = None
            v_descale = None
            if _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
                k_descale = torch.ones(
                    (batch_size,), dtype=torch.float32, device=self.device
                )
                v_descale = torch.ones(
                    (batch_size,), dtype=torch.float32, device=self.device
                )
            binding = plan.bind(
                scratch=scratch_storage,
                q=q,
                k_cache=key_cache,
                v_cache=value_cache,
                output=output,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                window_left=self.window_left,
                attention_sink_bias=sinks,
                k_descale=k_descale,
                v_descale=v_descale,
            )
            self._compile_paged_attention(binding=binding)
            # Compile-only warmup does not launch the device-side compact
            # scheduler. Execute the smallest fixed-capacity plan once for
            # every graph batch so both the shared dynamic-worklist CuTe entry
            # and each capture-static Triton batch variant are ready before
            # the inference JIT monitor is enabled.
            self._paged_attention_forward(binding=binding)

    def get_cutedsl_warmup_compile_units(self) -> tuple[CuTeDSLCompileUnit, ...]:
        common_key = (
            "b12x_paged_extend",
            str(self.device),
            str(self.dtype),
            str(self.kv_torch_dtype),
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            self.output_head_size,
            self.window_left,
            self._uses_packed_kv_cache,
            self.sinks is not None,
        )
        return tuple(
            CuTeDSLCompileUnit(
                name="b12x_paged_extend",
                key=(*common_key, page_size),
                compile=partial(self._compile_paged_extend_entry, page_size),
            )
            for page_size in _B12X_SUPPORTED_PAGE_SIZES
        )

    def _contig_workspace_specs(
        self,
        *,
        include_scratch: bool,
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            (
                (
                    self._contig_max_q_rows,
                    self.num_heads,
                    self.head_size,
                ),
                self.dtype,
            ),
            (
                (
                    self._contig_max_kv_rows,
                    self.num_kv_heads,
                    self.head_size,
                ),
                self.dtype,
            ),
            (
                (
                    self._contig_max_kv_rows,
                    self.num_kv_heads,
                    self.head_size,
                ),
                self.dtype,
            ),
            ((self._contig_max_batch + 1,), torch.int32),
            ((self._contig_max_batch + 1,), torch.int32),
        ]
        if include_scratch:
            specs.append(((self._contig_scratch_nbytes,), torch.uint8))
        return tuple(specs)

    def _prepare_sinks(
        self,
        sinks: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if sinks is None:
            return None
        if sinks.device != device:
            raise RuntimeError(
                "B12X_ATTN sinks must be on the same CUDA device as query."
            )
        if sinks.dtype == torch.float32 and sinks.is_contiguous():
            return sinks
        key = (
            int(sinks.data_ptr()),
            tuple(sinks.shape),
            tuple(sinks.stride()),
            str(sinks.dtype),
            str(sinks.device),
        )
        cached = self._sinks_cache.get(key)
        if cached is not None:
            return cached
        if _capture_alloc_forbidden():
            raise RuntimeError(
                "B12X_ATTN would convert attention sinks during CUDA graph "
                "capture. Warm the layer eagerly or store sinks as contiguous "
                "float32."
            )
        cached = sinks.to(dtype=torch.float32, device=device).contiguous()
        self._sinks_cache[key] = cached
        return cached

    def _prepare_fp8_descales(
        self,
        layer: AttentionLayer,
        num_reqs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            return None, None
        if num_reqs <= 0:
            raise ValueError("B12X_ATTN fp8 KV descale request count must be positive.")

        def _prepare(scale: torch.Tensor, name: str) -> torch.Tensor:
            if scale.device != device:
                raise RuntimeError(f"B12X_ATTN {name} must be on the query device.")
            if scale.dtype != torch.float32:
                raise RuntimeError(f"B12X_ATTN {name} must be float32.")
            if scale.ndim == 0:
                return scale.expand(num_reqs)
            if scale.ndim == 1:
                if int(scale.shape[0]) == 1:
                    return scale.expand(num_reqs)
                if int(scale.shape[0]) >= num_reqs:
                    return scale[:num_reqs]
            raise ValueError(
                f"B12X_ATTN {name} must be scalar or rank-1 with at least "
                f"{num_reqs} values; got shape {tuple(scale.shape)}."
            )

        return _prepare(layer._k_scale, "k_scale"), _prepare(layer._v_scale, "v_scale")

    def _prepare_contig_fp8_scales(
        self,
        layer: AttentionLayer,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int]:
        if not _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            return None, None, 0, 0

        def _prepare(scale: torch.Tensor, name: str) -> torch.Tensor:
            if scale.device != device:
                raise RuntimeError(f"B12X_ATTN {name} must be on the query device.")
            if scale.dtype != torch.float32:
                raise RuntimeError(f"B12X_ATTN {name} must be float32.")
            if not scale.is_contiguous():
                if _capture_alloc_forbidden():
                    raise RuntimeError(
                        f"B12X_ATTN would make {name} contiguous during CUDA "
                        "graph capture. Store FP8 descales contiguously."
                    )
                scale = scale.contiguous()
            flat = scale.reshape(-1)
            scale_count = int(flat.numel())
            if scale_count not in (1, self.num_kv_heads):
                raise ValueError(
                    f"B12X_ATTN DFlash contiguous {name} must be scalar or "
                    f"per-KV-head, got shape {tuple(scale.shape)}."
                )
            return flat

        k_scale = _prepare(layer._k_scale, "k_scale")
        v_scale = _prepare(layer._v_scale, "v_scale")
        return k_scale, v_scale, int(k_scale.numel()), int(v_scale.numel())

    def _kv_cache_views(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._uses_packed_kv_cache:
            key_cache = kv_cache[..., : self.head_size]
            value_cache = kv_cache[..., self.head_size :]
        else:
            key_cache, value_cache = kv_cache.unbind(1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if _is_b12x_fp8_kv_cache(self.kv_cache_dtype):
            fp8_dtype = current_platform.fp8_dtype()
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(fp8_dtype)
            if value_cache.dtype == torch.uint8:
                value_cache = value_cache.view(fp8_dtype)
        if (
            key_cache.dtype != self.kv_torch_dtype
            or value_cache.dtype != self.kv_torch_dtype
        ):
            raise TypeError(
                f"B12X_ATTN plan expects KV dtype {self.kv_torch_dtype}, got "
                f"{key_cache.dtype}/{value_cache.dtype}."
            )
        return key_cache, value_cache

    def _select_plan(
        self,
        attn_metadata: B12XPagedMetadata,
        total_q: int,
        q_capacity: int,
        num_reqs: int,
        page_size: int,
    ) -> Any:
        if attn_metadata.max_query_len <= 1 and int(total_q) == int(num_reqs):
            batch_size = int(total_q)
            plan_key = (page_size, batch_size)
            plan = self._decode_plans.get(plan_key)
            if plan is None:
                if _capture_alloc_forbidden():
                    raise RuntimeError(
                        "B12X_ATTN decode plan was not prepared before CUDA graph "
                        f"capture for page size {page_size}, batch size "
                        f"{batch_size}."
                    )
                plan = self._create_decode_plan(page_size, batch_size)
                if int(plan.layout.nbytes) > self._scratch_nbytes:
                    raise RuntimeError(
                        "B12X_ATTN lazily created decode plan exceeds reserved "
                        f"scratch: {int(plan.layout.nbytes)} > "
                        f"{self._scratch_nbytes} bytes."
                    )
                self._decode_plans[plan_key] = plan
            return plan
        elif (
            self._verify_q_per_req > 1
            and attn_metadata.max_query_len == self._verify_q_per_req
            and int(total_q) == int(num_reqs) * self._verify_q_per_req
        ):
            plan_key = (page_size, int(num_reqs))
            plan = self._verify_plans.get(plan_key)
            if plan is None:
                if _capture_alloc_forbidden():
                    raise RuntimeError(
                        "B12X_ATTN verifier plan was not prepared before CUDA "
                        f"graph capture for page size {page_size}, batch size "
                        f"{num_reqs}."
                    )
                plan = self._create_verify_plan(page_size, int(num_reqs))
                if int(plan.layout.nbytes) > self._scratch_nbytes:
                    raise RuntimeError(
                        "B12X_ATTN lazily created verifier plan exceeds reserved "
                        f"scratch: {int(plan.layout.nbytes)} > "
                        f"{self._scratch_nbytes} bytes."
                    )
                self._verify_plans[plan_key] = plan
            return plan
        extend_q_capacity = next(
            (
                capacity
                for capacity in self._extend_q_capacities
                if q_capacity <= capacity
            ),
            None,
        )
        if extend_q_capacity is None:
            raise ValueError(
                f"B12X_ATTN extend Q capacity {q_capacity} exceeds prepared "
                f"maximum {self._extend_q_capacities[-1]}."
            )
        return self._extend_plans[
            page_size,
            int(num_reqs),
            extend_q_capacity,
        ]

    def _forward_noncausal_contiguous(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: B12XPagedMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if not self._contig_noncausal_enabled:
            raise NotImplementedError(
                "B12X_ATTN non-causal attention currently requires DFlash "
                "sliding-window layers so the active KV suffix is bounded."
            )
        if (
            self._contig_scratch_plan is None
            or self._contig_attention_forward is None
            or self._contig_window_size is None
        ):
            raise RuntimeError("B12X_ATTN contiguous non-causal plan is not ready.")

        num_actual_tokens = min(
            int(attn_metadata.num_actual_tokens),
            int(query.shape[0]),
            int(output.shape[0]),
        )
        num_reqs = int(attn_metadata.seq_lens.shape[0])
        if num_reqs > self._contig_max_batch:
            raise ValueError(
                f"B12X_ATTN DFlash batch {num_reqs} exceeds contiguous "
                f"capacity {self._contig_max_batch}."
            )
        if num_actual_tokens > self._contig_max_q_rows:
            raise ValueError(
                f"B12X_ATTN DFlash query tokens {num_actual_tokens} exceed "
                f"contiguous capacity {self._contig_max_q_rows}."
            )
        if int(attn_metadata.max_query_len) > self._contig_q_per_req:
            raise ValueError(
                f"B12X_ATTN DFlash max query length {attn_metadata.max_query_len} "
                f"exceeds contiguous capacity {self._contig_q_per_req}."
            )
        if num_actual_tokens <= 0:
            return output

        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]
        if q.dtype != self.dtype or out.dtype != self.dtype:
            raise TypeError(
                f"B12X_ATTN contiguous plan expects dtype {self.dtype}, got "
                f"q={q.dtype}, output={out.dtype}."
            )

        page_table = _ensure_i32_contiguous(attn_metadata.block_table, "block_table")
        cache_seqlens = _ensure_i32_contiguous(attn_metadata.seq_lens, "seq_lens")
        cu_seqlens_q_src = _ensure_i32_contiguous(
            attn_metadata.query_start_loc,
            "query_start_loc",
        )
        sinks = self._prepare_sinks(self.sinks, q.device)
        k_scale, v_scale, k_scale_numel, v_scale_numel = (
            self._prepare_contig_fp8_scales(layer, q.device)
        )
        if k_scale is None:
            k_scale = q
        if v_scale is None:
            v_scale = q
        page_size = _kv_page_size(key_cache, value_cache)

        q_buf, k_buf, v_buf, cu_q, cu_k, scratch = (
            current_workspace_manager().get_simultaneous(
                *self._contig_workspace_specs(include_scratch=True)
            )
        )
        q_buf[:num_actual_tokens].copy_(q)

        _b12x_noncausal_cu_lens_kernel[(1,)](
            cache_seqlens,
            cu_seqlens_q_src,
            cu_q,
            cu_k,
            num_reqs,
            MAX_BATCH=self._contig_max_batch,
            KV_WINDOW=self._contig_max_kv_window,
        )

        block_t = 16
        block_d = 32
        _b12x_gather_paged_kv_to_contiguous_kernel[
            (
                self._contig_max_batch,
                cdiv(self._contig_max_kv_window, block_t),
                self.num_kv_heads * cdiv(self.head_size, block_d),
            )
        ](
            key_cache,
            value_cache,
            k_buf,
            v_buf,
            page_table,
            cache_seqlens,
            cu_k,
            k_scale,
            v_scale,
            num_reqs,
            BLOCK_TABLE_STRIDE=page_table.stride(0),
            K_CACHE_STRIDE_0=key_cache.stride(0),
            K_CACHE_STRIDE_1=key_cache.stride(1),
            K_CACHE_STRIDE_2=key_cache.stride(2),
            K_CACHE_STRIDE_3=key_cache.stride(3),
            V_CACHE_STRIDE_0=value_cache.stride(0),
            V_CACHE_STRIDE_1=value_cache.stride(1),
            V_CACHE_STRIDE_2=value_cache.stride(2),
            V_CACHE_STRIDE_3=value_cache.stride(3),
            PAGE_SIZE=page_size,
            NUM_KV_HEADS=self.num_kv_heads,
            HEAD_DIM=self.head_size,
            KV_WINDOW=self._contig_max_kv_window,
            HAS_FP8_KV=_is_b12x_fp8_kv_cache(self.kv_cache_dtype),
            K_SCALE_NUMEL=k_scale_numel,
            V_SCALE_NUMEL=v_scale_numel,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
        )

        binding = self._contig_scratch_plan.bind(
            scratch=scratch,
            q=q_buf,
            k=k_buf,
            v=v_buf,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=self._contig_q_per_req,
            max_seqlen_k=self._contig_max_kv_window,
            softmax_scale=self.scale,
            causal=False,
            window_size=self._contig_window_size,
            attention_sink_bias=sinks,
        )
        contig_out, _ = self._contig_attention_forward(binding=binding)
        out.copy_(contig_out[:num_actual_tokens])
        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: B12XPagedMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "B12X_ATTN does not support fused output quantization."
            )
        if attn_metadata is None:
            return output.fill_(0)
        if output.shape[-1] != self.output_head_size:
            raise ValueError(
                f"B12X_ATTN expected output head dim {self.output_head_size}, got "
                f"{output.shape[-1]}."
            )
        if kv_cache.numel() == 0:
            return output.fill_(0)

        # In FULL cudagraph mode vLLM may pad attention metadata to the graph
        # bucket while still passing per-layer Q/output tensors with only the
        # real rows. Use tensor capacity as the launch contract and avoid
        # selecting decode graph replay for padded virtual requests.
        q_capacity = min(
            int(query.shape[0]),
            int(output.shape[0]),
        )
        num_actual_tokens = min(
            int(attn_metadata.num_actual_tokens),
            q_capacity,
        )
        if num_actual_tokens <= 0:
            return output
        q = query[:num_actual_tokens]
        out = output[:num_actual_tokens]
        if q.dtype != self.dtype or out.dtype != self.dtype:
            raise TypeError(
                f"B12X_ATTN plan expects dtype {self.dtype}, got "
                f"q={q.dtype}, output={out.dtype}."
            )

        key_cache, value_cache = self._kv_cache_views(kv_cache)
        page_size = _kv_page_size(key_cache, value_cache)
        if not attn_metadata.causal:
            return self._forward_noncausal_contiguous(
                layer,
                query,
                key_cache,
                value_cache,
                attn_metadata,
                output,
            )

        page_table = _ensure_i32_contiguous(attn_metadata.block_table, "block_table")
        cache_seqlens = _ensure_i32_contiguous(attn_metadata.seq_lens, "seq_lens")
        cu_seqlens_q = _ensure_i32_contiguous(
            attn_metadata.query_start_loc,
            "query_start_loc",
        )
        num_reqs = int(cache_seqlens.shape[0])
        if attn_metadata.max_query_len <= 1 and num_actual_tokens < num_reqs:
            num_reqs = num_actual_tokens
            page_table = page_table[:num_reqs]
            cache_seqlens = cache_seqlens[:num_reqs]
            cu_seqlens_q = cu_seqlens_q[: num_reqs + 1]
        sinks = self._prepare_sinks(self.sinks, q.device)
        k_descale, v_descale = self._prepare_fp8_descales(
            layer,
            num_reqs,
            q.device,
        )
        plan = self._select_plan(
            attn_metadata,
            num_actual_tokens,
            q_capacity,
            num_reqs,
            page_size,
        )
        (scratch_storage,) = current_workspace_manager().get_simultaneous(
            ((self._scratch_nbytes,), torch.uint8),
        )
        binding = plan.bind(
            scratch=scratch_storage,
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            output=out,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            window_left=self.window_left,
            active_total_q=(None if plan.caps.mode == "extend" else num_actual_tokens),
            attention_sink_bias=sinks,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        self._paged_attention_forward(binding=binding)
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        if self._uses_packed_kv_cache:
            triton_reshape_and_cache_flash_diffkv(
                key,
                value,
                kv_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            return
        from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash

        key_cache, value_cache = kv_cache.unbind(1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )


_B12X_PAGED_BACKEND_BY_V_HEAD: dict[int, type[B12XPagedAttentionBackend]] = {}


def get_b12x_paged_attention_backend(
    head_size_v: int,
) -> type[B12XPagedAttentionBackend]:
    """Return a B12X paged backend class with a per-layer VO head dimension."""
    set_kv_cache_layout("NHD")
    head_size_v = int(head_size_v)
    if head_size_v <= 0 or head_size_v % 16 != 0:
        raise ValueError(
            "B12X_ATTN requires head_size_v to be a positive multiple of 16; "
            f"got {head_size_v}."
        )
    if head_size_v not in B12XPagedAttentionBackend.get_supported_head_sizes():
        logger.warning_once(
            "B12X_ATTN head_size_v=%d is outside the advertised head sizes %s; "
            "b12x will validate the exact paged kernel traits at runtime.",
            head_size_v,
            tuple(B12XPagedAttentionBackend.get_supported_head_sizes()),
        )

    backend = _B12X_PAGED_BACKEND_BY_V_HEAD.get(head_size_v)
    if backend is not None:
        return backend

    impl_name = f"B12XPagedAttentionImplV{head_size_v}"
    backend_name = f"B12XPagedAttentionBackendV{head_size_v}"
    impl_cls = type(
        impl_name,
        (B12XPagedAttentionImpl,),
        {
            "__module__": __name__,
            "__qualname__": impl_name,
            "head_size_v": head_size_v,
        },
    )
    backend_cls = type(
        backend_name,
        (B12XPagedAttentionBackend,),
        {
            "__module__": __name__,
            "__qualname__": backend_name,
            "head_size_v": head_size_v,
            "_impl_cls": impl_cls,
        },
    )
    _B12X_PAGED_BACKEND_BY_V_HEAD[head_size_v] = backend_cls
    return backend_cls
