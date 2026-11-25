# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpts from upstream b409a4f0d787fdddb75ec2e8ddb6dfcfd36235b7 (with preceding series entries).
# vllm/models/qwen4_exp/common/qsa_indexer.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged indexer cache ownership and metadata for Qwen4Exp QSA.

Each QSA layer keeps a fixed circular buffer of raw index keys (the
compressor state) and one compressed key. MRoPE models pack exact three-axis
positions beside the raw keys; text models derive group positions from
logical positions. The compressor state uses one block per request, while
    the compressed owner uses `MLAAttentionSpec.tokens_per_state` so its block
table follows the main KV-cache lifecycle. Their physical tensor storage is
shared by the generic cache-layout planner.
"""

import math
from dataclasses import dataclass
from functools import cache
from typing import ClassVar

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID,
    split_decodes_and_prefills,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayerConfig,
    KVCacheLayout,
    MLAAttentionSpec,
)


def canonical_qsa_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return exact per-token positions as `[tokens, 1, 3]` int64 rows."""
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(3, -1)
    elif positions.ndim != 2 or positions.shape[0] not in (1, 3):
        raise ValueError("QSA RoPE positions must be [tokens] or [1|3, tokens]")
    if positions.shape[0] == 1:
        positions = positions.expand(3, -1)
    return positions.transpose(0, 1).unsqueeze(1).to(torch.int64)

class _QSAIndexerCache:
    def get_attn_backend(self) -> type[AttentionBackend]:
        return QSAIndexerBackend


class QSARawKeyCache(_QSAIndexerCache):
    """Raw BF16 key, optionally followed by exact int64 MRoPE positions."""

    _BF16_PER_INT64 = 4
    _NUM_ROPE_AXES = 3

    def __init__(self, *, cache_rope_positions: bool = False, **kwargs) -> None:
        key_head_size = int(kwargs.pop("head_size"))
        self.key_head_size = key_head_size
        self.cache_rope_positions = bool(cache_rope_positions)
        self.rope_position_offset = round_up(key_head_size, self._BF16_PER_INT64)
        storage_head_size = key_head_size
        if self.cache_rope_positions:
            storage_head_size = self.rope_position_offset + (
                self._NUM_ROPE_AXES * self._BF16_PER_INT64
            )
        super().__init__(head_size=storage_head_size, **kwargs)

    # Derived on access so `kv_cache` stays the only reference to the bound
    # storage: clearing it (e.g. after CUDA graph memory profiling) must free
    # the cache, or the freed block stays pinned in the allocator.
    @property
    def key_cache(self) -> torch.Tensor:
        return self.kv_cache[..., : self.key_head_size]

    @property
    def rope_position_cache(self) -> torch.Tensor | None:
        if not self.cache_rope_positions:
            return None
        return self.kv_cache[..., self.rope_position_offset :].view(torch.int64)

    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:
        # Hold the open group's committed keys plus every row a speculative
        # step stores before acceptance is known, rounded up to whole groups so
        # the ring divides the attention block size (it joins the LCM that sets
        # the scheduler block size). Anything narrower lets a rejected draft row
        # overwrite a committed key the next step needs to close the group.
        span = self.compress_ratio + vllm_config.num_speculative_tokens
        capacity = round_up(span, self.compress_ratio)
        assert self.cache_config.block_size % capacity == 0, (
            f"QSA ring capacity {capacity} must divide the attention block "
            f"size {self.cache_config.block_size}"
        )
        spec = AttentionSpec(
            block_size=capacity,
            num_kv_heads=1,
            head_size=self.head_size,
            head_size_v=0,
            dtype=self.dtype,
        )
        return CircularBufferManager.layer_config(spec)
