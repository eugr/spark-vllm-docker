# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpt from upstream c4f2bb4abbdbad9ed5b0a71acd0bd585056eeab8.
from __future__ import annotations

class KernelConfig:
    linear_backend_per_quant: dict[str, LinearBackend] | None = Field(
        default=None, min_length=1
    )
    """Backend overrides keyed by linear quantization scheme. Overrides take
    precedence over ``linear_backend``; for example,
    ``{"nvfp4_w4a16": "humming"}``."""

    rdna_hybrid_w4a16_prefill_dequant: RDNAHybridW4A16PrefillDequantMode = "disable"
    """Cache a dense dequantized copy of each weight for the RDNA hybrid W4A16
    linear kernel's prefill GEMM. Applies only on gfx11/gfx12 RDNA GPUs, where
    that kernel is selected; a no-op for every other kernel. Prefill GEMMs
    then run on the dense copy instead of dequantizing int4 every step,
    cutting TTFT by 20-25% on gfx1151 for Qwen3 and Gemma-4. Decode stays on
    int4, so TPOT is unchanged. The copy is ~4x the size of the int4 weight
    and comes out of the gpu_memory_utilization budget, leaving less room for
    KV cache, so it suits unified-memory parts where TTFT matters more than KV
    capacity. Available options:

    - "disable": Keep prefill on the int4 path (default)
    - "enable": Cache every layer's copy; startup fails if they do not fit in
      the budget
    """

    @field_validator("moe_backend", mode="before")
    @classmethod
    def _normalize_moe_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    @field_validator("linear_backend", mode="before")
    @classmethod
    def _normalize_linear_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    @field_validator("sparse_indexer_topk_backend", mode="before")
    @classmethod
    def _normalize_sparse_indexer_topk_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value

    @field_validator("rdna_hybrid_w4a16_prefill_dequant", mode="before")
    @classmethod
    def _normalize_rdna_hybrid_w4a16_prefill_dequant(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower().replace("-", "_")
        return value
