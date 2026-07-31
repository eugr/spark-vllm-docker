"""Flash Attention 4 (CuTe DSL) with un-merged SM120 improvements.

This package is a downstream distribution of Dao-AILab/flash-attention
that bundles five open upstream PRs targeting SM120 (consumer Blackwell:
RTX PRO 6000, DGX Spark, RTX 5090):

- #2336 — SM120 split-KV (FlashDecoding) with FP32 partial outputs
- #2348 — SM120 kernel-level paged KV cache support (includes #2336)
- #2349 — SM120 TMA forward kernel with warp specialization
- #2389 — SM80/SM120 block-sparse forward attention support
- #2439 — FA4 dropout (Philox-based, per-element, all arches)

Once these merge upstream, users should prefer the upstream package.
Until then this lets SM120 users access the improvements today.

Contributed by Second Nature Computing (https://joinsecondnature.com).
"""

__version__ = "0.1.0"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
