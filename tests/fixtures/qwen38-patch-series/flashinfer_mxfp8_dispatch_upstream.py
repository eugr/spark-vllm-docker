# Runner dispatch excerpt from flashinfer-ai/flashinfer@431c9687f3bb.
from __future__ import annotations

def _cute_dsl_gemm_mxfp8_runner(
    sm_major: int,
    sm_minor: int,
    enable_pdl: bool,
    out_dtype: torch.dtype,
    warp_splitk_supports: Optional[Callable[[List[torch.Tensor]], bool]] = None,
    low_m_only: bool = False,
):
    """``low_m_only``: only the M <= 32 narrow-tile / split-K tactics (SM107)."""
    import cutlass

    from .kernels.dense_blockscaled_gemm_sm100 import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    # CPU stand-in for the remaining upstream SM10x implementation.
    return low_m_only
