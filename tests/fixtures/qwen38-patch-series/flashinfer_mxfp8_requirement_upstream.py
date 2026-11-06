# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Upstream excerpts from b1cc4f35a5e30bdf378ccb5f6c398ce154375774:
# flashinfer/gemm/gemm_base.py

def preceding_requirement():
    return True


@supported_compute_capability([100, 103, 107])
def _cute_dsl_gemm_mxfp8_requirement(
    a: torch.Tensor,  # unused
    b: torch.Tensor,  # unused
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,  # unused
    out_dtype: torch.dtype = torch.bfloat16,  # unused
    use_8x4_sf_layout: bool = True,  # unused
    backend: Literal["cutlass", "cute-dsl", "auto"] = "auto",  # unused
    batch_invariant: bool = False,
):
    # CuTe DSL MXFP8 path currently expects swizzled 1D block scales
    # in F8_128x4 layout for both A and B.
    if a_descale.ndim != 1 or b_descale.ndim != 1:
        raise ValueError(
            "cute_dsl mm_mxfp8 requires swizzled 1D scale tensors for a_descale and b_descale."
        )
    _check_cute_dsl_availability()
    _check_cute_dsl_arch(a.device)
    return True
