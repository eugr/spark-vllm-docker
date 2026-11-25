# Excerpts from flashinfer-ai/flashinfer@45e0842d4a48b5849422e2f55b04af5e8b7a7163.
from __future__ import annotations

def _cute_dsl_mxfp8_runners(
    sm_major: int,
    sm_minor: int,
    enable_pdl: bool,
    out_dtype: torch.dtype,
    inputs: List[torch.Tensor],
) -> List[TunableRunner]:
    """Runners of ``mm_mxfp8(backend="cute-dsl")``; ``[0]`` is the untuned default."""
    from ..cute_dsl.availability import is_cute_dsl_experimental_available

    # Older DSLs lack cutlass.experimental; only the block-scaled runner remains.
    if not is_cute_dsl_experimental_available():
        return [_cute_dsl_gemm_mxfp8_runner(sm_major, sm_minor, enable_pdl, out_dtype)]
    warp = _cute_dsl_warp_splitk_mxfp8_gemm_runner(enable_pdl)
    # One autotune call profiles every M bucket with these runners, so whether the
    # warp runner is registered must not depend on M; per profile it offers
    # tactics only for the M it serves.
    if not warp.supports_inputs(inputs, check_m=False):
        return [_cute_dsl_gemm_mxfp8_runner(sm_major, sm_minor, enable_pdl, out_dtype)]
    cluster_splitk = _cute_dsl_gemm_mxfp8_runner(
        sm_major,
        sm_minor,
        enable_pdl,
        out_dtype,
        warp_splitk_supports=warp.supports_inputs,
    )
    a, b = inputs[0], inputs[1]
    # Untuned, warp split-K runs first only while its 16-wide output tiles fit in
    # one wave. On wider N the cluster split-K default is faster in back-to-back
    # PDL calls, where its prologue overlaps the previous kernel (#6839).
    if warp.supports_inputs(inputs) and b.shape[1] // 16 <= get_device_sm_count(
        a.device
    ):
        return [warp, cluster_splitk]
    return [cluster_splitk, warp]
