# Excerpts from flashinfer-ai/flashinfer@45e0842d4a48b5849422e2f55b04af5e8b7a7163.
from __future__ import annotations

def select_tuning(backends, major):

    tuner = AutoTuner.get()

    tuning_config = (
        _MM_MXFP8_CUTE_DSL_TUNING_CONFIG
        if "cute-dsl" in backends or backends == ["cutedsl_low_latency"]
        else _MM_MXFP8_TUNING_CONFIG
    )

    runner, tactic = tuner.choose_one(
        custom_op="mxfp8_gemm",
    )
    return tuning_config
