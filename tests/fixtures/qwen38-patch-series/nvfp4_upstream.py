# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpts from upstream b409a4f0d787fdddb75ec2e8ddb6dfcfd36235b7 (with preceding series entries).
# tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py

import pytest

import torch

import torch.nn.functional as F

from nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
)

from vllm import _custom_ops as ops

from vllm.model_executor.kernels.linear.nvfp4 import NvFp4LinearLayerConfig

from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCuteDslNvFp4W4A16LinearKernel,
)

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)

from vllm.platforms import current_platform

from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer_b12x_gemm,
)

from vllm.utils.torch_utils import set_random_seed

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_flashinfer_cutedsl_nvfp4_w4a16_linear(
    shape: tuple[int, int, int],
    seed: int,
    device: str,
) -> None:
    supported, reason = FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported()
    if not supported:
        pytest.skip(reason)

    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    weight_quant_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / weight.abs().max()).to(
        torch.float32
    )
    weight_global_scale = weight_quant_scale.reciprocal()
    weight_fp4, weight_scale = ops.scaled_fp4_quant(
        weight,
        weight_quant_scale,
        is_sf_swizzled_layout=False,
    )
    weight_ref = dequantize_to_dtype(
        weight_fp4,
        weight_scale,
        weight_global_scale,
        dtype=torch.bfloat16,
        swizzle=False,
    )

    layer = torch.nn.Module()
    layer.output_size_per_partition = n
    layer.weight = torch.nn.Parameter(weight_fp4, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        weight_global_scale, requires_grad=False
    )
    kernel = FlashInferCuteDslNvFp4W4A16LinearKernel(NvFp4LinearLayerConfig())

    kernel.process_weights_after_loading(layer)
    output = kernel.apply_weights(layer, x)

    expected = F.linear(x, weight_ref)
    assert output.shape == (m, n)
    torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-1)
