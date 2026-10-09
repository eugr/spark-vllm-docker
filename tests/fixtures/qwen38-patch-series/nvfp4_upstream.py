# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpt from tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py
# at vllm-project/vllm@59ffd73e8597d860730f4a8c4f778f27c75a0dfa.
# Imports and the remaining helpers/tests are omitted; this fixture is parsed,
# never imported, so the patch regression runs without CUDA or vLLM.
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("token_shape", [(17,), (1, 17), (2, 17)])
@pytest.mark.parametrize(
    "kernel_cls",
    [FlashInferCutlassNvFp4LinearKernel, FlashInferCuteDslNvFp4LinearKernel],
)
@torch.inference_mode()
def test_flashinfer_nvfp4_preserves_quantized_input_shape(
    token_shape: tuple[int, ...],
    kernel_cls: type[FlashInferCutlassNvFp4LinearKernel]
    | type[FlashInferCuteDslNvFp4LinearKernel],
) -> None:
    """Pre-quantized inputs must match the tensor path and retain token dimensions."""
    supported, reason = kernel_cls.is_supported()
    if not supported:
        pytest.skip(reason)

    set_random_seed(42)
    hidden_size, output_size = 128, 128
    x = torch.randn(*token_shape, hidden_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(output_size, hidden_size, dtype=x.dtype, device=x.device)
    input_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / x.abs().max()).float()
    weight_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / weight.abs().max()).float()
    x_fp4, x_blockscale = ops.scaled_fp4_quant(x.reshape(-1, hidden_size), input_scale)
    weight_fp4, weight_blockscale = ops.scaled_fp4_quant(weight, weight_scale)
    quantized_input = QuantizedActivation(
        data=x_fp4.reshape(*token_shape, hidden_size // 2),
        scale=x_blockscale,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kNvfp4Dynamic,
    )
    layer = torch.nn.Module()
    layer.input_size_per_partition = hidden_size
    layer.output_size_per_partition = output_size
    layer.weight = weight_fp4
    layer.weight_scale = weight_blockscale
    layer.input_global_scale_inv = input_scale
    layer.alpha = (input_scale * weight_scale).reciprocal()
    kernel = kernel_cls(NvFp4LinearLayerConfig())

    expected = kernel.apply_weights(layer, x.reshape(-1, hidden_size))
    actual = kernel.apply_weights(layer, quantized_input)

    assert actual.shape == (*token_shape, output_size)
    assert actual.dtype == x.dtype
    torch.testing.assert_close(actual.reshape(expected.shape), expected)


def get_ref_results(
    a_fp4,
    b_fp4,
):
    pass
