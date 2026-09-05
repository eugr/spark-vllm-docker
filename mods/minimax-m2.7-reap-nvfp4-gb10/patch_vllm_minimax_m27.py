#!/usr/bin/env python3
"""Patches for serving scottgl/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10 on GB10.

Patches against the ``vllm-node`` image (vllm 0.25.1.dev lineage):

1. Humming kernel crash on ParallelLMHead (FP8 lm_head).
   The compressed-tensors FP8 W8A16 scheme routes to the Humming linear kernel
   for process_weights_after_loading.  Humming's prepare_humming_layer accesses
   ``layer.output_partition_sizes``, an attribute set by ``LinearBase`` but NOT
   by ``VocabParallelEmbedding`` (which ``ParallelLMHead`` extends).  The
   compressed-tensors ``create_weights`` already stores the same data as
   ``layer.logical_widths``, so we fall back to that when the LinearBase
   attribute is missing.  Similarly, ``input_size`` may be missing — the scheme
   already set ``input_size_per_partition`` on the layer, so the existing
   hasattr path covers it.
"""

from __future__ import annotations

from pathlib import Path

HUMMING_DEVICE_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/humming/utils/device.py"
)

# Humming calls pynvml.nvmlDeviceGetMaxClockInfo for NVML_CLOCK_MEM and
# NVML_CLOCK_SM during autotune.  GB10 (Grace Blackwell, SM12.1) doesn't
# expose max clocks through NVML, raising NVMLError_NotSupported.  Wrap both
# call sites with try/except and fall back to a hardcoded conservative MHz.
DEVICE_OLD_MEM = (
    "        mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)\n"
    "        return (mem_clock_mhz * 2 * bus_width) / 8 / 1000\n"
)
DEVICE_NEW_MEM = (
    "        try:\n"
    "            mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)\n"
    "        except pynvml.NVMLError:\n"
    "            # GB10 (SM12.1) doesn't expose mem clock via NVML.\n"
    "            mem_clock_mhz = 8100  # conservative for GB10 GDDR7\n"
    "        return (mem_clock_mhz * 2 * bus_width) / 8 / 1000\n"
)

BUSWIDTH_OLD = (
    "        return (mem_clock_mhz * 2 * bus_width) / 8 / 1000\n"
)
BUSWIDTH_NEW = (
    "        if not bus_width:\n"
    "            # GB10 (SM12.1) unified LPDDR5X reports bus_width=0 via NVML\n"
    "            # (not raised). 135 bits with the 8100 MHz mem-clock fallback\n"
    "            # yields the published DGX Spark ~273 GB/s, so the humming\n"
    "            # compute-bound heuristic denominator is not zero.\n"
    "            bus_width = 135\n"
    "        return (mem_clock_mhz * 2 * bus_width) / 8 / 1000\n"
)


DEVICE_OLD_SM = (
    "        max_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)\n"
)
DEVICE_NEW_SM = (
    "        try:\n"
    "            max_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)\n"
    "        except pynvml.NVMLError:\n"
    "            # GB10 (SM12.1) doesn't expose SM clock via NVML.\n"
    "            max_clock_mhz = 1600  # conservative for GB10\n"
)


HUMMING_UTILS_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/utils/humming_utils.py"
)

# The exact lines in prepare_humming_layer that crash on ParallelLMHead.
HUMMING_OLD = (
    "    if hasattr(layer, \"input_size_per_partition\"):\n"
    "        input_size_per_partition = layer.input_size_per_partition\n"
    "    else:\n"
    "        input_size_per_partition = layer.input_size\n"
    "    shape_k_stacks = [input_size_per_partition]\n"
    "    shape_n_stacks = layer.output_partition_sizes\n"
)

HUMMING_NEW = (
    "    if hasattr(layer, \"input_size_per_partition\"):\n"
    "        input_size_per_partition = layer.input_size_per_partition\n"
    "    elif hasattr(layer, \"input_size\"):\n"
    "        input_size_per_partition = layer.input_size\n"
    "    else:\n"
    "        # ParallelLMHead / VocabParallelEmbedding has no input_size;\n"
    "        # the weight tensor's input_dim (=1) gives us the hidden dim.\n"
    "        input_size_per_partition = layer.weight.shape[1]\n"
    "    shape_k_stacks = [input_size_per_partition]\n"
    "    if hasattr(layer, \"output_partition_sizes\"):\n"
    "        shape_n_stacks = layer.output_partition_sizes\n"
    "    elif hasattr(layer, \"logical_widths\"):\n"
    "        # CompressedTensorsW8A16Fp8.create_weights stores the partition\n"
    "        # sizes as logical_widths on non-LinearBase layers (lm_head).\n"
    "        shape_n_stacks = layer.logical_widths\n"
    "    else:\n"
    "        shape_n_stacks = [layer.weight.shape[0]]\n"
)

# Second use of output_partition_sizes at the prepare_layer_meta call (line ~492).
PREPARE_OLD = (
    "    HummingMethod.prepare_layer_meta(\n"
    "        layer=layer,\n"
    "        shape_n=sum(layer.output_partition_sizes),\n"
    "        shape_k=input_size_per_partition,\n"
    "        weight_schema=weight_schema,\n"
    "        input_schema=input_schema,\n"
    "        pad_n_to_multiple=256,\n"
    "        pad_k_to_multiple=128,\n"
    "        has_bias=layer.has_bias,\n"
    "        torch_dtype=layer.params_dtype,\n"
)
PREPARE_NEW = (
    "    HummingMethod.prepare_layer_meta(\n"
    "        layer=layer,\n"
    "        shape_n=sum(shape_n_stacks),\n"
    "        shape_k=input_size_per_partition,\n"
    "        weight_schema=weight_schema,\n"
    "        input_schema=input_schema,\n"
    "        pad_n_to_multiple=256,\n"
    "        pad_k_to_multiple=128,\n"
    "        has_bias=getattr(layer, \"has_bias\", False),\n"
    "        torch_dtype=layer.params_dtype,\n"
)


LINEAR_INIT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/"
    "linear/__init__.py"
)

# Root fix (replaces the need to make Humming swallow the lm_head at all):
# on SM121 (GB10) the Humming FP8 W8A16 kernel returns zeroed logits, so the
# lm_head argmaxes to token id 0 for every step (pure garbage output). The
# auto-selector walks _POSSIBLE_WFP8A16_KERNELS in order and takes the first
# kernel that can_implement the layer, and Humming is listed first. Marlin's
# FP8 kernel is the numerically-correct path upstream vLLM uses for W8A16 FP8
# and is the proven viable dense backend on SM12x. Put Marlin first; leave
# Humming as the last-resort fallback.
WFP8A16_OLD = (
    "_POSSIBLE_WFP8A16_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {\n"
    "    PlatformEnum.CUDA: [\n"
    "        HummingFP8ScaledMMLinearKernel,\n"
    "        MarlinFP8ScaledMMLinearKernel,\n"
    "    ],\n"
)
WFP8A16_NEW = (
    "_POSSIBLE_WFP8A16_KERNELS: dict[PlatformEnum, list[type[FP8ScaledMMLinearKernel]]] = {\n"
    "    PlatformEnum.CUDA: [\n"
    "        # GB10/SM121: Humming FP8 W8A16 returns zeroed logits (token-0\n"
    "        # garbage). Marlin first so the lm_head uses the correct path.\n"
    "        MarlinFP8ScaledMMLinearKernel,\n"
    "        HummingFP8ScaledMMLinearKernel,\n"
    "    ],\n"
)


def patch_once(target: Path, old: str, new: str) -> str:
    text = target.read_text()
    if new in text:
        return f"PATCH_ALREADY_APPLIED {target.name}"
    if old not in text:
        raise SystemExit(f"PATCH_TARGET_NOT_FOUND {target}")
    if text.count(old) != 1:
        raise SystemExit(f"PATCH_TARGET_NOT_UNIQUE {target} count={text.count(old)}")
    target.write_text(text.replace(old, new, 1))
    return f"PATCH_APPLIED {target.name}"


def main() -> int:
    import py_compile

    print(patch_once(LINEAR_INIT_TARGET, WFP8A16_OLD, WFP8A16_NEW))
    py_compile.compile(str(LINEAR_INIT_TARGET), doraise=True)
    print(patch_once(HUMMING_UTILS_TARGET, HUMMING_OLD, HUMMING_NEW))
    print(patch_once(HUMMING_UTILS_TARGET, PREPARE_OLD, PREPARE_NEW))
    py_compile.compile(str(HUMMING_UTILS_TARGET), doraise=True)
    print(patch_once(HUMMING_DEVICE_TARGET, DEVICE_OLD_MEM, DEVICE_NEW_MEM))
    print(patch_once(HUMMING_DEVICE_TARGET, BUSWIDTH_OLD, BUSWIDTH_NEW))
    print(patch_once(HUMMING_DEVICE_TARGET, DEVICE_OLD_SM, DEVICE_NEW_SM))
    py_compile.compile(str(HUMMING_DEVICE_TARGET), doraise=True)
    print("PATCH_COMPILE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())