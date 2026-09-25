#!/usr/bin/env python3
# Guard against NaN NVFP4 activation global scale from failed llm-compressor
# calibration on some REAP-pruned layers. Same failure class documented for
# the sibling checkpoint in mods/minimax-m2.7-reap-nvfp4-gb10/patch_nvfp4_scalefix.py
# (scottgl/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10, layers 60/61). CT stores
# input_global_scale as a divisor 1/scale; NaN -> 1/NaN=NaN -> alpha=NaN -> the
# W4A4 dynamic-local activation quant emits NaN, which propagates through
# attention to the final hidden state so every token argmaxes to id 0
# (null-byte garbage) -- exactly what we saw on
# dervig/m51Lab-MiniMax-M2.7-REAP-139B-A10B-NVFP4-GB10 with the default
# FLASHINFER_CUTLASS backend. Unlike the sibling patch, we don't know which
# layer indices are defective on this checkpoint, so this substitutes a
# generic safe scale (~0.02, the typical healthy-layer magnitude on the
# sibling checkpoint) for ANY NaN, rather than hardcoding specific layers.
import py_compile
from pathlib import Path

T = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py"
)
OLD = "            input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)\n"
NEW = (
    "            input_global_scale_inv = layer.input_global_scale.max().to(torch.float32)\n"
    "            if torch.isnan(input_global_scale_inv):\n"
    "                # Failed-calibration NaN activation scale in checkpoint;\n"
    "                # substitute a typical healthy-layer magnitude (~0.02) to\n"
    "                # avoid NaN cascading through attention (id-0 garbage).\n"
    "                input_global_scale_inv = torch.full_like(\n"
    "                    input_global_scale_inv, 1.0 / 0.02\n"
    "                )\n"
)

t = T.read_text()
if NEW in t:
    print("SCALEFIX_ALREADY")
elif OLD not in t:
    raise SystemExit("SCALEFIX_ANCHOR_NOT_FOUND")
else:
    T.write_text(t.replace(OLD, NEW, 1))
    py_compile.compile(str(T), doraise=True)
    print("SCALEFIX_APPLIED")
