#!/usr/bin/env python3
# Fix for scottgl/MiniMax-M2.7-REAP-172B-A10B-NVFP4-GB10: layers 60 and 61 ship
# a NaN NVFP4 activation global scale for the fused qkv_proj (failed calibration
# on the last two of the 62 layers). CT stores input_global_scale as a divisor
# 1/scale; NaN -> 1/NaN=NaN -> alpha=NaN -> the W4A4 dynamic-local activation
# quant emits NaN, which propagates through attention to the final hidden state
# so every token argmaxes to id 0 (null-byte garbage). Substitute the adjacent
# healthy layer 59 scale (~0.019, divisor ~1/0.019) for any NaN divisor. Only the
# defective layers are touched; finite scales pass through unchanged.
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
    "                # MiniMax-M2.7 REAP layers 60/61: NaN activation scale in\n"
    "                # checkpoint; use adjacent healthy layer 59 value (~0.019).\n"
    "                input_global_scale_inv = torch.full_like(\n"
    "                    input_global_scale_inv, 1.0 / 0.019\n"
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
