#!/usr/bin/env python3
"""Disable the MiniMax QK RMSNorm CUDA IPC fusion on multi-node DGX Spark."""

import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "vllm/model_executor/layers/minimax_rms_norm/rms_norm_tp.py"
marker = (
    '_MINIMAX_FUSED_AR_RMS_QK = getattr(torch.ops._C, '
    '"minimax_allreduce_rms_qk", None)'
)
replacement = (
    "_MINIMAX_FUSED_AR_RMS_QK = None  "
    "# Disabled for DGX Spark multi-node TP"
)

if target.exists() and marker in (text := target.read_text()):
    print("MiniMax QK norm fusion found; disabling CUDA IPC fused path")
    target.write_text(text.replace(marker, replacement))
elif target.exists() and replacement in target.read_text():
    print("MiniMax QK norm fusion already disabled")
else:
    print("MiniMax QK norm fusion marker not present; skipping patch")

if target.exists() and marker in target.read_text():
    raise SystemExit("ERROR: MiniMax QK norm fusion marker is still present")
