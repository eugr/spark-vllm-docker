#!/usr/bin/env python3
"""Let retained FP8 layers share the strict MXFP4 CUTLASS-AFP8 selector."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "QWEN_FP8_FLASHINFER_CUTLASS_AFP8_ALIAS_V2"
KERNEL_CONFIG_MARKER = "QWEN_FLASHINFER_CUTLASS_AFP8_KERNEL_CONFIG_V1"


def patch_source(source: str) -> str:
    if MARKER in source:
        raise ValueError("FP8 backend oracle is already alias-patched")
    old = '''        "flashinfer_cutlass": Fp8MoeBackend.FLASHINFER_CUTLASS,
        "marlin": Fp8MoeBackend.MARLIN,
'''
    new = f'''        "flashinfer_cutlass": Fp8MoeBackend.FLASHINFER_CUTLASS,
        # {MARKER}
        # MXFP4 uses the stricter `_afp8` spelling to forbid a BF16-activation
        # fallback. FlashInfer CUTLASS does not support Qwen's retained
        # block-128 FP8 scheme on SM121, while Triton does. Therefore the same
        # strict runner selector intentionally dispatches retained FP8 layers
        # to Triton and converted MXFP4 layers to FlashInfer CUTLASS.
        "flashinfer_cutlass_afp8": Fp8MoeBackend.TRITON,
        "marlin": Fp8MoeBackend.MARLIN,
'''
    if source.count(old) != 1:
        raise ValueError("expected FP8 backend mapping block exactly once")
    return source.replace(old, new)


def patch_kernel_source(source: str) -> str:
    if KERNEL_CONFIG_MARKER in source:
        raise ValueError("kernel config is already alias-patched")
    old = '''    "flashinfer_trtllm",
    "flashinfer_cutlass",
    "flashinfer_cutedsl",
'''
    new = f'''    "flashinfer_trtllm",
    "flashinfer_cutlass",
    # {KERNEL_CONFIG_MARKER}
    "flashinfer_cutlass_afp8",
    "flashinfer_cutedsl",
'''
    if source.count(old) != 1:
        raise ValueError("expected kernel MoE backend literal block exactly once")
    return source.replace(old, new)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "model_executor/layers/fused_moe/oracle/fp8.py"
        ),
    )
    parser.add_argument(
        "--kernel-config-path",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/config/kernel.py"
        ),
    )
    args = parser.parse_args()
    args.path.write_text(patch_source(args.path.read_text()))
    args.kernel_config_path.write_text(
        patch_kernel_source(args.kernel_config_path.read_text())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
