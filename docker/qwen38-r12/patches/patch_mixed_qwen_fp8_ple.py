#!/usr/bin/env python3
"""Enable Qwen's preserved global-scale FP8 PLE under mixed MXFP4 config."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "QWEN_MIXED_MXFP4_PRESERVED_FP8_PLE_V1"


def patch_source(source: str) -> str:
    if MARKER in source:
        raise ValueError("Qwen NVIDIA PLE layer is already mixed-config patched")
    old = '''    if not isinstance(quant_config, Fp8Config):
        return None
    if not quant_config.is_checkpoint_fp8_serialized:
        return None

    ignored_layers = quant_config.ignored_layers
    if is_layer_skipped(
        prefix,
        ignored_layers,
        quant_config.packed_modules_mapping,
        match_mode=quant_config.ignored_layers_match_mode,
    ):
        return None
    # PLE checkpoint shards form one runtime embedding parameter.
    shard_prefix = f"{prefix}.shard_"
    if any(name.startswith(shard_prefix) for name in ignored_layers):
        return None
    return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
'''
    new = f'''    # {MARKER}
    # Mixed Qwen checkpoints preserve the enormous n-gram/PLE table in its
    # original global-scale FP8 representation while only routed experts use
    # MXFP4. The mixed Mxfp4Config marks this explicitly; reuse the existing
    # Qwen FP8 embedding implementation so both weight and scale are loaded.
    if getattr(quant_config, "fp8_ple_embeddings", False):
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()

    if not isinstance(quant_config, Fp8Config):
        return None
    if not quant_config.is_checkpoint_fp8_serialized:
        return None

    ignored_layers = quant_config.ignored_layers
    if is_layer_skipped(
        prefix,
        ignored_layers,
        quant_config.packed_modules_mapping,
        match_mode=quant_config.ignored_layers_match_mode,
    ):
        return None
    # PLE checkpoint shards form one runtime embedding parameter.
    shard_prefix = f"{{prefix}}.shard_"
    if any(name.startswith(shard_prefix) for name in ignored_layers):
        return None
    return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
'''
    if source.count(old) != 1:
        raise ValueError("expected Qwen PLE quant-method selector exactly once")
    return source.replace(old, new)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "models/qwen3_8_flash_next/nvidia/ple_layer.py"
        ),
    )
    args = parser.parse_args()
    args.path.write_text(patch_source(args.path.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
