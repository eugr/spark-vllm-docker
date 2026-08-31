#!/usr/bin/env python3
"""Add explicit layer-level FP8 retention to vLLM's MXFP4 config."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "QWEN_MIXED_FP8_MXFP4_ROUTED_LAYERS_V3"


def patch_source(source: str) -> str:
    if MARKER in source:
        raise ValueError("mxfp4.py is already mixed-layer patched")

    marker = MARKER
    old_init = '''    def __init__(self, ignored_layers: list[str] | None = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()
'''
    new_init = f'''    # {marker}
    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        fp8_routed_layers: list[str] | None = None,
        fp8_mtp_routed_layers: list[str] | None = None,
        fp8_ple_embeddings: bool = False,
        fp8_weight_block_size: list[int] | None = None,
    ):
        super().__init__()
        self.ignored_layers = ignored_layers
        self.fp8_routed_layers = fp8_routed_layers or []
        self.fp8_mtp_routed_layers = fp8_mtp_routed_layers or []
        self.fp8_ple_embeddings = bool(fp8_ple_embeddings)
        self.fp8_weight_block_size = fp8_weight_block_size or [128, 128]
        if len(self.fp8_weight_block_size) != 2:
            raise ValueError("fp8_weight_block_size must have two dimensions")

    @classmethod
    def from_config(cls, config):
        return cls(
            ignored_layers=config.get("ignored_layers"),
            fp8_routed_layers=config.get("fp8_routed_layers"),
            fp8_mtp_routed_layers=config.get("fp8_mtp_routed_layers"),
            fp8_ple_embeddings=config.get("fp8_ple_embeddings", False),
            fp8_weight_block_size=config.get("fp8_weight_block_size"),
        )

    @staticmethod
    def _routed_layer_index(prefix: str) -> int | None:
        """Return the absolute decoder index from a routed-expert path.

        Qwen3.8-Flash-Next checkpoint names and vLLM runtime module names do
        not have the same root.  For example, the checkpoint stores
        ``model.language_model.layers.19.mlp.experts`` while the multimodal
        runtime constructs ``language_model.model.layers.19.mlp.experts``.
        The absolute layer index and the ``mlp.experts`` suffix are stable
        across those namespaces.
        """
        parts = prefix.split(".")
        for index in range(len(parts) - 3):
            if (
                parts[index] == "layers"
                and parts[index + 1].isdigit()
                and parts[index + 2] == "mlp"
                and parts[index + 3] == "experts"
            ):
                return int(parts[index + 1])
        return None

    def _matches_fp8_routed_layer(
        self,
        prefix: str,
        configured_prefixes: list[str],
    ) -> bool:
        # Preserve exact matching for ordinary vLLM checkpoints, then bridge
        # Qwen's HF-to-vLLM namespace remapping by absolute decoder index.
        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=configured_prefixes,
            fused_mapping=self.packed_modules_mapping,
        ):
            return True
        runtime_index = self._routed_layer_index(prefix)
        if runtime_index is None:
            return False
        configured_indices = {{
            layer_index
            for configured_prefix in configured_prefixes
            if (layer_index := self._routed_layer_index(configured_prefix)) is not None
        }}
        return runtime_index in configured_indices
'''
    old_routed = '''        elif isinstance(layer, RoutedExperts):
            return self._make_moe_method(layer.moe_config)
'''
    new_routed = '''        elif isinstance(layer, RoutedExperts):
            fp8_prefixes = self.fp8_routed_layers + self.fp8_mtp_routed_layers
            if fp8_prefixes and self._matches_fp8_routed_layer(
                prefix,
                fp8_prefixes,
            ):
                # The retained checkpoint tensors are the original serialized
                # block-FP8 weights and weight_scale_inv tensors. Reuse vLLM's
                # existing FP8 MoE loader and kernel; do not dequantize them.
                from vllm.model_executor.layers.quantization.fp8 import (
                    Fp8Config,
                    Fp8MoEMethod,
                )

                fp8_config = Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    activation_scheme="dynamic",
                    weight_block_size=self.fp8_weight_block_size,
                )
                return Fp8MoEMethod(fp8_config, layer)
            return self._make_moe_method(layer.moe_config)
'''
    if source.count(old_init) != 1:
        raise ValueError("expected Mxfp4Config init/from_config block exactly once")
    if source.count(old_routed) != 1:
        raise ValueError("expected Mxfp4Config RoutedExperts branch exactly once")
    return source.replace(old_init, new_init).replace(old_routed, new_routed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "model_executor/layers/quantization/mxfp4.py"
        ),
    )
    args = parser.parse_args()
    args.path.write_text(patch_source(args.path.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
