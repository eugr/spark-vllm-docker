#!/usr/bin/env python3
"""Patch the Qwen3.8 target model to prepare exact GDN stream overlap."""

from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/model.py"
)
source = path.read_text()

old_import = """from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
"""
new_import = """from .gdn_overlap import (
    QwenGatedDeltaNetAttention,
    prepare_gdn_projection_overlap,
)
"""
if source.count(old_import) != 1:
    raise RuntimeError("unexpected QwenGatedDeltaNetAttention import layout")
source = source.replace(old_import, new_import, 1)

conditional_marker = "class Qwen3_8FlashNextForConditionalGeneration("
marker_offset = source.find(conditional_marker)
if marker_offset < 0:
    raise RuntimeError("Qwen3_8FlashNextForConditionalGeneration is missing")
prefix, conditional_source = source[:marker_offset], source[marker_offset:]
old_load = """        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
"""
new_load = """        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        prepare_gdn_projection_overlap(self)
        return loaded

    @classmethod
    def get_mamba_state_dtype_from_config(
"""
if conditional_source.count(old_load) != 1:
    raise RuntimeError("unexpected outer conditional load_weights layout")
source = prefix + conditional_source.replace(old_load, new_load, 1)
path.write_text(source)
