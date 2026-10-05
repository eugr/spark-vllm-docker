#!/usr/bin/env python3
"""Fill the missing ple_embedding_dtype in Qwen3.8-Flash-Next NVFP4 configs.

The local-inference-lab/Qwen3.8-Flash-Next-NVFP4 checkpoint quantizes its PLE
n-gram embedding to NVFP4 (group_size 16, declared under
quantization_config.quantized_layers) but its config.json text_config omits
the ple_embedding_dtype field. vLLM defaults that field to "bfloat16", plans
the PLE table for bf16 storage, and weight loading fails with:

    ValueError: shape mismatch for PLE shard N weight: expected
    (rows, head_dim), got (rows, head_dim // 2)

This tool injects the storage dtype implied by the checkpoint's own
quantization_config. It never overrides an explicit, consistent
ple_embedding_dtype and refuses to touch configs that do not match the
expected checkpoint shape.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PLE_DTYPE_KEY = "ple_embedding_dtype"
NGRAM_MODULE_SUFFIX = "ple_embedding.ngram_embedding"
FAMILY_KEYS = ("ple_embed_dim", "ngram_vocab_size_base", "split_ngram_parts")

# Storage-mode mapping, mirroring
# vllm/models/qwen3_8_flash_next/ple_layer.py _STORAGE_MODES.
STORAGE_MODES = {
    "bfloat16": "bf16",
    "float8_e4m3fn": "fp8_e4m3_per_tensor",
    "float4_e2m1fn_x2": "nvfp4_group16",
    "nvfp4": "nvfp4_group16",
    "nvfp4_group16": "nvfp4_group16",
    "uint8": "nvfp4_group16",
}

# quant_algo values this mod knows how to translate into a vLLM PLE storage
# dtype. "nvfp4" is the canonical name for the nvfp4_group16 storage mode.
ALGO_TO_DTYPE = {
    "NVFP4": "nvfp4",
}


class ConfigPatchError(RuntimeError):
    """Raised when a config cannot be safely patched."""


def find_ngram_quant_entry(config: dict) -> dict | None:
    """Return the quantization entry for the PLE n-gram embedding, if any."""
    quant_config = config.get("quantization_config")
    if not isinstance(quant_config, dict):
        return None
    layers = quant_config.get("quantized_layers")
    if not isinstance(layers, dict):
        return None
    entries = [
        entry
        for name, entry in layers.items()
        if name.endswith(NGRAM_MODULE_SUFFIX) and isinstance(entry, dict)
    ]
    if len(entries) > 1:
        raise ConfigPatchError(
            f"multiple quantization entries for {NGRAM_MODULE_SUFFIX}"
        )
    return entries[0] if entries else None


def expected_dtype(config: dict) -> str:
    """Derive the PLE storage dtype from the checkpoint quantization config."""
    entry = find_ngram_quant_entry(config)
    if entry is None:
        raise ConfigPatchError(
            "no ple_embedding.ngram_embedding entry under "
            "quantization_config.quantized_layers; refusing to guess a PLE "
            "storage dtype"
        )
    algo = entry.get("quant_algo")
    dtype = ALGO_TO_DTYPE.get(algo)
    if dtype is None:
        raise ConfigPatchError(
            f"unsupported PLE quant_algo {algo!r}; this tool only translates "
            f"{sorted(ALGO_TO_DTYPE)}"
        )
    return dtype


def detect_indent(raw: str) -> int:
    """Detect the JSON indentation unit from the first indented line."""
    for line in raw.splitlines()[1:]:
        stripped = line.lstrip(" ")
        if stripped and stripped != line:
            return len(line) - len(stripped)
    return 2


def patch_config_file(path: str) -> str:
    """Patch one config.json; returns an action label for logging."""
    config_path = Path(path)
    try:
        raw = config_path.read_text(encoding="utf-8")
        config = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigPatchError(f"{config_path}: unreadable config ({error})") from error

    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ConfigPatchError(f"{config_path}: no text_config object")
    missing = [key for key in FAMILY_KEYS if key not in text_config]
    if missing:
        raise ConfigPatchError(
            f"{config_path}: text_config lacks PLE family keys {missing}; "
            "not a Qwen3.8-Flash-Next checkpoint"
        )

    expected_mode = STORAGE_MODES[expected_dtype(config)]
    existing = text_config.get(PLE_DTYPE_KEY)
    if existing is not None:
        if not isinstance(existing, str) or existing not in STORAGE_MODES:
            raise ConfigPatchError(
                f"{config_path}: {PLE_DTYPE_KEY}={existing!r} is not a "
                "recognized PLE storage dtype"
            )
        if STORAGE_MODES[existing] != expected_mode:
            raise ConfigPatchError(
                f"{config_path}: {PLE_DTYPE_KEY}={existing!r} contradicts the "
                f"checkpoint quantization ({expected_mode}); refusing to load"
            )
        print(
            f"{config_path}: {PLE_DTYPE_KEY} already set to {existing!r}; skipping"
        )
        return "already-set"

    dtype = expected_dtype(config)
    text_config[PLE_DTYPE_KEY] = dtype
    indent = detect_indent(raw)
    trailing_newline = raw.endswith("\n")
    updated = json.dumps(config, indent=indent, ensure_ascii=False)
    if trailing_newline:
        updated += "\n"

    # Snapshot config.json is a symlink into the HF blob store; resolve it and
    # replace the blob atomically so the snapshot symlink survives.
    target = config_path.resolve()
    tmp = target.with_name(f"{target.name}.plefix-{os.getpid()}.tmp")
    try:
        tmp.write_text(updated, encoding="utf-8")
        os.replace(tmp, target)
    except OSError as error:
        tmp.unlink(missing_ok=True)
        raise ConfigPatchError(f"{config_path}: write failed ({error})") from error

    print(f"{config_path}: set {PLE_DTYPE_KEY}={dtype!r} in text_config")
    return "patched"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: patch_config.py CONFIG_JSON [CONFIG_JSON ...]",
            file=sys.stderr,
        )
        return 2
    try:
        for path in argv[1:]:
            patch_config_file(path)
    except ConfigPatchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
