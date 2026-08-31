#!/usr/bin/env python3
"""Add per-file model-index filtering for zero-copy retained source shards."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "QWEN_INDEX_FILTERED_SAFETENSORS_SOURCE_VIEW_V1"


def patch_weight_utils(source: str) -> str:
    if MARKER in source:
        raise ValueError("weight_utils.py is already index-filter patched")
    old_signature = '''    *,
    safetensors_prefetch_num_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
'''
    new_signature = f'''    *,
    # {MARKER}
    allowed_weight_names_by_file: dict[str, set[str]] | None = None,
    safetensors_prefetch_num_threads: int = DEFAULT_SAFETENSORS_PREFETCH_NUM_THREADS,
'''
    old_loop = '''    for st_file in tqdm(
        sorted_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        if safetensors_load_strategy == "eager":
'''
    new_loop = '''    for st_file in tqdm(
        sorted_files,
        desc=loading_desc,
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        allowed_names = None
        if allowed_weight_names_by_file is not None:
            allowed_names = allowed_weight_names_by_file.get(os.path.realpath(st_file))
            if allowed_names is None:
                raise ValueError(
                    f"checkpoint index has no allowed-name set for {st_file}"
                )
        if safetensors_load_strategy == "eager":
'''
    old_eager = '''            for name, param in state_dict.items():
                if not should_skip_weight(name, local_expert_ids):
                    yield name, param
'''
    new_eager = '''            for name, param in state_dict.items():
                if (allowed_names is None or name in allowed_names) and not should_skip_weight(
                    name, local_expert_ids
                ):
                    yield name, param
'''
    old_torchao = '''                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    state_dict[name] = f.get_tensor(name)
'''
    new_torchao = '''                for name in f.keys():  # noqa: SIM118
                    if (
                        (allowed_names is not None and name not in allowed_names)
                        or should_skip_weight(name, local_expert_ids)
                    ):
                        continue
                    state_dict[name] = f.get_tensor(name)
'''
    old_standard = '''                for name in f.keys():  # noqa: SIM118
                    if should_skip_weight(name, local_expert_ids):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
'''
    new_standard = '''                for name in f.keys():  # noqa: SIM118
                    if (
                        (allowed_names is not None and name not in allowed_names)
                        or should_skip_weight(name, local_expert_ids)
                    ):
                        continue
                    param = f.get_tensor(name)
                    yield name, param
'''
    replacements = (
        (old_signature, new_signature, "safetensors iterator signature"),
        (old_loop, new_loop, "safetensors file loop"),
        (old_eager, new_eager, "eager filtering branch"),
        (old_torchao, new_torchao, "torchao filtering branch"),
        (old_standard, new_standard, "standard filtering branch"),
    )
    for old, new, label in replacements:
        if source.count(old) != 1:
            raise ValueError(f"expected {label} exactly once")
        source = source.replace(old, new)
    return source


def patch_default_loader(source: str) -> str:
    if MARKER in source:
        raise ValueError("default_loader.py is already index-filter patched")
    old_import = '''import glob
import os
import time
'''
    new_import = '''import glob
import json
import os
import time
'''
    old_prepare = '''        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if self.load_config.load_format == "npcache":
'''
    new_prepare = f'''        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        # {MARKER}. A source-view checkpoint may hardlink a source shard that
        # contains tensors not mapped by this candidate's index. Build an exact
        # per-file allow-list and use the pre-read filtered iterator below.
        requires_index_tensor_filtering = False
        allowed_weight_names_by_file: dict[str, set[str]] | None = None
        if use_safetensors:
            index_path = os.path.join(hf_folder, SAFE_WEIGHTS_INDEX_NAME)
            if os.path.isfile(index_path):
                with open(index_path, encoding="utf-8") as index_handle:
                    checkpoint_index = json.load(index_handle)
                requires_index_tensor_filtering = bool(
                    checkpoint_index.get("metadata", {{}}).get(
                        "requires_index_tensor_filtering", False
                    )
                )
                if requires_index_tensor_filtering:
                    allowed_weight_names_by_file = {{}}
                    for name, filename in checkpoint_index["weight_map"].items():
                        path = os.path.realpath(os.path.join(hf_folder, filename))
                        allowed_weight_names_by_file.setdefault(path, set()).add(name)
                    prepared_files = {{os.path.realpath(path) for path in hf_weights_files}}
                    indexed_files = set(allowed_weight_names_by_file)
                    if prepared_files != indexed_files:
                        raise ValueError(
                            "prepared safetensors files differ from the filtered index: "
                            f"unindexed={{sorted(prepared_files-indexed_files)}}, "
                            f"missing={{sorted(indexed_files-prepared_files)}}"
                        )
        if self.load_config.load_format == "npcache":
'''
    old_safetensors = '''        elif use_safetensors:
            if self.load_config.load_format == "fastsafetensors":
'''
    new_safetensors = '''        elif use_safetensors:
            fast_source_view = (
                requires_index_tensor_filtering
                and self.load_config.load_format == "fastsafetensors"
                and os.environ.get(
                    "QWEN_FAST_SOURCE_VIEW_DUPLICATES_IDENTICAL", "0"
                ) == "1"
            )
            if fast_source_view:
                logger.warning_once(
                    "Using audited fastsafetensors source-view path; physically "
                    "duplicate tensor payloads must be byte-identical."
                )
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
            elif requires_index_tensor_filtering:
                if self.load_config.load_format in (
                    "fastsafetensors",
                    "instanttensor",
                ) or extra_config.get("enable_multithread_load"):
                    logger.warning_once(
                        "Checkpoint requires per-file index filtering; using the "
                        "standard safetensors iterator instead of %s.",
                        self.load_config.load_format,
                    )
                weights_iterator = safetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    self.load_config.safetensors_load_strategy,
                    local_expert_ids=self.local_expert_ids,
                    allowed_weight_names_by_file=allowed_weight_names_by_file,
                    safetensors_prefetch_num_threads=(
                        self.load_config.safetensors_prefetch_num_threads
                    ),
                    safetensors_prefetch_block_size=(
                        self.load_config.safetensors_prefetch_block_size
                    ),
                )
            elif self.load_config.load_format == "fastsafetensors":
'''
    replacements = (
        (old_import, new_import, "json import"),
        (old_prepare, new_prepare, "checkpoint-index preparation"),
        (old_safetensors, new_safetensors, "filtered iterator dispatch"),
    )
    for old, new, label in replacements:
        if source.count(old) != 1:
            raise ValueError(f"expected {label} exactly once")
        source = source.replace(old, new)
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight-utils",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "model_executor/model_loader/weight_utils.py"
        ),
    )
    parser.add_argument(
        "--default-loader",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "model_executor/model_loader/default_loader.py"
        ),
    )
    args = parser.parse_args()
    args.weight_utils.write_text(patch_weight_utils(args.weight_utils.read_text()))
    args.default_loader.write_text(patch_default_loader(args.default_loader.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
