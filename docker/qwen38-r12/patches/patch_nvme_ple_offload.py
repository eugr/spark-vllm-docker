#!/usr/bin/env python3
"""Feature-gate Qwen PLE behind local-NVMe row lookup in vLLM.

This patch preserves vLLM's native asynchronous PLE worker, pinned output
buffers, H2D streams, and CUDA semaphores.  It changes only table ownership and
lookup: the CPU worker no longer allocates or loads the 51.2 GB embedding and
the standard loader skips those tensors before materialization.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MARKER = "QWEN_LOCAL_NVME_PLE_OFFLOAD_V1"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected {label} exactly once, found {source.count(old)}")
    return source.replace(old, new)


def patch_ple_layer(source: str) -> str:
    if MARKER in source:
        raise ValueError("PLE layer is already NVMe patched")
    source = replace_once(
        source,
        '''import math
from collections.abc import Iterable, Sequence
''',
        '''import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
''',
        "PLE imports",
    )
    source = replace_once(
        source,
        '''from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from ..common.ple import copy_ple_embedding_shard_
''',
        f'''from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.ple_offload.nvme_table import (
    DEFAULT_PLE_PREFIX,
    PleMmapTable,
    load_qwen_ple_hash_parameters,
)

from ..common.ple import copy_ple_embedding_shard_

# {MARKER}
''',
        "NVMe module imports",
    )
    source = replace_once(
        source,
        '''class Qwen3_8FlashNextNGramEmbedding(PleOffloadLayer):
''',
        '''class Qwen3_8FlashNextPleNvmeEmbedding(nn.Module):
    """Parameter-free shape contract for the disk-backed FP8 table."""

    def __init__(self, org_vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.org_vocab_size = org_vocab_size
        self.embedding_dim = embedding_dim


class Qwen3_8FlashNextNGramEmbedding(PleOffloadLayer):
''',
        "parameter-free embedding class",
    )
    source = replace_once(
        source,
        '''        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
''',
        '''        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")
        nvme_path = os.environ.get("VLLM_PLE_NVME_PATH")
        self._nvme_path = (
            Path(nvme_path).resolve()
            if nvme_path and is_offload_process()
            else None
        )
        self._nvme_backend = os.environ.get("VLLM_PLE_NVME_BACKEND", "auto")
        if self._nvme_backend not in {"auto", "mmap", "pread"}:
            raise ValueError(
                "VLLM_PLE_NVME_BACKEND must be 'auto', 'mmap', or 'pread'"
            )
        self._nvme_pread_workers = int(
            os.environ.get("VLLM_PLE_NVME_PREAD_WORKERS", "16")
        )
        self._nvme_pread_max_tokens = int(
            os.environ.get("VLLM_PLE_NVME_PREAD_MAX_TOKENS", "32")
        )
        if self._nvme_pread_workers <= 0 or self._nvme_pread_max_tokens <= 0:
            raise ValueError("NVMe PLE pread worker and token limits must be positive")
        workspace_device = torch.device("cpu") if self._nvme_path else None

        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
''',
        "NVMe constructor gate",
    )
    source = replace_once(
        source,
        '''            torch.tensor(multipliers, dtype=torch.long),
''',
        '''            torch.tensor(
                multipliers, dtype=torch.long, device=workspace_device
            ),
''',
        "multiplier device",
    )
    source = replace_once(
        source,
        '''            torch.tensor(sizes, dtype=torch.long),
''',
        '''            torch.tensor(sizes, dtype=torch.long, device=workspace_device),
''',
        "vocab-size device",
    )
    source = replace_once(
        source,
        '''            torch.tensor(offsets, dtype=torch.long),
''',
        '''            torch.tensor(offsets, dtype=torch.long, device=workspace_device),
''',
        "offset device",
    )
    source = replace_once(
        source,
        '''        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),
        )
''',
        '''        self._nvme_table: PleMmapTable | None = None
        if self._nvme_path is not None:
            self.ngram_embedding = Qwen3_8FlashNextPleNvmeEmbedding(
                padded_vocab_size, self.head_dim
            )
            self._nvme_table = PleMmapTable.from_checkpoint(
                self._nvme_path,
                prefix=DEFAULT_PLE_PREFIX,
                expected_parts=self.split_ngram_parts,
                expected_embedding_dim=self.head_dim,
                expected_dtype="F8_E4M3",
                require_local_filesystem=True,
            )
            if self._nvme_table.total_rows != padded_vocab_size:
                raise ValueError(
                    "NVMe PLE rows do not match the configured padded vocabulary: "
                    f"{self._nvme_table.total_rows} != {padded_vocab_size}"
                )
            checkpoint_hash = load_qwen_ple_hash_parameters(
                self._nvme_path, prefix=DEFAULT_PLE_PREFIX
            )
            generated = (
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
            )
            expected = (
                checkpoint_hash.layer_multipliers,
                checkpoint_hash.head_vocab_sizes,
                checkpoint_hash.head_offsets,
            )
            if not all(
                np.array_equal(actual.detach().cpu().numpy(), reference)
                for actual, reference in zip(generated, expected, strict=True)
            ):
                raise ValueError("configured PLE hash tensors differ from checkpoint")
        else:
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim,
                params_dtype=params_dtype,
                padding_size=divisor,
                prefix=f"{prefix}.ngram_embedding",
                quant_method=_get_ple_embedding_quant_method(
                    quant_config, f"{prefix}.ngram_embedding"
                ),
            )
''',
        "embedding allocation gate",
    )
    source = replace_once(
        source,
        '''            torch.arange(max_total_tokens, dtype=torch.int64),
''',
        '''            torch.arange(
                max_total_tokens, dtype=torch.int64, device=workspace_device
            ),
''',
        "positions workspace device",
    )
    source = replace_once(
        source,
        '''                dtype=torch.int64,
            ),
            persistent=False,
        )

    @staticmethod
''',
        '''                dtype=torch.int64,
                device=workspace_device,
            ),
            persistent=False,
        )

    @staticmethod
''',
        "padded workspace device",
    )
    source = replace_once(
        source,
        '''        ngram_ids = torch.cat(id_blocks, dim=-1)
        if output_buffer is not None:
''',
        '''        ngram_ids = torch.cat(id_blocks, dim=-1)
        if getattr(self, "_nvme_table", None) is not None:
            if output_buffer is None:
                raise ValueError("NVMe PLE requires the offload worker output buffer")
            row_ids = ngram_ids.detach().cpu().numpy()
            use_pread = self._nvme_backend == "pread" or (
                self._nvme_backend == "auto"
                and num_valid_tokens <= self._nvme_pread_max_tokens
            )
            if use_pread:
                raw_rows = self._nvme_table.gather_pread(
                    row_ids,
                    deduplicate=True,
                    workers=self._nvme_pread_workers,
                )
            else:
                raw_rows = self._nvme_table.gather(row_ids, deduplicate=True)
            output = output_buffer[:num_tokens, : self.embedding_dim]
            raw_output = output.view(torch.uint8)
            source = torch.from_numpy(
                raw_rows.reshape(num_tokens, self.embedding_dim)
            )
            raw_output.copy_(source)
            return output
        if output_buffer is not None:
''',
        "NVMe forward gather",
    )
    source = replace_once(
        source,
        '''        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)
''',
        '''        if getattr(self, "_nvme_table", None) is not None:
            return torch.float8_e4m3fn
        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)
''',
        "NVMe output dtype",
    )
    return source


def patch_offload_worker(source: str) -> str:
    if MARKER in source:
        raise ValueError("PLE offload worker is already NVMe patched")
    source = replace_once(
        source,
        '''import contextlib
import multiprocessing.process
import pickle
''',
        '''import contextlib
import multiprocessing.process
import os
import pickle
''',
        "worker os import",
    )
    source = replace_once(
        source,
        '''    READY_STR = "READY"
''',
        '''    READY_STR = "READY"
    INITIALIZED_STR = "INITIALIZED"
''',
        "PLE initialization handshake constant",
    )
    source = replace_once(
        source,
        '''    @staticmethod
    def wait_for_ready(handle: PleOffloadWorkerHandle) -> None:
        """Wait until weights and all GPU registrations are ready to serve."""
''',
        f'''    @staticmethod
    def wait_for_initialized(handle: PleOffloadWorkerHandle) -> None:
        """Wait until the CPU/meta discovery model is built before GPU loading.

        # {MARKER}. On unified-memory GPUs, allowing the parent and child to
        # construct concurrently creates a race: the parent can consume most
        # physical memory before the child's nominally-meta model reaches a
        # small backend allocation. Keep the child ahead of the parent.
        """
        reader = handle.ready_pipe_reader
        if reader is None:
            return
        if not reader.poll(envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT):
            raise TimeoutError(
                "PLE offload worker did not initialize within "
                f"{{envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT}}s."
            )
        try:
            message = reader.recv()
        except EOFError as error:
            raise RuntimeError(
                "PLE offload worker exited during initialization"
            ) from error
        if message.get("status") != PleOffloadWorker.INITIALIZED_STR:
            raise RuntimeError(
                "PLE offload worker failed during initialization: "
                f"{{message.get('error', 'unknown error')}}"
            )
        logger.info("PLE offload worker initialized before parent model load.")

    @staticmethod
    def wait_for_ready(handle: PleOffloadWorkerHandle) -> None:
        """Wait until weights and all GPU registrations are ready to serve."""
''',
        "PLE initialization handshake wait",
    )
    source = replace_once(
        source,
        '''            # READY means that the process can immediately serve requests. Wait
            # for every DP/TP worker to register before notifying the parent.
            runner.accept_registrations(pull_socket, num_workers)
''',
        f'''            # {MARKER}. Confirm that CPU/meta discovery is complete before the
            # parent starts loading the large GPU model. READY remains the
            # second message and still means registrations can be served.
            ready_writer.send({{"status": PleOffloadWorker.INITIALIZED_STR}})

            # READY means that the process can immediately serve requests. Wait
            # for every DP/TP worker to register before notifying the parent.
            runner.accept_registrations(pull_socket, num_workers)
''',
        "PLE initialization handshake send",
    )
    source = replace_once(
        source,
        '''        logger.info(
            "Found %d PleOffloadLayer(s): %s",
            len(offload_layers),
            sorted(offload_layers),
        )
        offload_prefixes = tuple(f"{name}." for name in offload_layers)
''',
        f'''        logger.info(
            "Found %d PleOffloadLayer(s): %s",
            len(offload_layers),
            sorted(offload_layers),
        )
        # {MARKER}. The constructors have already validated and opened the
        # local-NVMe table plus the three tiny hash tensors. Bypass the default
        # loader so it cannot allocate or stream the 51.2 GB table in this CPU
        # worker. The GPU model loader independently retains the global FP8
        # scale and skips only the 128 table shards before materialization.
        if os.environ.get("VLLM_PLE_NVME_PATH"):
            missing_tables = [
                name
                for name, layer in offload_layers.items()
                if getattr(layer, "_nvme_table", None) is None
            ]
            if missing_tables:
                raise RuntimeError(
                    f"NVMe PLE layers did not open their table: {{missing_tables}}"
                )
            self._layers.update(offload_layers)
            del model
            logger.info(
                "NVMe PLE initialization complete; skipped resident table load."
            )
            return
        offload_prefixes = tuple(f"{{name}}." for name in offload_layers)
''',
        "worker resident-load bypass",
    )
    return source


def patch_weight_utils(source: str) -> str:
    if MARKER in source:
        raise ValueError("weight_utils.py is already NVMe patched")
    source = replace_once(
        source,
        '''_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\\n"  # noqa: E501


def enable_tqdm(use_tqdm_on_load: bool):
''',
        f'''_BAR_FORMAT = "{{desc}}: {{percentage:3.0f}}% Completed | {{n_fmt}}/{{total_fmt}} [{{elapsed}}<{{remaining}}, {{rate_fmt}}]\\n"  # noqa: E501


def _qwen_should_skip_nvme_ple(name: str) -> bool:
    # {MARKER}. Keep the global scale and tiny hash metadata; skip only the
    # 128 enormous table tensors before safetensors materializes them.
    return bool(os.environ.get("VLLM_PLE_NVME_PATH")) and (
        ".ple.ple_embedding.ngram_embedding.shard_" in name
        and name.endswith(".weight")
    )


def enable_tqdm(use_tqdm_on_load: bool):
''',
        "weight skip helper",
    )
    source = replace_once(
        source,
        '''                if (allowed_names is None or name in allowed_names) and not should_skip_weight(
                    name, local_expert_ids
                ):
''',
        '''                if (
                    (allowed_names is None or name in allowed_names)
                    and not _qwen_should_skip_nvme_ple(name)
                    and not should_skip_weight(name, local_expert_ids)
                ):
''',
        "eager PLE filter",
    )
    old = '''                        (allowed_names is not None and name not in allowed_names)
                        or should_skip_weight(name, local_expert_ids)
'''
    new = '''                        (allowed_names is not None and name not in allowed_names)
                        or _qwen_should_skip_nvme_ple(name)
                        or should_skip_weight(name, local_expert_ids)
'''
    if source.count(old) != 2:
        raise ValueError(
            "expected the torchao and standard safetensors filters exactly twice"
        )
    return source.replace(old, new)


def patch_base_loader(source: str) -> str:
    if MARKER in source:
        raise ValueError("base_loader.py is already single-Spark patched")
    source = replace_once(
        source,
        '''from abc import ABC, abstractmethod

import torch
''',
        '''from abc import ABC, abstractmethod
import gc
import os

import torch
''',
        "base-loader cleanup imports",
    )
    return replace_once(
        source,
        '''            self.load_weights(model, model_config)

            # Log peak GPU memory after loading weights. This is needed
''',
        f'''            self.load_weights(model, model_config)

            # {MARKER}. Standard safetensors can leave the final large shard's
            # staging allocation in the CUDA caching allocator. On GB10 that
            # cache consumes the same physical LPDDR needed by post-processing.
            if os.environ.get("QWEN_SINGLE_SPARK_AGGRESSIVE_CLEANUP") == "1":
                torch.accelerator.synchronize(target_device)
                gc.collect()
                torch.accelerator.empty_cache()
                logger.info(
                    "Released loader staging cache before weight post-processing."
                )

            # Log peak GPU memory after loading weights. This is needed
''',
        "pre-postprocess cleanup gate",
    )


def patch_loader_utils(source: str) -> str:
    if MARKER in source:
        raise ValueError("model-loader utils.py is already single-Spark patched")
    source = replace_once(
        source,
        '''import inspect
import warnings
''',
        '''import gc
import inspect
import os
import time
import warnings
''',
        "loader-utils cleanup imports",
    )
    source = replace_once(
        source,
        '''def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    for _, module in model.named_modules():
''',
        f'''def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    # {MARKER}. The correctness-first fallback ("all") performs a full GC and
    # cache release after every quantized module. The optimized TP=1 path
    # ("moe") releases only after large routed-expert modules and relies on
    # Python reference counting for their temporary tensors. "pressure" keeps
    # upstream UMA pressure-based behavior.
    cleanup_mode = os.environ.get(
        "QWEN_SINGLE_SPARK_POSTPROCESS_CLEANUP",
        "all"
        if os.environ.get("QWEN_SINGLE_SPARK_AGGRESSIVE_CLEANUP") == "1"
        else "pressure",
    )
    if cleanup_mode not in {{"all", "moe", "pressure"}}:
        raise ValueError(
            "QWEN_SINGLE_SPARK_POSTPROCESS_CLEANUP must be "
            "all, moe, or pressure"
        )
    forced_cleanup_runs = 0
    forced_cleanup_seconds = 0.0

    for _, module in model.named_modules():
''',
        "postprocess cleanup policy initialization",
    )
    source = replace_once(
        source,
        '''            # Repacking transients above can leave large amounts of memory in
            # the caching allocator, which starves the OS on UMA devices.
            release_device_memory_under_pressure(target_device)
''',
        f'''            # Repacking transients above can leave large amounts of memory in
            # the caching allocator, which starves the OS on UMA devices.
            # {MARKER}. Retain-12 TP=1 cannot allow adjacent large expert
            # repacks to coexist, but global GC after every small quantized
            # projection made startup several minutes slower. Detect the routed
            # expert containers by their two canonical packed weights.
            is_routed_expert = hasattr(module, "w13_weight") and hasattr(
                module, "w2_weight"
            )
            force_cleanup = cleanup_mode == "all" or (
                cleanup_mode == "moe" and is_routed_expert
            )
            if force_cleanup:
                cleanup_started = time.perf_counter()
                torch.accelerator.synchronize(target_device)
                if cleanup_mode == "all":
                    gc.collect()
                torch.accelerator.empty_cache()
                forced_cleanup_runs += 1
                forced_cleanup_seconds += time.perf_counter() - cleanup_started
            else:
                release_device_memory_under_pressure(target_device)
''',
        "per-module selective cleanup gate",
    )
    return replace_once(
        source,
        '''
    # Initialize post-load attention weights for any attention layer and MM
''',
        '''
    logger.info(
        "Single-Spark postprocess cleanup mode=%s runs=%d elapsed=%.2fs",
        cleanup_mode,
        forced_cleanup_runs,
        forced_cleanup_seconds,
    )

    # Initialize post-load attention weights for any attention layer and MM
''',
        "postprocess cleanup summary",
    )


def patch_uniproc_executor(source: str) -> str:
    if MARKER in source:
        raise ValueError("uniproc_executor.py is already NVMe PLE patched")
    return replace_once(
        source,
        '''        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.driver_worker.elastic_ep_execute("load_model")
        else:
            self.driver_worker.load_model()
        current_platform.update_block_size_for_backend(self.vllm_config)
''',
        f'''        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        # {MARKER}. MultiProcExecutor starts the node-local PLE worker before
        # model loading and waits for its READY handshake afterwards. TP=1 uses
        # UniProcExecutor, which omitted both calls and could wait forever on
        # the first PLE CUDA semaphore during kernel warmup.
        if envs.VLLM_PLE_CPU_OFFLOAD:
            self.driver_worker.spawn_ple_offload()
            from vllm.v1.ple_offload.worker import PleOffloadWorker

            PleOffloadWorker.wait_for_initialized(
                self.driver_worker._ple_offload_worker_handle
            )

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.driver_worker.elastic_ep_execute("load_model")
        else:
            self.driver_worker.load_model()

        if envs.VLLM_PLE_CPU_OFFLOAD:
            self.driver_worker.wait_ple_offload_ready()
        current_platform.update_block_size_for_backend(self.vllm_config)
''',
        "UniProc PLE worker lifecycle",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    site = Path("/usr/local/lib/python3.12/dist-packages")
    parser.add_argument(
        "--ple-layer",
        type=Path,
        default=site / "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py",
    )
    parser.add_argument(
        "--worker",
        type=Path,
        default=site / "vllm/v1/ple_offload/worker.py",
    )
    parser.add_argument(
        "--weight-utils",
        type=Path,
        default=site / "vllm/model_executor/model_loader/weight_utils.py",
    )
    parser.add_argument(
        "--base-loader",
        type=Path,
        default=site / "vllm/model_executor/model_loader/base_loader.py",
    )
    parser.add_argument(
        "--loader-utils",
        type=Path,
        default=site / "vllm/model_executor/model_loader/utils.py",
    )
    parser.add_argument(
        "--uniproc-executor",
        type=Path,
        default=site / "vllm/v1/executor/uniproc_executor.py",
    )
    parser.add_argument(
        "--nvme-source",
        type=Path,
        default=Path("src/qwen_mxfp4/ple_nvme.py"),
    )
    parser.add_argument(
        "--nvme-destination",
        type=Path,
        default=site / "vllm/v1/ple_offload/nvme_table.py",
    )
    args = parser.parse_args()
    args.ple_layer.write_text(patch_ple_layer(args.ple_layer.read_text()))
    args.worker.write_text(patch_offload_worker(args.worker.read_text()))
    args.weight_utils.write_text(patch_weight_utils(args.weight_utils.read_text()))
    args.base_loader.write_text(patch_base_loader(args.base_loader.read_text()))
    args.loader_utils.write_text(patch_loader_utils(args.loader_utils.read_text()))
    args.uniproc_executor.write_text(
        patch_uniproc_executor(args.uniproc_executor.read_text())
    )
    args.nvme_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.nvme_source, args.nvme_destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
