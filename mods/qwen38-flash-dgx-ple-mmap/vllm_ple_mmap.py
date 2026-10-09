"""vllm_ple_mmap — serve the Qwen3.8-Flash-Next N-gram (PLE) table from NVMe via mmap.

Why: the 51B-parameter n-gram table is 44 GiB in FP8 and vLLM keeps it resident
(GPU, or pinned host RAM with VLLM_PLE_CPU_OFFLOAD). On a DGX Spark / GX10 the
host and the GPU share one 121 GiB pool, so neither fits next to the 78 GiB main
model. But a token only ever touches 16 rows x 160 bytes of that table, so the
table can live on disk and be served through the page cache — exactly what
llama.cpp does with its GGUF mmap.

How: with VLLM_PLE_MMAP=1 this module patches ``Qwen3_8FlashNextNGramEmbedding``:
  * ``__init__`` swaps the 44/95 GiB ``VocabParallelEmbedding`` for a tiny
    placeholder whose ``forward(ids)`` gathers rows from ``np.memmap`` views of the
    checkpoint's ``model-plefp8-*.safetensors`` shards (zero-copy, page-cache backed);
  * ``load_weights`` drops the 128 shard tensors on the floor, keeps the global FP8
    ``weight_scale`` (as ``_offload_weight_scale``, which the untouched
    ``Qwen3_8FlashNextPLELayer._dequantize_embeddings`` already consumes) and opens
    the memmaps.
  * ``forward_impl`` (hashing + lookup) is wrapped in a custom op
    ``vllm::ple_mmap_lookup`` so that (a) torch.compile treats it as opaque — the
    stock version trips an Inductor int64 indexing assert on sm_121 — and (b) it can
    be listed in ``-cc.splitting_ops`` and run OUTSIDE piecewise CUDA graphs: the
    gather is CPU work + a pageable H2D copy, which cannot live inside a capture.
    Use ``-cc.cudagraph_mode=PIECEWISE`` (not FULL*) with the splitting op list in
    serve-flashnext-vllm.sh, or ``--enforce-eager``.
Nothing else in vLLM changes: the n-gram hashing, the short-conv, the dequant path
are the stock ones.

Knobs (env):
  VLLM_PLE_MMAP=1            enable
  VLLM_PLE_MMAP_WORKERS=32   gather threads (page faults overlap across threads)
  VLLM_PLE_MMAP_CHUNK=2048   rows per gather task
  VLLM_PLE_MMAP_PREWARM=0    1 = stream the whole table once at load to fill the
                             page cache with whatever memory is free (harmless,
                             evictable; ~10 s at 4.7 GB/s)

Install: the Dockerfile copies this file next to vllm and appends
``_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)`` to the end of
``vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py``. See the repo README.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("vllm.ple_mmap")

ENV_ENABLE = "VLLM_PLE_MMAP"
_FP8_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0").lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# safetensors header parsing (no dependency on the safetensors package: we need
# raw file offsets, which its Python API does not expose)
# --------------------------------------------------------------------------- #
def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) of a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


class MmapPleTable:
    """Row gather over a table split into ``split_ngram_parts`` shard files.

    ``shards``: {shard_index: (path, absolute_byte_offset, rows)}. Shard ``i``
    holds global rows ``[i*shard_size, i*shard_size + rows)`` (vLLM's
    ``copy_ple_embedding_shard_`` layout).
    """

    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int = 32,
        chunk: int = 2048,
    ) -> None:
        if not shards:
            raise ValueError("no PLE shards")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.chunk = max(1, int(chunk))
        self.paths: list[str | None] = [None] * (max(shards) + 1)
        self.mm: list[np.memmap | None] = [None] * (max(shards) + 1)
        self.rows_total = 0
        for idx, (path, offset, rows) in shards.items():
            self.paths[idx] = path
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            self.rows_total += rows
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [N] global row ids -> uint8 [N, row_bytes] (a fresh array)."""
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        # Dedupe + sort: repeated n-grams are common, and sorted rows improve
        # locality inside a shard.
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.shard_size * len(self.mm):
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            si = int(shard[s])
            for c in range(s, e, self.chunk):
                tasks.append((si, c, min(c + self.chunk, e)))

        def run(task: tuple[int, int, int]) -> None:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE shard {si} missing")
            # Fancy indexing on a memmap: page faults do the I/O; NumPy releases
            # the GIL for the copy, so tasks overlap across threads.
            out[a:b] = mm[local[a:b]]

        if len(tasks) == 1:
            run(tasks[0])
        else:
            for _ in self.pool.map(run, tasks):
                pass
        return out[inverse]

    def prewarm(self) -> None:
        """Stream every shard once so the page cache holds as much as it can."""
        block = 64 << 20
        for path, mm in zip(self.paths, self.mm):
            if path is None or mm is None:
                continue
            start = mm.offset
            end = start + mm.shape[0] * mm.shape[1]
            with open(path, "rb", buffering=0) as f:
                pos = start
                while pos < end:
                    n = f.readinto(bytearray(min(block, end - pos)))  # noqa: F841
                    if not n:
                        break
                    pos += n


# --------------------------------------------------------------------------- #
# Placeholder that stands in for VocabParallelEmbedding
# --------------------------------------------------------------------------- #
class _MmapNgramEmbedding(nn.Module):
    """Duck-types the bits of VocabParallelEmbedding the PLE code reads.

    No ``weight`` attribute on purpose: ``Qwen3_8FlashNextPLELayer`` then falls
    back to ``ple_embedding._offload_weight_scale`` for the FP8 scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.table: MmapPleTable | None = None
        self._zeros_dtype = torch.bfloat16

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.table
        if table is None:
            # Weights never loaded (e.g. --load-format dummy): keep the plumbing
            # alive with zeros so kernel tests can run without the 44 GiB table.
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self._zeros_dtype,
                device=ids.device,
            )
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        rows = table.gather(ids_np)  # uint8 [N, row_bytes], fresh & writable
        out = torch.from_numpy(rows).view(table.torch_dtype)
        out = out.to(ids.device, non_blocking=True)
        return out.reshape(*ids.shape, self.embedding_dim)


# --------------------------------------------------------------------------- #
# Patch
# --------------------------------------------------------------------------- #
def _find_shards(
    model_path: str, layer_idx: int
) -> tuple[dict[int, tuple[str, int, int]], str | None, tuple[str, int, int] | None]:
    """Locate ``layers.<idx>.ple.ple_embedding.ngram_embedding.shard_N.weight``.

    Returns (shards, dtype_str, scale_entry) where scale_entry is
    (path, abs_offset, nbytes) of ``ngram_embedding.weight_scale`` or None.
    """
    shard_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
    )
    scale_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(
            {
                os.path.join(model_path, fn)
                for name, fn in weight_map.items()
                if shard_re.search(name) or scale_re.search(name)
            }
        )
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    shards: dict[int, tuple[str, int, int]] = {}
    dtype_str: str | None = None
    scale_entry: tuple[str, int, int] | None = None
    for path in files:
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            m = shard_re.search(name)
            if m:
                start, end = meta["data_offsets"]
                rows, cols = meta["shape"]
                if dtype_str is None:
                    dtype_str = meta["dtype"]
                elif meta["dtype"] != dtype_str:
                    raise ValueError("PLE shards have mixed dtypes")
                if end - start != rows * cols * _itemsize(dtype_str):
                    raise ValueError(f"PLE shard {name}: size/shape mismatch")
                shards[int(m.group(1))] = (path, data_start + start, rows)
                shard_cols = cols
            elif scale_re.search(name):
                start, end = meta["data_offsets"]
                scale_entry = (path, data_start + start, end - start, meta["dtype"])  # type: ignore[assignment]
    if shards:
        # Return cols through dtype_str consumer; keep it simple: stash on dict.
        shards["__cols__"] = shard_cols  # type: ignore[index]
    return shards, dtype_str, scale_entry


def _itemsize(dtype_str: str) -> int:
    return {
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "U8": 1,
        "I8": 1,
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }[dtype_str]


def _read_scale(entry: tuple) -> torch.Tensor:
    path, offset, nbytes, dtype_str = entry
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if dtype_str == "F32":
        return torch.tensor(struct.unpack("<f", raw[:4])[0], dtype=torch.float32)
    if dtype_str == "BF16":
        u16 = struct.unpack("<H", raw[:2])[0]
        return torch.tensor(u16 << 16, dtype=torch.int32).view(torch.float32).squeeze()
    if dtype_str == "F16":
        return torch.frombuffer(bytearray(raw[:2]), dtype=torch.float16).clone().squeeze()
    raise ValueError(f"unsupported weight_scale dtype {dtype_str}")


_REGISTRY: dict[str, nn.Module] = {}
_OP_NAME = "ple_mmap_lookup"


def _lookup_impl(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = _REGISTRY[layer_name]
    result = layer._ple_mmap_orig_forward_impl(
        None, input_ids, query_start_loc, ngram_context
    )
    output[: result.shape[0]].copy_(result.to(output.dtype))


def _lookup_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _register_op() -> None:
    if hasattr(torch.ops.vllm, _OP_NAME):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_lookup_impl,
        mutates_args=["output"],
        fake_impl=_lookup_fake,
    )


def apply(cls: type) -> None:
    """Patch ``Qwen3_8FlashNextNGramEmbedding`` (pass the class) when enabled."""
    if not enabled():
        return
    if getattr(cls, "_ple_mmap_patched", False):
        return
    mod = sys.modules[cls.__module__]
    orig_init = cls.__init__
    orig_load_weights = cls.load_weights

    def __init__(self, config, embedding_dim, ple_dense_layer_id, max_total_tokens,
                 max_num_reqs, prefix, quant_config=None, params_dtype=None):
        # Run the stock constructor (hash buffers, workspaces, ...) with the
        # embedding class swapped for our placeholder so nothing large is
        # allocated. quant_config=None keeps the stock code from selecting an
        # FP8 quant method that would create an FP8 weight parameter.
        real_embedding_cls = mod.VocabParallelEmbedding
        mod.VocabParallelEmbedding = lambda n, d, **_kw: _MmapNgramEmbedding(n, d)
        try:
            orig_init(self, config, embedding_dim, ple_dense_layer_id,
                      max_total_tokens, max_num_reqs, prefix,
                      quant_config=None, params_dtype=params_dtype)
        finally:
            mod.VocabParallelEmbedding = real_embedding_cls
        self._ple_mmap_prefix = prefix
        _REGISTRY[prefix] = self
        self._ple_mmap_model_path = None
        try:
            from vllm.config import get_current_vllm_config
            self._ple_mmap_model_path = get_current_vllm_config().model_config.model
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("PLE mmap: cannot read model path from vllm config: %s", exc)
        if params_dtype is not None:
            self.ngram_embedding._zeros_dtype = params_dtype
        logger.info(
            "PLE mmap: %s -> placeholder embedding (%d rows x %d), table will be mmapped",
            prefix, self.ngram_embedding.org_vocab_size, self.head_dim,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        rest: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                loaded.add(name)  # served from disk, never materialised
                continue
            if name == "ngram_embedding.weight_scale":
                self.register_buffer(
                    "_offload_weight_scale",
                    w.detach().to(device=torch.accelerator.current_accelerator()),
                    persistent=False,
                )
                loaded.add(name)
                continue
            rest.append((name, w))
        loaded.update(orig_load_weights(self, rest))
        _setup_table(self)
        return loaded

    def _setup_table(self) -> None:
        if self.ngram_embedding.table is not None:
            return
        model_path = self._ple_mmap_model_path
        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                f"PLE mmap: model path {model_path!r} is not a local directory; "
                "point --model at the downloaded snapshot"
            )
        m = re.search(r"layers\.(\d+)\.", self._ple_mmap_prefix)
        if not m:
            raise RuntimeError(f"PLE mmap: cannot find layer index in {self._ple_mmap_prefix!r}")
        layer_idx = int(m.group(1))
        shards, dtype_str, scale_entry = _find_shards(model_path, layer_idx)
        if not shards:
            raise RuntimeError(f"PLE mmap: no shard tensors for layer {layer_idx} under {model_path}")
        cols = shards.pop("__cols__")  # type: ignore[arg-type]
        if cols != self.head_dim:
            raise RuntimeError(f"PLE mmap: shard width {cols} != head_dim {self.head_dim}")
        if dtype_str not in _FP8_DTYPES:
            raise RuntimeError(f"PLE mmap: only FP8 shards are supported, got {dtype_str}")
        if not hasattr(self, "_offload_weight_scale"):
            if scale_entry is None:
                raise RuntimeError("PLE mmap: FP8 shards without ngram_embedding.weight_scale")
            self.register_buffer(
                "_offload_weight_scale",
                _read_scale(scale_entry).to(torch.accelerator.current_accelerator()),
                persistent=False,
            )
        parts = int(self.split_ngram_parts)
        vocab = int(self.ngram_embedding.org_vocab_size)
        shard_size = math.ceil(vocab / parts)
        for idx, (_p, _o, rows) in shards.items():
            expected = max(0, min(shard_size, vocab - idx * shard_size))
            if rows != expected:
                raise RuntimeError(
                    f"PLE mmap: shard {idx} has {rows} rows, expected {expected}"
                )
        table = MmapPleTable(
            shards, shard_size, cols, _FP8_DTYPES[dtype_str],
            workers=_env_int("VLLM_PLE_MMAP_WORKERS", 32),
            chunk=_env_int("VLLM_PLE_MMAP_CHUNK", 2048),
        )
        if _env_int("VLLM_PLE_MMAP_PREWARM", 0):
            logger.info("PLE mmap: prewarming page cache (%.1f GiB)...", table.rows_total * cols / 2**30)
            table.prewarm()
        self.ngram_embedding.table = table
        logger.info(
            "PLE mmap: layer %d, %d shards, %d rows x %d B (%.1f GiB on disk), dtype %s, %d workers",
            layer_idx, len(shards), table.rows_total, cols,
            table.rows_total * cols / 2**30, dtype_str, table.pool._max_workers,
        )

    def forward_impl(self, hidden_states, input_ids, query_start_loc, ngram_context,
                     output_buffer=None):
        del hidden_states, output_buffer
        num_tokens = input_ids.reshape(-1).shape[0]
        table = self.ngram_embedding.table
        dtype = table.torch_dtype if table is not None else self.ngram_embedding._zeros_dtype
        output = torch.empty(
            (num_tokens, self.embedding_dim), dtype=dtype, device=input_ids.device
        )
        getattr(torch.ops.vllm, _OP_NAME)(
            input_ids, query_start_loc, ngram_context, output, self._ple_mmap_prefix
        )
        return output

    _register_op()
    cls._ple_mmap_orig_forward_impl = cls.forward_impl
    cls.forward_impl = forward_impl
    cls.__init__ = __init__
    cls.load_weights = load_weights
    cls._setup_table = _setup_table
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch applied to %s.%s", cls.__module__, cls.__name__)
