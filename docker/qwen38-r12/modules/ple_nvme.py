"""Read-only NVMe-backed access to Qwen3.8 PLE safetensor shards.

The production PLE table is FP8, but lookup does not require interpreting the
floating-point values.  This module therefore keeps rows as their exact raw
bytes and leaves dequantization to the existing vLLM PLE path.
"""

from __future__ import annotations

import bisect
import concurrent.futures
import json
import mmap
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_PLE_PREFIX: Final = (
    "model.language_model.layers.1.ple.ple_embedding"
)
_SAFETENSORS_MAX_HEADER_BYTES: Final = 128 * 1024 * 1024
_PLE_ROW_TRACE_MAGIC: Final = b"QPLEROW1"
_PLE_ROW_TRACE_RECORD: Final = struct.Struct("<QII")
_PLE_REQUEST_TRACE_MAGIC: Final = b"QPLEREQ1"
_PLE_REQUEST_TRACE_RECORD: Final = struct.Struct("<QI")
_ITEM_SIZES: Final = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_NUMPY_DTYPES: Final = {
    "BOOL": np.dtype("?"),
    "U8": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "I16": np.dtype("<i2"),
    "U16": np.dtype("<u2"),
    "I32": np.dtype("<i4"),
    "U32": np.dtype("<u4"),
    "I64": np.dtype("<i8"),
    "U64": np.dtype("<u8"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
}


@dataclass(frozen=True)
class SafetensorSlice:
    """Validated location of one tensor payload within a safetensors file."""

    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    byte_start: int
    byte_end: int

    @property
    def item_size(self) -> int:
        return _ITEM_SIZES[self.dtype]

    @property
    def payload_bytes(self) -> int:
        return self.byte_end - self.byte_start


@dataclass(frozen=True)
class PleShard:
    """One checkpoint PLE shard and its logical global-row interval."""

    shard_index: int
    tensor: SafetensorSlice
    global_row_start: int
    global_row_end: int
    row_width: int

    @property
    def rows(self) -> int:
        return self.global_row_end - self.global_row_start

    @property
    def row_bytes(self) -> int:
        return self.row_width * self.tensor.item_size


@dataclass(frozen=True)
class QwenPleHashParameters:
    """Small persistent tensors needed to reproduce Qwen's PLE row IDs."""

    layer_multipliers: NDArray[np.int64]
    head_vocab_sizes: NDArray[np.int64]
    head_offsets: NDArray[np.int64]


def _decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def filesystem_type(path: Path) -> str | None:
    """Return Linux's best matching mount type, or ``None`` off Linux."""

    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.exists():
        return None
    resolved = path.resolve()
    candidates: list[tuple[int, str]] = []
    for line in mountinfo.read_text().splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        right_fields = right.split()
        if len(fields) < 5 or not right_fields:
            continue
        mount_point = Path(_decode_mount_path(fields[4]))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidates.append((len(str(mount_point)), right_fields[0]))
    return max(candidates, default=(0, None))[1]


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, object]]:
    file_size = path.stat().st_size
    with path.open("rb", buffering=0) as handle:
        raw_length = os.pread(handle.fileno(), 8, 0)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if not 2 <= header_length <= _SAFETENSORS_MAX_HEADER_BYTES:
            raise ValueError(
                f"invalid safetensors header length {header_length}: {path}"
            )
        header_raw = os.pread(handle.fileno(), header_length, 8)
    if len(header_raw) != header_length:
        raise ValueError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header: {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    data_start = 8 + header_length
    if data_start > file_size:
        raise ValueError(f"safetensors data starts past EOF: {path}")
    return data_start, header


def resolve_safetensor_slice(path: Path, tensor_name: str) -> SafetensorSlice:
    """Resolve and validate one tensor's exact byte interval."""

    data_start, header = _read_safetensors_header(path)
    entry = header.get(tensor_name)
    if not isinstance(entry, dict):
        raise KeyError(f"tensor {tensor_name!r} is absent from {path}")
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if dtype not in _ITEM_SIZES:
        raise ValueError(f"unsupported safetensors dtype {dtype!r}: {tensor_name}")
    if (
        not isinstance(shape, list)
        or not all(isinstance(value, int) and value >= 0 for value in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) and value >= 0 for value in offsets)
    ):
        raise ValueError(f"invalid safetensors metadata for {tensor_name}")
    relative_start, relative_end = offsets
    if relative_start > relative_end:
        raise ValueError(f"reversed data offsets for {tensor_name}")
    element_count = int(np.prod(shape, dtype=np.int64))
    expected_bytes = element_count * _ITEM_SIZES[dtype]
    if relative_end - relative_start != expected_bytes:
        raise ValueError(
            f"payload size mismatch for {tensor_name}: "
            f"{relative_end - relative_start} != {expected_bytes}"
        )
    byte_start = data_start + relative_start
    byte_end = data_start + relative_end
    if byte_end > path.stat().st_size:
        raise ValueError(f"tensor {tensor_name} extends past EOF in {path}")
    return SafetensorSlice(
        path=path,
        name=tensor_name,
        dtype=dtype,
        shape=tuple(shape),
        byte_start=byte_start,
        byte_end=byte_end,
    )


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    value = json.loads(index_path.read_text())
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        raise ValueError(f"invalid weight_map in {index_path}")
    return weight_map


def read_indexed_tensor(
    model_dir: Path,
    tensor_name: str,
    *,
    weight_map: dict[str, str] | None = None,
) -> NDArray[np.generic]:
    """Read one small indexed tensor without materializing sibling tensors."""

    mapping = weight_map if weight_map is not None else _load_weight_map(model_dir)
    filename = mapping.get(tensor_name)
    if filename is None:
        raise KeyError(f"tensor {tensor_name!r} is absent from the checkpoint index")
    tensor = resolve_safetensor_slice(model_dir / filename, tensor_name)
    numpy_dtype = _NUMPY_DTYPES.get(tensor.dtype)
    if numpy_dtype is None:
        raise ValueError(
            f"tensor {tensor_name} uses {tensor.dtype}, which has no NumPy decoder"
        )
    with tensor.path.open("rb", buffering=0) as handle:
        payload = os.pread(handle.fileno(), tensor.payload_bytes, tensor.byte_start)
    if len(payload) != tensor.payload_bytes:
        raise ValueError(f"short read for tensor {tensor_name}")
    return np.frombuffer(payload, dtype=numpy_dtype).reshape(tensor.shape).copy()


def load_qwen_ple_hash_parameters(
    model_dir: Path,
    *,
    prefix: str = DEFAULT_PLE_PREFIX,
) -> QwenPleHashParameters:
    """Load only the three tiny tensors used by Qwen's n-gram hash."""

    weight_map = _load_weight_map(model_dir)
    values = [
        read_indexed_tensor(
            model_dir, f"{prefix}.{suffix}", weight_map=weight_map
        ).astype(np.int64, copy=False)
        for suffix in (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
        )
    ]
    return QwenPleHashParameters(*values)


class PleMmapTable:
    """Read-only mmap view over all checkpoint PLE table shards."""

    def __init__(self, shards: tuple[PleShard, ...]) -> None:
        if not shards:
            raise ValueError("at least one PLE shard is required")
        self.shards = shards
        self.embedding_dim = shards[0].row_width
        self.row_bytes = shards[0].row_bytes
        self.total_rows = shards[-1].global_row_end
        self.payload_bytes = sum(shard.tensor.payload_bytes for shard in shards)
        self._ends = [shard.global_row_end for shard in shards]
        self._ends_array = np.asarray(self._ends, dtype=np.int64)
        self._handles: dict[Path, object] = {}
        self._maps: dict[Path, mmap.mmap] = {}
        self._views: list[NDArray[np.uint8]] = []
        self.last_pread_stats: dict[str, int] | None = None
        self._pread_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pread_executor_workers = 0
        cache_rows_text = os.environ.get("VLLM_PLE_NVME_ROW_CACHE_ROWS", "0")
        try:
            self._row_cache_rows = int(cache_rows_text)
        except ValueError as error:
            raise ValueError(
                "VLLM_PLE_NVME_ROW_CACHE_ROWS must be an integer"
            ) from error
        if self._row_cache_rows < 0:
            raise ValueError("VLLM_PLE_NVME_ROW_CACHE_ROWS must be non-negative")
        self._row_cache_tags: NDArray[np.int64] | None = None
        self._row_cache_data: NDArray[np.uint8] | None = None
        self.pread_cache_stats = {"hits": 0, "misses": 0}
        if self._row_cache_rows:
            self._row_cache_tags = np.full(
                self._row_cache_rows, -1, dtype=np.int64
            )
            self._row_cache_data = np.empty(
                (self._row_cache_rows, self.row_bytes), dtype=np.uint8
            )
        self._row_trace_call = 0
        self._row_trace_handle = None
        row_trace_path = os.environ.get("QWEN_PLE_ROW_TRACE_PATH")
        if row_trace_path:
            trace_path = Path(row_trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._row_trace_handle = trace_path.open("wb", buffering=0)
            self._row_trace_handle.write(_PLE_ROW_TRACE_MAGIC)
        self._request_trace_call = 0
        self._request_trace_handle = None
        request_trace_path = os.environ.get("QWEN_PLE_REQUEST_TRACE_PATH")
        if request_trace_path:
            trace_path = Path(request_trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._request_trace_handle = trace_path.open("xb", buffering=0)
            self._request_trace_handle.write(_PLE_REQUEST_TRACE_MAGIC)
        for shard in shards:
            if shard.row_width != self.embedding_dim or shard.row_bytes != self.row_bytes:
                raise ValueError("PLE shards disagree on row layout")
            mapped = self._maps.get(shard.tensor.path)
            if mapped is None:
                handle = shard.tensor.path.open("rb", buffering=0)
                mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                self._handles[shard.tensor.path] = handle
                self._maps[shard.tensor.path] = mapped
            view = np.ndarray(
                shape=(shard.rows, shard.row_bytes),
                dtype=np.uint8,
                buffer=mapped,
                offset=shard.tensor.byte_start,
            )
            view.flags.writeable = False
            self._views.append(view)

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: Path,
        *,
        prefix: str = DEFAULT_PLE_PREFIX,
        expected_parts: int | None = 128,
        expected_embedding_dim: int | None = 160,
        expected_dtype: str | None = "F8_E4M3",
        require_local_filesystem: bool = False,
    ) -> "PleMmapTable":
        """Build a fully validated descriptor without touching tensor pages."""

        model_dir = model_dir.resolve()
        if require_local_filesystem:
            fs_type = filesystem_type(model_dir)
            if fs_type is None:
                raise ValueError("cannot verify the checkpoint filesystem type")
            if fs_type.startswith("nfs") or fs_type in {"cifs", "smb3", "fuse.sshfs"}:
                raise ValueError(
                    f"PLE production backing must be local, found filesystem {fs_type}"
                )
        weight_map = _load_weight_map(model_dir)
        pattern = re.compile(
            rf"^{re.escape(prefix)}\.ngram_embedding\.shard_(\d+)\.weight$"
        )
        indexed: list[tuple[int, str, str]] = []
        for name, filename in weight_map.items():
            match = pattern.match(name)
            if match:
                indexed.append((int(match.group(1)), name, filename))
        indexed.sort()
        indices = [item[0] for item in indexed]
        if expected_parts is not None and len(indexed) != expected_parts:
            raise ValueError(
                f"expected {expected_parts} PLE shards, found {len(indexed)}"
            )
        if indices != list(range(len(indices))):
            raise ValueError(f"PLE shard indices are not contiguous: {indices}")

        shards: list[PleShard] = []
        global_row_start = 0
        for shard_index, name, filename in indexed:
            tensor = resolve_safetensor_slice(model_dir / filename, name)
            if tensor.dtype != expected_dtype and expected_dtype is not None:
                raise ValueError(
                    f"PLE shard {shard_index} dtype is {tensor.dtype}, "
                    f"expected {expected_dtype}"
                )
            if len(tensor.shape) != 2:
                raise ValueError(
                    f"PLE shard {shard_index} must be rank two, got {tensor.shape}"
                )
            rows, row_width = tensor.shape
            if expected_embedding_dim is not None and row_width != expected_embedding_dim:
                raise ValueError(
                    f"PLE shard {shard_index} width is {row_width}, "
                    f"expected {expected_embedding_dim}"
                )
            shards.append(
                PleShard(
                    shard_index=shard_index,
                    tensor=tensor,
                    global_row_start=global_row_start,
                    global_row_end=global_row_start + rows,
                    row_width=row_width,
                )
            )
            global_row_start += rows
        return cls(tuple(shards))

    def close(self) -> None:
        if self._pread_executor is not None:
            self._pread_executor.shutdown(wait=True, cancel_futures=True)
            self._pread_executor = None
            self._pread_executor_workers = 0
        self._views.clear()
        for mapped in self._maps.values():
            mapped.close()
        self._maps.clear()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._row_cache_tags = None
        self._row_cache_data = None
        if self._row_trace_handle is not None:
            self._row_trace_handle.close()
            self._row_trace_handle = None
        if self._request_trace_handle is not None:
            self._request_trace_handle.close()
            self._request_trace_handle = None

    def __enter__(self) -> "PleMmapTable":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def gather(
        self,
        row_ids: ArrayLike,
        *,
        deduplicate: bool = True,
    ) -> NDArray[np.uint8]:
        """Gather exact FP8 row bytes while preserving the input ID shape."""

        ids = np.asarray(row_ids, dtype=np.int64)
        flat = ids.reshape(-1)
        if np.any(flat < 0) or np.any(flat >= self.total_rows):
            raise IndexError(
                f"PLE row ID outside [0, {self.total_rows}): "
                f"min={flat.min(initial=0)}, max={flat.max(initial=0)}"
            )
        if self._request_trace_handle is not None:
            self._trace_requested_rows(flat)
        if deduplicate:
            unique, inverse = np.unique(flat, return_inverse=True)
            unique_rows = self._gather_flat(unique)
            result = unique_rows[inverse]
        else:
            result = self._gather_flat(flat)
        return result.reshape(*ids.shape, self.row_bytes)

    def _gather_flat(self, flat_ids: NDArray[np.int64]) -> NDArray[np.uint8]:
        output = np.empty((flat_ids.size, self.row_bytes), dtype=np.uint8)
        if flat_ids.size == 0:
            return output
        shard_ids = np.searchsorted(self._ends_array, flat_ids, side="right")
        for shard_index in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_index)
            shard = self.shards[int(shard_index)]
            local_rows = flat_ids[positions] - shard.global_row_start
            output[positions] = self._views[int(shard_index)][local_rows]
        return output

    def direct_read_row(self, row_id: int) -> bytes:
        """Read one row with ``pread`` as an independent mmap-gather oracle."""

        if not 0 <= row_id < self.total_rows:
            raise IndexError(row_id)
        shard_index = bisect.bisect_right(self._ends, row_id)
        shard = self.shards[shard_index]
        local_row = row_id - shard.global_row_start
        offset = shard.tensor.byte_start + local_row * shard.row_bytes
        handle = self._handles[shard.tensor.path]
        payload = os.pread(handle.fileno(), shard.row_bytes, offset)
        if len(payload) != shard.row_bytes:
            raise OSError(f"short PLE row read at global row {row_id}")
        return payload

    def gather_pread(
        self,
        row_ids: ArrayLike,
        *,
        deduplicate: bool = True,
        coalesce_gap_bytes: int = 0,
        max_span_bytes: int = 1024 * 1024,
        workers: int = 1,
    ) -> NDArray[np.uint8]:
        """Gather rows with bounded ``pread`` spans instead of page faults.

        Sorted neighboring rows in the same physical file are coalesced. Random
        decode rows remain exact 160-byte reads, avoiding mmap readahead at the
        cost of one syscall per unique row until an asynchronous backend is
        introduced.
        """

        if (
            coalesce_gap_bytes < 0
            or max_span_bytes < self.row_bytes
            or workers <= 0
        ):
            raise ValueError("invalid pread coalescing bounds")
        ids = np.asarray(row_ids, dtype=np.int64)
        flat = ids.reshape(-1)
        if np.any(flat < 0) or np.any(flat >= self.total_rows):
            raise IndexError("PLE row ID outside the table")
        if self._request_trace_handle is not None:
            self._trace_requested_rows(flat)
        if deduplicate:
            unique, inverse = np.unique(flat, return_inverse=True)
        else:
            unique = flat.copy()
            inverse = np.arange(flat.size, dtype=np.int64)
        self._trace_pread_rows(unique, flat.size)
        unique_output = np.empty((unique.size, self.row_bytes), dtype=np.uint8)

        cache_hits = np.zeros(unique.size, dtype=np.bool_)
        cache_slots: NDArray[np.int64] | None = None
        if self._row_cache_tags is not None:
            assert self._row_cache_data is not None
            cache_slots = np.remainder(unique, self._row_cache_rows)
            cache_hits = self._row_cache_tags[cache_slots] == unique
            hit_positions = np.flatnonzero(cache_hits)
            unique_output[hit_positions] = self._row_cache_data[
                cache_slots[hit_positions]
            ]
        storage_positions = np.flatnonzero(~cache_hits)
        storage_ids = unique[storage_positions]

        requests_by_path: dict[Path, list[tuple[int, int]]] = {}
        shard_ids = np.searchsorted(self._ends_array, storage_ids, side="right")
        for shard_index in np.unique(shard_ids):
            relative_positions = np.flatnonzero(shard_ids == shard_index)
            positions = storage_positions[relative_positions]
            shard = self.shards[int(shard_index)]
            local_rows = storage_ids[relative_positions] - shard.global_row_start
            offsets = shard.tensor.byte_start + local_rows * shard.row_bytes
            requests_by_path.setdefault(shard.tensor.path, []).extend(
                (int(offset), int(position))
                for offset, position in zip(offsets, positions, strict=True)
            )

        def read_path(
            path: Path, requests: list[tuple[int, int]]
        ) -> tuple[int, int, int]:
            requests.sort()
            groups: list[list[tuple[int, int]]] = []
            for request in requests:
                if not groups:
                    groups.append([request])
                    continue
                start = groups[-1][0][0]
                previous_end = groups[-1][-1][0] + self.row_bytes
                next_end = request[0] + self.row_bytes
                if (
                    request[0] <= previous_end + coalesce_gap_bytes
                    and next_end - start <= max_span_bytes
                ):
                    groups[-1].append(request)
                else:
                    groups.append([request])
            handle = self._handles[path]
            local_read_calls = 0
            local_requested_bytes = 0
            local_span_bytes = 0
            for group in groups:
                start = group[0][0]
                end = group[-1][0] + self.row_bytes
                payload = os.pread(handle.fileno(), end - start, start)
                if len(payload) != end - start:
                    raise OSError(f"short PLE span read in {path}")
                local_read_calls += 1
                local_requested_bytes += len(group) * self.row_bytes
                local_span_bytes += len(payload)
                for offset, output_index in group:
                    relative = offset - start
                    unique_output[output_index] = np.frombuffer(
                        payload,
                        dtype=np.uint8,
                        count=self.row_bytes,
                        offset=relative,
                    )
            return local_read_calls, local_requested_bytes, local_span_bytes

        if workers == 1 or len(requests_by_path) <= 1:
            path_stats = [
                read_path(path, requests)
                for path, requests in requests_by_path.items()
            ]
        else:
            effective_workers = min(workers, len(requests_by_path))
            if self._pread_executor_workers != effective_workers:
                if self._pread_executor is not None:
                    self._pread_executor.shutdown(wait=True, cancel_futures=True)
                self._pread_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=effective_workers,
                    thread_name_prefix="ple-pread",
                )
                self._pread_executor_workers = effective_workers
            assert self._pread_executor is not None
            futures = [
                self._pread_executor.submit(read_path, path, requests)
                for path, requests in requests_by_path.items()
            ]
            path_stats = [future.result() for future in futures]
        read_calls = sum(value[0] for value in path_stats)
        requested_bytes = sum(value[1] for value in path_stats)
        span_bytes = sum(value[2] for value in path_stats)
        cache_hit_count = int(cache_hits.sum())
        cache_miss_count = int(storage_positions.size)
        self.pread_cache_stats["hits"] += cache_hit_count
        self.pread_cache_stats["misses"] += cache_miss_count
        if cache_slots is not None and cache_miss_count:
            assert self._row_cache_tags is not None
            assert self._row_cache_data is not None
            miss_slots = cache_slots[storage_positions]
            # Multiple IDs in one call can map to the same direct-mapped slot.
            # Retain the last sorted ID for a deterministic tag/payload pair.
            _, reversed_first = np.unique(miss_slots[::-1], return_index=True)
            selected = storage_positions[
                miss_slots.size - 1 - reversed_first
            ]
            selected_slots = cache_slots[selected]
            self._row_cache_data[selected_slots] = unique_output[selected]
            self._row_cache_tags[selected_slots] = unique[selected]
        self.last_pread_stats = {
            "read_calls": read_calls,
            "requested_bytes": requested_bytes,
            "span_bytes": span_bytes,
            "unique_rows": int(unique.size),
        }
        if self._row_cache_rows:
            self.last_pread_stats.update(
                {
                    "cache_capacity_rows": self._row_cache_rows,
                    "cache_hits": cache_hit_count,
                    "cache_misses": cache_miss_count,
                    "cache_hits_total": self.pread_cache_stats["hits"],
                    "cache_misses_total": self.pread_cache_stats["misses"],
                }
            )
        result = unique_output[inverse] if deduplicate else unique_output
        return result.reshape(*ids.shape, self.row_bytes)

    def _trace_pread_rows(
        self,
        unique: NDArray[np.int64],
        requested_rows: int,
    ) -> None:
        """Record exact post-dedup lookup IDs for offline cache simulation.

        This opt-in diagnostic writes to one unbuffered binary stream so no
        row payloads or model data are copied.  Production runs leave the
        environment variable unset and therefore pay only one predictable
        branch per PLE lookup.
        """

        handle = self._row_trace_handle
        if handle is None:
            return
        row_ids = np.asarray(unique, dtype="<i8")
        header = _PLE_ROW_TRACE_RECORD.pack(
            self._row_trace_call,
            requested_rows,
            int(row_ids.size),
        )
        written = os.writev(handle.fileno(), (header, memoryview(row_ids)))
        expected = len(header) + row_ids.nbytes
        if written != expected:
            raise OSError(f"short PLE row-trace write: {written} != {expected}")
        self._row_trace_call += 1

    def _trace_requested_rows(self, flat: NDArray[np.int64]) -> None:
        """Record ordered pre-dedup row IDs for rollback correctness audits."""

        handle = self._request_trace_handle
        if handle is None:
            return
        row_ids = np.asarray(flat, dtype="<i8")
        header = _PLE_REQUEST_TRACE_RECORD.pack(
            self._request_trace_call,
            int(row_ids.size),
        )
        written = os.writev(handle.fileno(), (header, memoryview(row_ids)))
        expected = len(header) + row_ids.nbytes
        if written != expected:
            raise OSError(
                f"short PLE request-trace write: {written} != {expected}"
            )
        self._request_trace_call += 1

    def advise_dontneed(self) -> None:
        """Ask Linux to discard this process's file-backed PLE pages."""

        advice = getattr(mmap, "MADV_DONTNEED", None)
        if advice is None:
            raise RuntimeError("mmap MADV_DONTNEED is unavailable")
        for mapped in self._maps.values():
            mapped.madvise(advice)


def qwen_ple_ngram_row_ids(
    input_ids: ArrayLike,
    query_start_loc: ArrayLike,
    ngram_context: ArrayLike,
    *,
    eos_token_id: int,
    parameters: QwenPleHashParameters,
    ngram_size: int = 3,
    heads_per_ngram: int = 8,
) -> NDArray[np.int64]:
    """Reproduce vLLM/Qwen's exact bigram/trigram PLE row IDs on CPU."""

    inputs = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    starts = np.asarray(query_start_loc, dtype=np.int64).reshape(-1)
    contexts = np.asarray(ngram_context, dtype=np.int64)
    if starts.size < 2 or starts[0] != 0 or starts[-1] != inputs.size:
        raise ValueError("query_start_loc must span the flattened input IDs")
    if np.any(starts[1:] < starts[:-1]):
        raise ValueError("query_start_loc must be monotonic")
    num_requests = starts.size - 1
    context_length = ngram_size - 1
    if contexts.shape != (num_requests, context_length):
        raise ValueError(
            f"ngram_context shape must be {(num_requests, context_length)}, "
            f"got {contexts.shape}"
        )
    expected_heads = (ngram_size - 1) * heads_per_ngram
    if (
        parameters.layer_multipliers.shape != (ngram_size,)
        or parameters.head_vocab_sizes.shape != (expected_heads,)
        or parameters.head_offsets.shape != (expected_heads,)
    ):
        raise ValueError("PLE hash parameter shapes do not match the configuration")

    output = np.empty((inputs.size, expected_heads), dtype=np.int64)
    for request_index in range(num_requests):
        begin, end = int(starts[request_index]), int(starts[request_index + 1])
        sequence = np.concatenate((contexts[request_index], inputs[begin:end]))
        positions = np.arange(sequence.size, dtype=np.int64)
        eos_positions = np.where(sequence == eos_token_id, positions, -1)
        previous_eos_inclusive = np.maximum.accumulate(eos_positions)
        previous_eos = np.concatenate(
            (np.asarray([-1], dtype=np.int64), previous_eos_inclusive[:-1])
        )
        position_in_segment = positions - previous_eos - 1
        shifted = [sequence]
        for shift in range(1, ngram_size):
            source = positions - shift
            candidate = sequence[np.maximum(source, 0)]
            valid = (source >= 0) & (position_in_segment >= shift)
            shifted.append(np.where(valid, candidate, eos_token_id))
        token_slice = slice(context_length, sequence.size)
        blocks: list[NDArray[np.int64]] = []
        for ngram in range(2, ngram_size + 1):
            head_begin = (ngram - 2) * heads_per_ngram
            head_end = head_begin + heads_per_ngram
            with np.errstate(over="ignore"):
                mixed = shifted[0] * parameters.layer_multipliers[0]
                for index in range(1, ngram):
                    mixed = np.bitwise_xor(
                        mixed,
                        shifted[index] * parameters.layer_multipliers[index],
                    )
            sizes = parameters.head_vocab_sizes[head_begin:head_end]
            offsets = parameters.head_offsets[head_begin:head_end]
            ids = np.remainder(mixed[token_slice, None], sizes) + offsets
            blocks.append(ids.astype(np.int64, copy=False))
        output[begin:end] = np.concatenate(blocks, axis=-1)
    return output
