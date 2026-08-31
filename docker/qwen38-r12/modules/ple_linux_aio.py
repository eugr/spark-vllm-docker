"""Exact native Linux-AIO gathers for the NVMe-backed Qwen PLE table."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from vllm.v1.ple_offload.nvme_table import PleMmapTable


_DEFAULT_LIBRARY = Path("/usr/local/lib/qwen38/libple_linux_aio.so")


class PleLinuxAioReader:
    """One persistent native-AIO context owned by one PLE table instance."""

    def __init__(self, entries: int | None = None) -> None:
        if entries is None:
            entries = int(os.environ.get("VLLM_PLE_NVME_AIO_ENTRIES", "512"))
        if entries <= 0:
            raise ValueError("VLLM_PLE_NVME_AIO_ENTRIES must be positive")
        library = Path(
            os.environ.get("VLLM_PLE_NVME_AIO_LIBRARY", str(_DEFAULT_LIBRARY))
        )
        self._entries = entries
        self._library = ctypes.CDLL(str(library))
        self._library.ple_aio_create.argtypes = [ctypes.c_uint]
        self._library.ple_aio_create.restype = ctypes.c_void_p
        self._library.ple_aio_destroy.argtypes = [ctypes.c_void_p]
        self._library.ple_aio_error.restype = ctypes.c_char_p
        self._library.ple_aio_read_rows.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self._library.ple_aio_read_rows.restype = ctypes.c_int
        self._context = self._library.ple_aio_create(entries)
        if not self._context:
            raise OSError(self._library.ple_aio_error().decode())

    def close(self) -> None:
        if self._context:
            self._library.ple_aio_destroy(self._context)
            self._context = None

    def gather(
        self,
        table: "PleMmapTable",
        row_ids: ArrayLike,
        *,
        deduplicate: bool = True,
    ) -> NDArray[np.uint8]:
        ids = np.asarray(row_ids, dtype=np.int64)
        flat = ids.reshape(-1)
        if np.any(flat < 0) or np.any(flat >= table.total_rows):
            raise IndexError("PLE row ID outside the table")
        if table._request_trace_handle is not None:
            table._trace_requested_rows(flat)
        if deduplicate:
            unique, inverse = np.unique(flat, return_inverse=True)
        else:
            unique = flat.copy()
            inverse = np.arange(flat.size, dtype=np.int64)
        if unique.size == 0:
            return np.empty((*ids.shape, table.row_bytes), dtype=np.uint8)
        if unique.size > self._entries:
            raise ValueError(
                f"PLE Linux-AIO batch has {unique.size} rows but the context "
                f"holds only {self._entries}"
            )

        shard_ids = np.searchsorted(table._ends_array, unique, side="right")
        fds = np.empty(unique.size, dtype=np.int32)
        offsets = np.empty(unique.size, dtype=np.int64)
        for shard_index in np.unique(shard_ids):
            positions = np.flatnonzero(shard_ids == shard_index)
            shard = table.shards[int(shard_index)]
            local_rows = unique[positions] - shard.global_row_start
            handle = table._handles[shard.tensor.path]
            fds[positions] = handle.fileno()
            offsets[positions] = (
                shard.tensor.byte_start + local_rows * shard.row_bytes
            )

        unique_output = np.empty(
            (unique.size, table.row_bytes), dtype=np.uint8
        )
        result = self._library.ple_aio_read_rows(
            self._context,
            fds.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            unique_output.ctypes.data_as(ctypes.c_void_p),
            unique.size,
            table.row_bytes,
        )
        if result != 0:
            raise OSError(self._library.ple_aio_error().decode())
        table.last_pread_stats = {
            "read_calls": int(unique.size),
            "requested_bytes": int(unique.size * table.row_bytes),
            "span_bytes": int(unique.size * table.row_bytes),
            "unique_rows": int(unique.size),
            "native_aio_submissions": 1,
        }
        output = unique_output[inverse] if deduplicate else unique_output
        return output.reshape(*ids.shape, table.row_bytes)

    def __enter__(self) -> "PleLinuxAioReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
