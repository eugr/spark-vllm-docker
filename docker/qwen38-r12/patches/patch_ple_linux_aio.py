#!/usr/bin/env python3
"""Add the production native Linux-AIO backend to the NVMe PLE runtime.

This patch deliberately targets the uninstrumented output of
``patch_nvme_ple_offload.py``. The profiling image used during development
wrapped every lookup in trace collection; the release runtime does not carry
that profiling code or its per-token overhead.
"""

from __future__ import annotations

from pathlib import Path


SITE = Path("/usr/local/lib/python3.12/dist-packages")
NVME_TABLE = SITE / "vllm/v1/ple_offload/nvme_table.py"
PLE_LAYER = SITE / "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"
MARKER = "QWEN_PLE_LINUX_AIO_PRODUCTION_V1"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def patch_nvme_table() -> None:
    source = NVME_TABLE.read_text()
    if MARKER in source:
        raise RuntimeError("NVMe table is already Linux-AIO patched")
    source = replace_once(
        source,
        "from numpy.typing import ArrayLike, NDArray\n",
        "from numpy.typing import ArrayLike, NDArray\n\n"
        "from vllm.v1.ple_offload.ple_linux_aio import PleLinuxAioReader\n\n"
        f"# {MARKER}\n",
        "Linux-AIO import",
    )
    source = replace_once(
        source,
        "        self._pread_executor: "
        "concurrent.futures.ThreadPoolExecutor | None = None\n"
        "        self._pread_executor_workers = 0\n",
        "        self._pread_executor: "
        "concurrent.futures.ThreadPoolExecutor | None = None\n"
        "        self._pread_executor_workers = 0\n"
        "        self._linux_aio_reader: PleLinuxAioReader | None = None\n",
        "reader state",
    )
    source = replace_once(
        source,
        "    def close(self) -> None:\n"
        "        if self._pread_executor is not None:\n",
        "    def close(self) -> None:\n"
        "        if self._linux_aio_reader is not None:\n"
        "            self._linux_aio_reader.close()\n"
        "            self._linux_aio_reader = None\n"
        "        if self._pread_executor is not None:\n",
        "reader close",
    )
    source = replace_once(
        source,
        "    def advise_dontneed(self) -> None:\n",
        "    def gather_linux_aio(\n"
        "        self, row_ids: ArrayLike, *, deduplicate: bool = True\n"
        "    ) -> NDArray[np.uint8]:\n"
        "        \"\"\"Gather exact rows through one persistent native-AIO context.\"\"\"\n"
        "        if self._linux_aio_reader is None:\n"
        "            self._linux_aio_reader = PleLinuxAioReader()\n"
        "        return self._linux_aio_reader.gather(\n"
        "            self, row_ids, deduplicate=deduplicate\n"
        "        )\n\n"
        "    def advise_dontneed(self) -> None:\n",
        "gather method",
    )
    NVME_TABLE.write_text(source)


def patch_ple_layer() -> None:
    source = PLE_LAYER.read_text()
    if MARKER in source:
        raise RuntimeError("PLE layer is already Linux-AIO patched")
    source = replace_once(
        source,
        '        if self._nvme_backend not in {"auto", "mmap", "pread"}:\n'
        '            raise ValueError(\n'
        '                "VLLM_PLE_NVME_BACKEND must be \'auto\', \'mmap\', or \'pread\'"\n'
        '            )\n',
        '        if self._nvme_backend not in {\n'
        '            "auto", "mmap", "pread", "linux_aio"\n'
        '        }:\n'
        '            raise ValueError(\n'
        '                "VLLM_PLE_NVME_BACKEND must be \'auto\', \'mmap\', "\n'
        '                "\'pread\', or \'linux_aio\'"\n'
        '            )\n'
        f'        # {MARKER}\n',
        "backend validation",
    )
    old = '''            use_pread = self._nvme_backend == "pread" or (
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
'''
    new = '''            use_linux_aio = (
                self._nvme_backend == "linux_aio"
                and num_valid_tokens <= self._nvme_pread_max_tokens
            )
            use_pread = (
                self._nvme_backend == "pread"
                or (self._nvme_backend == "linux_aio" and not use_linux_aio)
                or (
                    self._nvme_backend == "auto"
                    and num_valid_tokens <= self._nvme_pread_max_tokens
                )
            )
            if use_linux_aio:
                raw_rows = self._nvme_table.gather_linux_aio(
                    row_ids, deduplicate=True
                )
            elif use_pread:
                raw_rows = self._nvme_table.gather_pread(
                    row_ids,
                    deduplicate=True,
                    workers=self._nvme_pread_workers,
                )
            else:
                raw_rows = self._nvme_table.gather(row_ids, deduplicate=True)
'''
    source = replace_once(source, old, new, "forward backend dispatch")
    PLE_LAYER.write_text(source)


if __name__ == "__main__":
    patch_nvme_table()
    patch_ple_layer()
