#!/usr/bin/env python3
"""Clamp K-pool tail slot mapping to the one-block circular contract.

``KpoolTailSpec`` is a one-block circular scratch cache: both
``max_admission_blocks_per_request`` and ``max_num_blocks_per_req`` return 1,
so its block-table row is a single entry. Slot mapping still uses the generic
paged Triton kernel, which does::

    block_indices = pos // block_size
    block_numbers = block_table[req, block_indices]

The mask only guards token validity. Nothing bounds ``block_indices`` against
the row width. For the tail group every token at ``pos >= block_size`` reads
past that one entry; the kpool seed/update kernels then write through the
garbage block id. Long generations (~2k tokens) hit this reliably. A finished
request is not proof the writes were in-bounds — most overruns land inside the
shared pool and silently corrupt another layer's indexer.

The fix clamps the index to the row::

    block_indices = tl.minimum(block_indices, block_table_stride - 1)

For the tail group ``block_table_stride == 1``, so the slot becomes
``block_table[req, 0] * block_size + pos % block_size`` — the addressing
``_kpool_tail_seed_kernel`` already documents. For every other group a request
never legitimately needs more blocks than its row holds, so the clamp is
identity.

Mechanism from vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe
(docs/KPOOL_TAIL_BUG.md). Fail-closed, idempotent, preflights the pinned
anchor before writing.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_BLOCK_TABLE_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/block_table.py",
    )
)
MARK = "    # [glm53-kpool-tail-slotmap] Never index past the request's\n"

ANCHOR = """        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
"""

PATCHED = """        block_indices = (
            virtual_block_indices * BLOCKS_PER_KV_BLOCK
            + local_block_offsets // block_size
        )
        # [glm53-kpool-tail-slotmap] Never index past the request's
        # block-table row. KpoolTailSpec is a one-block circular scratch
        # whose row is a single entry; without this clamp every token at
        # pos >= block_size reads adjacent memory and the kpool kernels
        # write through garbage. Clamping pins that group to entry 0:
        # block_table[req, 0] * block_size + pos % block_size.
        # For every other group this is identity.
        block_indices = tl.minimum(block_indices, block_table_stride - 1)
        block_numbers = tl.load(
            block_table_ptr + row_offset + block_indices,
            mask=mask & is_local,
            other=0,
        ).to(tl.int64)
"""


def count_overruns(
    positions: list[int], *, block_size: int, stride: int
) -> int:
    """How many positions the unpatched kernel would index past ``stride``."""
    if block_size < 1 or stride < 1:
        raise ValueError("block_size and stride must be >= 1")
    return sum(1 for pos in positions if (pos // block_size) >= stride)


def circular_slot_ids(
    positions: list[int],
    block_table_row: list[int],
    block_size: int,
    *,
    clamp: bool,
) -> list[int]:
    """CPU replica of the Triton mapping, with or without the row clamp."""
    if not block_table_row:
        raise ValueError("block_table_row must be non-empty")
    stride = len(block_table_row)
    out: list[int] = []
    for pos in positions:
        idx = pos // block_size
        if clamp:
            idx = min(idx, stride - 1)
        elif idx < 0 or idx >= stride:
            raise IndexError(
                f"pos={pos} indexes block {idx} past stride {stride}"
            )
        out.append(block_table_row[idx] * block_size + (pos % block_size))
    return out


def verified_state(text: str) -> bool:
    return (
        text.count(ANCHOR) == 0
        and text.count(PATCHED) == 1
        and text.count(MARK) == 1
        and "tl.minimum(block_indices, block_table_stride - 1)" in text
    )


def prepare(source: str) -> tuple[str, str]:
    marker_count = source.count(MARK)
    if marker_count:
        if marker_count != 1 or not verified_state(source):
            raise ValueError(
                "partial/inconsistent kpool tail slot-map patch "
                f"(marker={marker_count})"
            )
        return source, "already present"
    if verified_state(source):
        return source, "already patched"
    n_anchor = source.count(ANCHOR)
    if n_anchor != 1:
        raise ValueError(
            "pinned block_table slot-mapping anchor drifted "
            f"(anchor={n_anchor})"
        )
    if "tl.minimum(block_indices, block_table_stride - 1)" in source:
        raise ValueError(
            "block_table already clamps block_indices but the pinned "
            "anchor was not found — re-derive the patch"
        )
    patched = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(patched):
        raise ValueError("kpool tail slot-map post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-kpool-tail.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob("block_table*.pyc"):
        pyc.unlink(missing_ok=True)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"kpool tail slot-map preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
        clear_pyc(TARGET)
    print(f"{TARGET.name}: kpool tail slot-map {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
