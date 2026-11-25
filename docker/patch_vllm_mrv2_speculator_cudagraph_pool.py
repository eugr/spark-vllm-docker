#!/usr/bin/env python3
"""Isolate MRV2 speculator graphs during CUDA-graph memory profiling."""

from __future__ import annotations

import sys
from pathlib import Path


TARGET_REL = Path("vllm/v1/worker/gpu/cudagraph_utils.py")
MARKER = "spark-vllm-docker: isolate speculator profiling graphs"

DECLARATION_ANCHOR = """    all_wrappers: list[Any] = []
    original_pools: dict[int, Any] = {}
    try:
"""

DECLARATION_REPLACEMENT = """    all_wrappers: list[Any] = []
    original_pools: dict[int, Any] = {}
    speculator_managers: list[tuple[str, CudaGraphManager]] = []
    try:
"""

POOL_ANCHOR = """        manager.pool = current_platform.graph_pool_handle()
        if manager.use_breakable_cg:
"""

POOL_REPLACEMENT = f"""        manager.pool = current_platform.graph_pool_handle()
        # {MARKER}. Speculator prefill and decode use separate
        # CudaGraphManager instances, so redirect them before their lazy capture
        # runners are created. Otherwise their discarded profiling graphs use
        # the persistent global pool and leave its allocator handle stale.
        if runner.speculator is not None:
            speculator_managers = [
                (name, value)
                for name, value in vars(runner.speculator).items()
                if isinstance(value, CudaGraphManager)
            ]
            for _, speculator_manager in speculator_managers:
                speculator_manager.pool = manager.pool
        if manager.use_breakable_cg:
"""

CLEANUP_ANCHOR = """        CUDAGraphWrapper.clear_all_graphs()
        BreakableCUDAGraphWrapper.clear_all_graphs()
        for wrapper in all_wrappers:
"""

CLEANUP_REPLACEMENT = """        CUDAGraphWrapper.clear_all_graphs()
        BreakableCUDAGraphWrapper.clear_all_graphs()
        # Drop speculator FULL graphs and their references to the throwaway
        # pool before the profiling state is garbage-collected. The real KV
        # initialization constructs fresh managers immediately afterwards.
        for name, speculator_manager in speculator_managers:
            speculator_manager.graphs.clear()
            speculator_manager.pool = None
            if (
                runner.speculator is not None
                and getattr(runner.speculator, name, None) is speculator_manager
            ):
                setattr(runner.speculator, name, None)
        for wrapper in all_wrappers:
"""


class PatchError(RuntimeError):
    """The vulnerable profiler is present in an unsupported source layout."""


def is_fixed(source: str) -> bool:
    local_fix = (
        MARKER in source
        and "speculator_manager.pool = manager.pool" in source
        and "speculator_manager.graphs.clear()" in source
        and "setattr(runner.speculator, name, None)" in source
    )
    manager_collection_fix = all(
        anchor in source
        for anchor in (
            "def _profiling_cudagraph_managers(",
            "speculator = runner.speculator",
            "candidate = getattr(speculator, name, None)",
            "graph_managers = _profiling_cudagraph_managers(runner)",
            "graph_manager.pool = manager.pool",
            "graph_manager.graphs.clear()",
            "graph_manager.pool = original_manager_pools[id(graph_manager)]",
        )
    )
    return local_fix or manager_collection_fix


def is_affected_profiler(source: str) -> bool:
    return (
        "def profile_cudagraph_memory" in source
        and "manager.pool = current_platform.graph_pool_handle()" in source
        and "PIECEWISE, encoder and speculator graphs are measured in full" in source
    )


def replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise PatchError(
            f"expected exactly one {label} anchor, found {count}; "
            "refusing to patch an unknown MRV2 profiler layout"
        )
    return source.replace(anchor, replacement, 1)


def patch_source(source: str) -> tuple[str, str]:
    if is_fixed(source):
        return source, "Equivalent MRV2 speculator CUDA-graph pool fix is present; skipping"
    if not is_affected_profiler(source):
        return source, "Affected MRV2 CUDA-graph memory profiler is absent; skipping"

    patched = replace_once(
        source,
        DECLARATION_ANCHOR,
        DECLARATION_REPLACEMENT,
        "profiling-state declaration",
    )
    patched = replace_once(
        patched,
        POOL_ANCHOR,
        POOL_REPLACEMENT,
        "throwaway-pool assignment",
    )
    patched = replace_once(
        patched,
        CLEANUP_ANCHOR,
        CLEANUP_REPLACEMENT,
        "profiling cleanup",
    )
    if not is_fixed(patched):
        raise PatchError("MRV2 speculator CUDA-graph pool patch postcondition failed")
    return patched, "Applied MRV2 speculator CUDA-graph profiling-pool fix"


def main() -> None:
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    target = source_root / TARGET_REL
    if not target.exists():
        print(f"{TARGET_REL} is absent; targeted MRV2 profiling patch is not applicable")
        return

    source = target.read_text()
    try:
        patched, message = patch_source(source)
    except PatchError as exc:
        raise SystemExit(f"MRV2 speculator CUDA-graph pool patch failed: {exc}") from exc

    if patched != source:
        target.write_text(patched)
    print(message)


if __name__ == "__main__":
    main()
