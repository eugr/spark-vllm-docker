#!/usr/bin/env python3
"""Align InstantTensor's memory budget with vLLM's CUDA-on-WSL policy."""

import argparse
import ast
import importlib.util
from pathlib import Path


ORIGINAL = """        free_bytes, total_bytes = torch.cuda.mem_get_info()
        avail_bytes = int(free_bytes * max_free_mem_usage)
"""
PATCHED = """        # Share vLLM's UMA accounting and its CUDA-on-WSL policy.
        from vllm.utils.mem_utils import MemorySnapshot

        free_bytes = MemorySnapshot(device=self.device).free_memory
        avail_bytes = int(free_bytes * max_free_mem_usage)
"""
# InstantTensor 0.2.1 added Linux UMA accounting, but its integrated-GPU check
# also selects host RAM under WSL. Keep vLLM's policy inside upstream's try block
# so fraction validation and error synchronization across ranks still apply.
ORIGINAL_UMA = """            # Managed-memory support alone does not imply shared physical memory.
            if sys.platform == "linux" and torch.cuda.get_device_properties(self.device).is_integrated:
                avail_bytes = _host_available_bytes()
                debug_log("MemAvailable: %d bytes", avail_bytes)
            else:
                avail_bytes = torch.cuda.mem_get_info(self.device)[0]
                debug_log("CUDA free memory: %d bytes", avail_bytes)
            avail_bytes = int(avail_bytes * max_free_mem_usage)
"""
PATCHED_UMA = """            # Share vLLM's UMA accounting and its CUDA-on-WSL policy.
            from vllm.utils.mem_utils import MemorySnapshot

            avail_bytes = MemorySnapshot(device=self.device).free_memory
            debug_log("vLLM free memory: %d bytes", avail_bytes)
            avail_bytes = int(avail_bytes * max_free_mem_usage)
"""
QUERY_VARIANTS = ((ORIGINAL, PATCHED), (ORIGINAL_UMA, PATCHED_UMA))
# 0.2.3 moved budgeting into _internal/resources.py. Its native UMA policy
# accounts for cgroups, ranks sharing a host, and host/device buffer sharing.
# Preserve that accounting; only WSL must use CUDA's budget instead of guest RAM.
ORIGINAL_RESOURCES = """        integrated = torch.cuda.get_device_properties(device).is_integrated
        free_mem = torch.cuda.mem_get_info(device)[0]
"""
PATCHED_RESOURCES = """        from vllm.platforms import current_platform
        from vllm.platforms.interface import in_wsl

        # WSL guest RAM does not describe CUDA's memory allocation budget.
        integrated = torch.cuda.get_device_properties(device).is_integrated and not (
            current_platform.is_cuda() and in_wsl()
        )
        free_mem = torch.cuda.mem_get_info(device)[0]
"""


def patch_memory_query(source: str) -> str:
    tree = ast.parse(source)
    queries = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "query_rank_resources"
    ]
    methods = [
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "safe_open"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "_determine_io_params"
    ]
    if len(queries) + len(methods) != 1:
        raise ValueError("Expected exactly one known InstantTensor memory-budget function")

    method = (queries or methods)[0]
    variants = ((ORIGINAL_RESOURCES, PATCHED_RESOURCES),) if queries else QUERY_VARIANTS
    lines = source.splitlines(keepends=True)
    start, end = method.lineno - 1, method.end_lineno
    body = "".join(lines[start:end])
    original_count = sum(body.count(original) for original, _ in variants)
    patched_count = sum(body.count(patched) for _, patched in variants)
    if original_count == 0 and patched_count == 1:
        return source
    if original_count != 1 or patched_count != 0:
        raise ValueError("Expected exactly one known InstantTensor memory-budget query")

    for original, replacement in variants:
        if original in body:
            lines[start:end] = [body.replace(original, replacement, 1)]
            break
    patched = "".join(lines)
    compile(patched, "instanttensor_memory.py", "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", nargs="?", type=Path,
        help="Path to instanttensor/_impl.py or instanttensor/_internal/resources.py",
    )
    parser.add_argument(
        "--installed", action="store_true", help="Patch installed InstantTensor"
    )
    args = parser.parse_args()
    if args.installed:
        if args.target is not None:
            parser.error("Use either target or --installed")
        # Locate the top-level package without importing InstantTensor or CUDA.
        spec = importlib.util.find_spec("instanttensor")
        locations = list(spec.submodule_search_locations or []) if spec else []
        if len(locations) != 1:
            raise SystemExit("Unable to locate the installed InstantTensor package")
        package_root = Path(locations[0])
        candidates = [package_root / name for name in ("_impl.py", "_internal/resources.py")]
        targets = [path for path in candidates if path.is_file()]
        if len(targets) != 1:
            raise SystemExit(
                f"Unable to locate one InstantTensor memory-budget module in {package_root}; "
                "expected either _impl.py or _internal/resources.py"
            )
        target = targets[0]
    elif args.target is not None:
        target = args.target
    else:
        parser.error("Provide target or --installed")

    try:
        source = target.read_text()
        patched = patch_memory_query(source)
        if patched == source:
            print("InstantTensor memory accounting is already patched; skipping")
        else:
            target.write_text(patched)
            print(f"Patched InstantTensor memory policy in {target}")
    except (OSError, ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to patch {target}: {exc}") from exc


if __name__ == "__main__":
    main()
