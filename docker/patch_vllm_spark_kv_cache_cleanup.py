#!/usr/bin/env python3
"""Clean profiling allocations before vLLM sizes and creates the KV cache."""

import re
import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "vllm/v1/worker/gpu_worker.py"

if not target.exists():
    raise SystemExit(f"{target} not found; cannot apply KV cache cleanup patch")

text = target.read_text()
lines = text.splitlines(keepends=True)
changed = False

# The B12X fork reclaims profiling allocations immediately before its final
# snapshot and accounts for allocations retained after activation profiling.
# A final snapshot alone does not prove that cleanup happened.
native_profile_cleanup = re.search(
    r"(?m)^[ \t]+gc\.collect\(\)\n"
    r"[ \t]+torch\.accelerator\.synchronize\(\)\n"
    r"[ \t]+torch\.accelerator\.empty_cache\(\)\n"
    r"(?:[ \t]*(?:#.*)?\n)*"
    r"[ \t]+final_profile_snapshot = MemorySnapshot\(device=self\.device\)\n"
    r"(?:[ \t]*(?:#.*)?\n)*"
    r"[ \t]+late_persistent_memory = ",
    text,
)
upstream_profile_cleanup = native_profile_cleanup is not None
profile_cleanup_present = upstream_profile_cleanup or (
    "profile_result.after_profile.measure()" in text
    and "diff_from_create.non_torch_memory" in text
)
initialize_index = next(
    (
        index
        for index, line in enumerate(lines)
        if re.match(r"^[ \t]+def initialize_from_config\(self,\s*kv_cache_config\b", line)
    ),
    None,
)
native_prealloc_cleanup = initialize_index is not None and re.search(
    r"(?m)^[ \t]+gc\.collect\(\)\n"
    r"[ \t]+torch\.accelerator\.synchronize\(\)\n"
    r"[ \t]+torch\.accelerator\.empty_cache\(\)",
    "".join(lines[initialize_index : initialize_index + 25]),
)
prealloc_cleanup_present = (
    bool(native_prealloc_cleanup)
    or (
        "memory_reserved(self.device)" in text
        and "memory_allocated(self.device)" in text
        and "empty_cache()" in text
    )
)
needs_cleanup = not (profile_cleanup_present and prealloc_cleanup_present)

if needs_cleanup and not re.search(r"(?m)^import gc$", text):
    insert_at = None
    last_future_import = None
    for i, line in enumerate(lines):
        if line.startswith("from __future__ import "):
            last_future_import = i
        elif insert_at is None and (
            line.startswith("import ") or line.startswith("from ")
        ):
            insert_at = i
    if last_future_import is not None:
        lines.insert(last_future_import + 1, "import gc\n")
    elif insert_at is not None:
        lines.insert(insert_at, "import gc\n")
    else:
        lines.insert(0, "import gc\n")
    changed = True


def find_line(pattern: str) -> tuple[int, re.Match[str]]:
    regex = re.compile(pattern)
    for index, line in enumerate(lines):
        match = regex.match(line)
        if match:
            return index, match
    raise SystemExit(f"Could not find expected vLLM pattern: {pattern}")


def find_first_line(patterns: tuple[str, ...]) -> tuple[int, re.Match[str]]:
    for pattern in patterns:
        regex = re.compile(pattern)
        for index, line in enumerate(lines):
            match = regex.match(line)
            if match:
                return index, match
    expected = "\n".join(f"  - {pattern}" for pattern in patterns)
    raise SystemExit(f"Could not find any expected vLLM pattern:\n{expected}")


def insert_after_docstring(func_index: int, func_indent: str, block: list[str]) -> None:
    insert_at = func_index + 1
    if insert_at < len(lines):
        stripped = lines[insert_at].lstrip()
        quote = None
        if stripped.startswith(chr(34) * 3):
            quote = chr(34) * 3
        elif stripped.startswith(chr(39) * 3):
            quote = chr(39) * 3

        if quote is not None:
            if stripped.count(quote) >= 2 and not stripped.startswith(quote * 2):
                insert_at += 1
            else:
                insert_at += 1
                while insert_at < len(lines):
                    if quote in lines[insert_at]:
                        insert_at += 1
                        break
                    insert_at += 1

    lines[insert_at:insert_at] = block


if not profile_cleanup_present:
    # The B12X fork takes a final snapshot after CUDA-graph profiling so it can
    # account for late persistent allocations. Clean up before that snapshot;
    # otherwise the memory it is meant to release remains charged to the KV
    # cache budget. Upstream vLLM currently reads after_profile directly.
    profile_cleanup_anchors = (
        r"^(?P<indent>[ \t]+)final_profile_snapshot = "
        r"MemorySnapshot\(device=self\.device\)\n$",
        r"^(?P<indent>[ \t]+)free_gpu_memory = "
        r"profile_result\.after_profile\.free_memory\n$",
    )
    index, match = find_first_line(profile_cleanup_anchors)
    indent = match.group("indent")
    lines[index:index] = [
        f"{indent}# spark-vllm-docker: post-profile cleanup before KV sizing\n",
        f'{indent}if self.device_config.device_type == "cuda":\n',
        f"{indent}    before_cleanup = profile_result.after_profile.free_memory\n",
        f'{indent}    if hasattr(self.model_runner, "_cleanup_profiling_kv_cache"):\n',
        f"{indent}        self.model_runner._cleanup_profiling_kv_cache()\n",
        f"{indent}    gc.collect()\n",
        f"{indent}    torch.cuda.synchronize(self.device)\n",
        f"{indent}    torch.cuda.empty_cache()\n",
        f"{indent}    profile_result.after_profile.measure()\n",
        f"{indent}    diff_from_create = (\n",
        f"{indent}        profile_result.after_profile "
        "- profile_result.before_create\n",
        f"{indent}    )\n",
        f"{indent}    profile_result.non_torch_increase = (\n",
        f"{indent}        diff_from_create.non_torch_memory\n",
        f"{indent}    )\n",
        f'{indent}    if hasattr(profile_result, "transient_peak_headroom"):\n',
        f"{indent}        # Newer vLLM measures all persistent allocations from\n",
        f"{indent}        # device free memory, including Torch allocations.\n",
        f"{indent}        profile_result.total_consumed = (\n",
        f"{indent}            profile_result.before_create.free_memory\n",
        f"{indent}            - profile_result.after_profile.free_memory\n",
        f"{indent}        )\n",
        f"{indent}        profile_result.non_kv_cache_memory = (\n",
        f"{indent}            profile_result.total_consumed\n",
        f"{indent}            + profile_result.transient_peak_headroom\n",
        f"{indent}        )\n",
        f"{indent}    else:\n",
        f"{indent}        # Compatibility with older profiling results.\n",
        f"{indent}        profile_result.non_kv_cache_memory = (\n",
        f"{indent}            profile_result.non_torch_increase\n",
        f"{indent}            + profile_result.torch_peak_increase\n",
        f"{indent}            + profile_result.weights_memory\n",
        f"{indent}        )\n",
        f"{indent}    cleanup_freed = (\n",
        f"{indent}        profile_result.after_profile.free_memory - before_cleanup\n",
        f"{indent}    )\n",
        f"{indent}    if cleanup_freed > 0:\n",
        f"{indent}        logger.info_once(\n",
        f'{indent}            "Freed %.2f GiB before KV cache sizing; "\n',
        f'{indent}            "non-torch profile increase is %.2f GiB.",\n',
        f"{indent}            cleanup_freed / (1024**3),\n",
        f"{indent}            profile_result.non_torch_increase / (1024**3),\n",
        f"{indent}        )\n",
        "\n",
    ]
    changed = True

if not prealloc_cleanup_present:
    func_index = None
    func_indent = None
    for i, line in enumerate(lines):
        match = re.match(
            r"^(?P<indent>[ \t]+)def initialize_from_config"
            r"\(self,\s*kv_cache_config\b",
            line,
        )
        if match:
            func_index = i
            func_indent = match.group("indent")
            break

    if func_index is None or func_indent is None:
        raise SystemExit("Could not find initialize_from_config in vLLM gpu_worker.py")

    body_indent = func_indent + "    "
    block = [
        f"{body_indent}# spark-vllm-docker: pre-KV cache allocator cleanup\n",
        f'{body_indent}if self.device_config.device_type == "cuda":\n',
        f"{body_indent}    gc.collect()\n",
        f"{body_indent}    torch.cuda.synchronize(self.device)\n",
        f"{body_indent}    cached_memory = max(\n",
        f"{body_indent}        torch.cuda.memory_reserved(self.device)\n",
        f"{body_indent}        - torch.cuda.memory_allocated(self.device),\n",
        f"{body_indent}        0,\n",
        f"{body_indent}    )\n",
        f"{body_indent}    torch.cuda.empty_cache()\n",
        f"{body_indent}    if cached_memory > 0:\n",
        f"{body_indent}        logger.info_once(\n",
        f'{body_indent}            "Cleared %.2f GiB of cached CUDA allocator '
        'memory before "\n',
        f'{body_indent}            "KV cache allocation.",\n',
        f"{body_indent}            cached_memory / (1024**3),\n",
        f"{body_indent}        )\n",
        "\n",
    ]
    insert_after_docstring(func_index, func_indent, block)
    changed = True

if upstream_profile_cleanup:
    print("Upstream post-profile cleanup and late-persistent accounting present")

if changed:
    target.write_text("".join(lines))
    print("Applied Spark KV cache cleanup patch")
else:
    print("Equivalent Spark KV cache cleanup already present; skipping")
