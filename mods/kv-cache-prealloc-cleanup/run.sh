#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
TARGET="$PYTHON_ROOT/vllm/v1/worker/gpu_worker.py"
CACHE_CONFIG="$PYTHON_ROOT/vllm/config/cache.py"

if [ ! -f "$TARGET" ]; then
  echo "[kv-cache-prealloc-cleanup] vLLM gpu_worker.py not found at $TARGET" >&2
  exit 1
fi

if [ ! -f "$CACHE_CONFIG" ]; then
  echo "[kv-cache-prealloc-cleanup] vLLM cache.py not found at $CACHE_CONFIG" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[kv-cache-prealloc-cleanup] python3 is required to apply this mod." >&2
  exit 1
fi

python3 - "$TARGET" "$CACHE_CONFIG" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
lines = text.splitlines(keepends=True)
changed = False

if not re.search(r"(?m)^import gc$", text):
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
    raise SystemExit(
        f"[kv-cache-prealloc-cleanup] Could not find expected pattern: {pattern}"
    )


def insert_after_docstring(func_index: int, func_indent: str, block: list[str]) -> None:
    insert_at = func_index + 1
    if insert_at < len(lines):
        stripped = lines[insert_at].lstrip()
        quote = None
        if stripped.startswith('"""'):
            quote = '"""'
        elif stripped.startswith("'''"):
            quote = "'''"

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


skip_graph_marker = "spark-vllm-docker: skip CUDA graph memory profiling when disabled"
if skip_graph_marker not in "".join(lines):
    profile_call = (
        r"^(?P<indent>[ \t]+)cudagraph_memory_estimate = "
        r"self\.model_runner\.profile_cudagraph_memory\(\)\n$"
    )
    index, match = find_line(profile_call)
    indent = match.group("indent")
    lines[index : index + 1] = [
        f"{indent}# {skip_graph_marker}\n",
        f"{indent}if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:\n",
        f"{indent}    cudagraph_memory_estimate = "
        "self.model_runner.profile_cudagraph_memory()\n",
        f"{indent}else:\n",
        f"{indent}    logger.info_once(\n",
        f'{indent}        "Skipping CUDA graph memory profiling because "\n',
        f'{indent}        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0."\n',
        f"{indent}    )\n",
    ]
    changed = True


profile_cleanup_marker = "spark-vllm-docker: post-profile cleanup before KV sizing"
if profile_cleanup_marker not in "".join(lines):
    snapshot_line = (
        r"^(?P<indent>[ \t]+)free_gpu_memory = "
        r"profile_result\.after_profile\.free_memory\n$"
    )
    index, match = find_line(snapshot_line)
    indent = match.group("indent")
    lines[index:index] = [
        f"{indent}# {profile_cleanup_marker}\n",
        f'{indent}if self.device_config.device_type == "cuda":\n',
        f"{indent}    before_cleanup = profile_result.after_profile.free_memory\n",
        f'{indent}    if hasattr(self.model_runner, "_cleanup_profiling_kv_cache"):\n',
        f"{indent}        self.model_runner._cleanup_profiling_kv_cache()\n",
        f"{indent}    gc.collect()\n",
        f"{indent}    torch.cuda.synchronize(self.device)\n",
        f"{indent}    torch.cuda.empty_cache()\n",
        f"{indent}    profile_result.after_profile.measure()\n",
        f"{indent}    diff_from_create = (\n",
        f"{indent}        profile_result.after_profile - profile_result.before_create\n",
        f"{indent}    )\n",
        f"{indent}    profile_result.non_torch_increase = (\n",
        f"{indent}        diff_from_create.non_torch_memory\n",
        f"{indent}    )\n",
        f"{indent}    profile_result.non_kv_cache_memory = (\n",
        f"{indent}        profile_result.non_torch_increase\n",
        f"{indent}        + profile_result.torch_peak_increase\n",
        f"{indent}        + profile_result.weights_memory\n",
        f"{indent}    )\n",
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
    raise SystemExit(
        "[kv-cache-prealloc-cleanup] Could not find initialize_from_config "
        "in vLLM gpu_worker.py"
    )

prealloc_marker = "spark-vllm-docker: pre-KV cache allocator cleanup"
if prealloc_marker not in "".join(lines):
    body_indent = func_indent + "    "
    block = [
        f"{body_indent}# {prealloc_marker}\n",
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
        f'{body_indent}            "Cleared %.2f GiB of cached CUDA allocator memory before "\n',
        f'{body_indent}            "KV cache allocation.",\n',
        f"{body_indent}            cached_memory / (1024**3),\n",
        f"{body_indent}        )\n",
        "\n",
    ]
    insert_after_docstring(func_index, func_indent, block)
    changed = True

if changed:
    path.write_text("".join(lines))
    print("[kv-cache-prealloc-cleanup] Applied KV cache memory cleanup fixes.")
else:
    print("[kv-cache-prealloc-cleanup] Cleanup fixes are already present; skipping.")

cache_path = Path(sys.argv[2])
cache_text = cache_path.read_text()
cache_lines = cache_text.splitlines(keepends=True)
cache_changed = False
cache_marker = (
    "spark-vllm-docker: allow fixed GiB reservation with manual KV cache"
)

if cache_marker not in cache_text:
    has_conflict_validator = (
        "Cannot specify both gpu_memory_utilization_gb" in cache_text
    )
    if has_conflict_validator:
        validator_pattern = re.compile(
            r"^(?P<indent>[ \t]+)def _validate_memory_params"
            r"\(self\) -> \"CacheConfig\":\n$"
        )
        for index, line in enumerate(cache_lines):
            match = validator_pattern.match(line)
            if match:
                body_indent = match.group("indent") + "    "
                cache_lines[index + 1:index + 1] = [
                    f"{body_indent}# {cache_marker}\n",
                    f"{body_indent}return self\n",
                    "\n",
                ]
                cache_changed = True
                break
        else:
            raise SystemExit(
                "[kv-cache-prealloc-cleanup] Found the memory-parameter "
                "conflict validator in cache.py, but could not patch it."
            )

if cache_changed:
    cache_path.write_text("".join(cache_lines))
    print(
        "[kv-cache-prealloc-cleanup] Allowed --gpu-memory-utilization-gb "
        "with --kv-cache-memory-bytes."
    )
elif cache_marker in cache_text:
    print(
        "[kv-cache-prealloc-cleanup] Manual KV cache compatibility fix is "
        "already present; skipping."
    )
PY

echo "=====> vLLM will clear cached CUDA allocator memory before KV cache allocation"
