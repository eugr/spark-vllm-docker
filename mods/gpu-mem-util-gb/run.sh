#!/usr/bin/env bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[gpu-mem-util-gb] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[gpu-mem-util-gb] python3 is required to apply this mod." >&2
  exit 1
fi

python3 - "$PYTHON_ROOT" <<'PY'
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
changed_paths: set[Path] = set()


def die(message: str) -> None:
    raise SystemExit(f"[gpu-mem-util-gb] {message}")


def read(rel: str) -> tuple[Path, str]:
    path = root / rel
    if not path.exists():
        die(f"vLLM source file not found: {path}")
    return path, path.read_text()


def write(path: Path, old: str, new: str) -> None:
    if old != new:
        path.write_text(new)
        changed_paths.add(path)


def replace_once(
    text: str,
    old: str,
    new: str,
    description: str,
    *,
    already: str | None = None,
) -> tuple[str, bool]:
    if already is not None and already in text:
        return text, False
    if old not in text:
        die(f"Could not find expected source anchor for {description}.")
    return text.replace(old, new, 1), True


def replace_regex_once(
    text: str,
    pattern: str,
    repl: str,
    description: str,
    *,
    flags: int = 0,
    already: str | None = None,
) -> tuple[str, bool]:
    if already is not None and already in text:
        return text, False
    new_text, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        die(f"Could not find expected source anchor for {description}.")
    return new_text, True


def replace_function(
    text: str,
    name: str,
    replacement: str,
    description: str,
    *,
    already: str | None = None,
) -> tuple[str, bool]:
    if already is not None and already in text:
        return text, False

    match = re.search(rf"(?m)^def {re.escape(name)}\(", text)
    if match is None:
        die(f"Could not find function {name} for {description}.")

    start = match.start()
    next_match = re.search(r"(?m)^def \w+\(", text[match.end() :])
    if next_match is None:
        end = len(text)
    else:
        end = match.end() + next_match.start()

    return text[:start] + replacement + text[end:], True


def replace_between(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    description: str,
    *,
    already: str | None = None,
) -> tuple[str, bool]:
    if already is not None and already in text:
        return text, False
    start = text.find(start_marker)
    if start == -1:
        die(f"Could not find start anchor for {description}.")
    end = text.find(end_marker, start)
    if end == -1:
        die(f"Could not find end anchor for {description}.")
    return text[:start] + replacement + text[end:], True


def patch_cache_config() -> None:
    path, text = read("vllm/config/cache.py")
    original = text

    if "gpu_memory_utilization_gb: float | None" not in text:
        field_block = (
            r'(    gpu_memory_utilization: float = Field\(default=0\.92, gt=0, le=1\)\n'
            r'    """The fraction of GPU memory.*?'
            r'set the GPU memory utilization to 0\.5 for each instance\."""\n)'
        )
        addition = (
            r"\1"
            "    gpu_memory_utilization_gb: float | None = Field(default=None, gt=0)\n"
            "    \"\"\"Amount of GPU memory to be used in GiB. This provides fine-grained\n"
            "    control over GPU memory usage and is particularly useful on unified memory\n"
            "    systems where available memory changes dynamically. If specified, it\n"
            "    overrides gpu_memory_utilization. Cannot be used simultaneously with\n"
            "    kv_cache_memory_bytes.\"\"\"\n"
        )
        text, _ = replace_regex_once(
            text,
            field_block,
            addition,
            "CacheConfig.gpu_memory_utilization_gb field",
            flags=re.DOTALL,
        )

    text, _ = replace_once(
        text,
        '            "gpu_memory_utilization",\n',
        '            "gpu_memory_utilization",\n'
        '            "gpu_memory_utilization_gb",\n',
        "CacheConfig hash ignored factors",
        already='"gpu_memory_utilization_gb",',
    )

    if "def _validate_memory_params" not in text:
        validator = (
            "    @model_validator(mode=\"after\")\n"
            "    def _validate_memory_params(self) -> \"CacheConfig\":\n"
            "        if (\n"
            "            self.gpu_memory_utilization_gb is not None\n"
            "            and self.kv_cache_memory_bytes is not None\n"
            "        ):\n"
            "            raise ValueError(\n"
            "                \"Cannot specify both gpu_memory_utilization_gb and \"\n"
            "                \"kv_cache_memory_bytes. Please use only one of them.\"\n"
            "            )\n"
            "        return self\n"
            "\n"
        )
        text, _ = replace_once(
            text,
            '    @field_validator("calculate_kv_scales", mode="after")\n',
            validator + '    @field_validator("calculate_kv_scales", mode="after")\n',
            "CacheConfig memory parameter validator",
        )

    write(path, original, text)


def patch_engine_args() -> None:
    path, text = read("vllm/engine/arg_utils.py")
    original = text

    text, _ = replace_once(
        text,
        "    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization\n",
        "    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization\n"
        "    gpu_memory_utilization_gb: float | None = CacheConfig.gpu_memory_utilization_gb\n",
        "EngineArgs gpu_memory_utilization_gb field",
        already="    gpu_memory_utilization_gb: float | None = CacheConfig.gpu_memory_utilization_gb\n",
    )

    text, _ = replace_once(
        text,
        '        cache_group.add_argument(\n'
        '            "--gpu-memory-utilization", **cache_kwargs["gpu_memory_utilization"]\n'
        "        )\n",
        '        cache_group.add_argument(\n'
        '            "--gpu-memory-utilization", **cache_kwargs["gpu_memory_utilization"]\n'
        "        )\n"
        "        cache_group.add_argument(\n"
        '            "--gpu-memory-utilization-gb", **cache_kwargs["gpu_memory_utilization_gb"]\n'
        "        )\n",
        "EngineArgs --gpu-memory-utilization-gb CLI argument",
        already='"--gpu-memory-utilization-gb"',
    )

    text, _ = replace_once(
        text,
        "            gpu_memory_utilization=self.gpu_memory_utilization,\n",
        "            gpu_memory_utilization=self.gpu_memory_utilization,\n"
        "            gpu_memory_utilization_gb=self.gpu_memory_utilization_gb,\n",
        "EngineArgs CacheConfig gpu_memory_utilization_gb wiring",
        already="gpu_memory_utilization_gb=self.gpu_memory_utilization_gb",
    )

    write(path, original, text)


def patch_llm_entrypoint() -> None:
    path, text = read("vllm/entrypoints/llm.py")
    original = text

    if "gpu_memory_utilization_gb: Amount of GPU memory to reserve in GiB." not in text:
        text, _ = replace_once(
            text,
            "        kv_cache_memory_bytes: Size of KV Cache per GPU in bytes. By default,\n",
            "        gpu_memory_utilization_gb: Amount of GPU memory to reserve in GiB.\n"
            "            This provides fine-grained control over GPU memory usage and is\n"
            "            particularly useful on unified memory systems where available memory\n"
            "            changes dynamically. If specified, it overrides gpu_memory_utilization.\n"
            "            Cannot be used simultaneously with kv_cache_memory_bytes.\n"
            "        kv_cache_memory_bytes: Size of KV Cache per GPU in bytes. By default,\n",
            "LLM gpu_memory_utilization_gb docstring",
        )

    text, _ = replace_once(
        text,
        "        gpu_memory_utilization: float = 0.92,\n",
        "        gpu_memory_utilization: float = 0.92,\n"
        "        gpu_memory_utilization_gb: float | None = None,\n",
        "LLM gpu_memory_utilization_gb parameter",
        already="        gpu_memory_utilization_gb: float | None = None,\n",
    )

    text, _ = replace_once(
        text,
        "            gpu_memory_utilization=gpu_memory_utilization,\n",
        "            gpu_memory_utilization=gpu_memory_utilization,\n"
        "            gpu_memory_utilization_gb=gpu_memory_utilization_gb,\n",
        "LLM EngineArgs gpu_memory_utilization_gb wiring",
        already="gpu_memory_utilization_gb=gpu_memory_utilization_gb",
    )

    write(path, original, text)


def patch_request_memory() -> None:
    path, text = read("vllm/v1/worker/utils.py")
    original = text

    replacement = '''def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    if cache_config.gpu_memory_utilization_gb is not None:
        requested_memory = math.ceil(cache_config.gpu_memory_utilization_gb * 1024**3)
        if requested_memory > init_snapshot.total_memory:
            raise ValueError(
                f"Requested memory ({format_gib(requested_memory)} GiB) exceeds "
                f"total GPU memory ({format_gib(init_snapshot.total_memory)} GiB). "
                f"Reduce gpu_memory_utilization_gb or use a smaller value."
            )
        if requested_memory > init_snapshot.free_memory:
            raise ValueError(
                f"Requested memory ({format_gib(requested_memory)} GiB) exceeds "
                f"available memory ({format_gib(init_snapshot.free_memory)} GiB). "
                f"Reduce gpu_memory_utilization_gb or free up GPU memory."
            )
    else:
        requested_memory = math.ceil(
            init_snapshot.total_memory * cache_config.gpu_memory_utilization
        )

        if init_snapshot.free_memory < requested_memory:
            raise ValueError(
                f"Free memory on device {init_snapshot.device_} "
                f"({format_gib(init_snapshot.free_memory)}/"
                f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({cache_config.gpu_memory_utilization}, "
                f"{format_gib(requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes."
            )

    return requested_memory


'''
    text, _ = replace_function(
        text,
        "request_memory",
        replacement,
        "request_memory GiB support",
        already="cache_config.gpu_memory_utilization_gb is not None",
    )

    write(path, original, text)


def patch_gpu_worker() -> None:
    path, text = read("vllm/v1/worker/gpu_worker.py")
    original = text

    text, _ = replace_once(
        text,
        "            by adjusting the `gpu_memory_utilization` parameter.\n",
        "            by adjusting the `gpu_memory_utilization` or\n"
        "            `gpu_memory_utilization_gb` parameter.\n",
        "GPU worker memory tip",
        already="`gpu_memory_utilization_gb` parameter",
    )

    text, _ = replace_once(
        text,
        '                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "\n',
        '                "gpu_memory_utilization or gpu_memory_utilization_gb config. "\n'
        '                "Only use kv_cache_memory_bytes "\n',
        "GPU worker kv_cache_memory_bytes info",
        already="gpu_memory_utilization or gpu_memory_utilization_gb config",
    )

    text, _ = replace_once(
        text,
        '        logger.debug(\n'
        '            "Initial free memory: %s GiB; Requested memory: %f (util), %s GiB",\n'
        "            format_gib(self.init_snapshot.free_memory),\n"
        "            self.cache_config.gpu_memory_utilization,\n"
        "            format_gib(self.requested_memory),\n"
        "        )\n",
        "        requested_memory_config = (\n"
        "            f\"{self.cache_config.gpu_memory_utilization_gb} GiB\"\n"
        "            if self.cache_config.gpu_memory_utilization_gb is not None\n"
        "            else f\"{self.cache_config.gpu_memory_utilization} (util)\"\n"
        "        )\n"
        "        logger.debug(\n"
        "            \"Initial free memory: %s GiB; Requested memory: %s, %s GiB\",\n"
        "            format_gib(self.init_snapshot.free_memory),\n"
        "            requested_memory_config,\n"
        "            format_gib(self.requested_memory),\n"
        "        )\n",
        "GPU worker requested memory debug log",
        already="requested_memory_config = (\n            f\"{self.cache_config.gpu_memory_utilization_gb} GiB\"",
    )

    cudagraph_block = '''        if cudagraph_memory_estimate > 0:
            if self.cache_config.gpu_memory_utilization_gb is not None:
                current_gb = self.cache_config.gpu_memory_utilization_gb
                cg_gb_delta = cudagraph_memory_estimate / GiB_bytes
                if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                    equiv_gb = round(max(current_gb - cg_gb_delta, 0), 4)
                    suggested_gb = round(current_gb + cg_gb_delta, 4)
                    logger.info(
                        "CUDA graph memory profiling is enabled (default since "
                        "v0.21.0). The current --gpu-memory-utilization-gb=%.4f "
                        "is equivalent to --gpu-memory-utilization-gb=%.4f "
                        "without CUDA graph memory profiling. To maintain the "
                        "same effective KV cache size as before, increase "
                        "--gpu-memory-utilization-gb to %.4f. To disable, set "
                        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.",
                        current_gb,
                        equiv_gb,
                        suggested_gb,
                    )
                else:
                    suggested_gb = round(current_gb + cg_gb_delta, 4)
                    logger.warning(
                        "CUDA graph memory profiling is disabled "
                        "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0). "
                        "Without it, CUDA graph memory is not accounted for "
                        "during KV cache allocation, which may require lowering "
                        "--gpu-memory-utilization-gb to avoid OOM. Consider "
                        "re-enabling it (the default as of v0.21.0) and "
                        "increasing --gpu-memory-utilization-gb from %.4f to "
                        "%.4f.",
                        current_gb,
                        suggested_gb,
                    )
            else:
                total_mem = self.init_snapshot.total_memory
                current_util = self.cache_config.gpu_memory_utilization
                cg_util_delta = cudagraph_memory_estimate / total_mem
                if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:
                    equiv_util = round(current_util - cg_util_delta, 4)
                    suggested_util = min(
                        round(current_util + cg_util_delta, 4),
                        1.0,
                    )
                    logger.info(
                        "CUDA graph memory profiling is enabled (default since "
                        "v0.21.0). The current --gpu-memory-utilization=%.4f is "
                        "equivalent to --gpu-memory-utilization=%.4f without "
                        "CUDA graph memory profiling. To maintain the same "
                        "effective KV cache size as before, increase "
                        "--gpu-memory-utilization to %.4f. To disable, set "
                        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.",
                        current_util,
                        equiv_util,
                        suggested_util,
                    )
                else:
                    suggested_util = min(
                        round(current_util + cg_util_delta, 4),
                        1.0,
                    )
                    logger.warning(
                        "CUDA graph memory profiling is disabled "
                        "(VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0). "
                        "Without it, CUDA graph memory is not accounted for "
                        "during KV cache allocation, which may require lowering "
                        "--gpu-memory-utilization to avoid OOM. Consider "
                        "re-enabling it (the default as of v0.21.0) and increasing "
                        "--gpu-memory-utilization from %.4f to %.4f.",
                        current_util,
                        suggested_util,
                    )

'''
    text, _ = replace_between(
        text,
        "        if cudagraph_memory_estimate > 0:\n",
        "        return self._reserve_mm_ipc_gpu_memory(\n",
        cudagraph_block,
        "GPU worker CUDA graph memory suggestions",
        already="--gpu-memory-utilization-gb=%.4f",
    )

    text, _ = replace_once(
        text,
        '"different video backend, or increase gpu_memory_utilization."\n',
        '"different video backend, or increase gpu_memory_utilization or "\n'
        '                "gpu_memory_utilization_gb."\n',
        "GPU worker multimodal reserve error",
        already='"gpu_memory_utilization_gb."\n',
    )
    text = text.replace(
        '\n"gpu_memory_utilization_gb."\n',
        '\n                "gpu_memory_utilization_gb."\n',
        1,
    )

    if "Requested GPU memory is {requested_memory_config}" not in text:
        text, _ = replace_once(
            text,
            "            msg = (\n"
            "                f\"Free memory on device \"\n"
            "                f\"({format_gib(self.init_snapshot.free_memory)}/\"\n"
            "                f\"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. \"\n"
            "                f\"Desired GPU memory utilization is \"\n"
            "                f\"({self.cache_config.gpu_memory_utilization}, \"\n"
            "                f\"{format_gib(self.requested_memory)} GiB). \"\n"
            "                f\"Actual usage is {format_gib(self.model_runner.model_memory_usage)} \"\n"
            "                f\"GiB for weight, {format_gib(self.peak_activation_memory)} GiB \"\n"
            "                f\"for peak activation, {format_gib(self.non_torch_memory)} GiB \"\n"
            "                f\"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} \"\n"
            "                f\"GiB for CUDAGraph memory. Replace gpu_memory_utilization \"\n"
            "                f\"config with `--kv-cache-memory=\"\n",
            "            if self.cache_config.gpu_memory_utilization_gb is not None:\n"
            "                requested_memory_config = (\n"
            "                    f\"--gpu-memory-utilization-gb=\"\n"
            "                    f\"{self.cache_config.gpu_memory_utilization_gb}\"\n"
            "                )\n"
            "            else:\n"
            "                requested_memory_config = (\n"
            "                    f\"--gpu-memory-utilization=\"\n"
            "                    f\"{self.cache_config.gpu_memory_utilization}\"\n"
            "                )\n"
            "\n"
            "            msg = (\n"
            "                f\"Free memory on device \"\n"
            "                f\"({format_gib(self.init_snapshot.free_memory)}/\"\n"
            "                f\"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. \"\n"
            "                f\"Requested GPU memory is {requested_memory_config} \"\n"
            "                f\"({format_gib(self.requested_memory)} GiB). \"\n"
            "                f\"Actual usage is {format_gib(self.model_runner.model_memory_usage)} \"\n"
            "                f\"GiB for weight, {format_gib(self.peak_activation_memory)} GiB \"\n"
            "                f\"for peak activation, {format_gib(self.non_torch_memory)} GiB \"\n"
            "                f\"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} \"\n"
            "                f\"GiB for CUDAGraph memory. Replace memory utilization config \"\n"
            "                f\"with `--kv-cache-memory=\"\n",
            "GPU worker warmup suggestion message",
        )

    write(path, original, text)


def patch_messages() -> None:
    replacements: dict[str, list[tuple[str, str, str, str | None]]] = {
        "vllm/config/compilation.py": [
            (
                '"gpu_memory_utilization."\n',
                '"gpu_memory_utilization or gpu_memory_utilization_gb."\n',
                "Mamba cudagraph cache error",
                '"gpu_memory_utilization or gpu_memory_utilization_gb."\n',
            ),
        ],
        "vllm/v1/core/kv_cache_utils.py": [
            (
                '            "Try increasing `gpu_memory_utilization` when initializing the engine "\n'
                '            "(this flag also controls CPU memory reservation on the CPU "\n'
                '            "backend, despite its name). "\n',
                '            "Try increasing `gpu_memory_utilization` or "\n'
                '            "`gpu_memory_utilization_gb` when initializing the engine "\n'
                '            "(`gpu_memory_utilization` also controls CPU memory reservation "\n'
                '            "on the CPU backend, despite its name). "\n',
                "KV cache no-memory error",
                "`gpu_memory_utilization_gb` when initializing the engine",
            ),
            (
                '            f"Try increasing `gpu_memory_utilization` (which also controls "\n'
                '            f"CPU memory on the CPU backend) or decreasing `max_model_len` "\n'
                '            f"when initializing the engine. "\n',
                '            f"Try increasing `gpu_memory_utilization` or "\n'
                '            f"`gpu_memory_utilization_gb`, or decreasing `max_model_len` "\n'
                '            f"when initializing the engine (`gpu_memory_utilization` also "\n'
                '            f"controls CPU memory on the CPU backend). "\n',
                "KV cache insufficient-memory error",
                'f"`gpu_memory_utilization_gb`, or decreasing `max_model_len`',
            ),
            (
                '            "to serve even a single token. Try increasing `gpu_memory_utilization`."\n',
                '            "to serve even a single token. Try increasing `gpu_memory_utilization` "\n'
                '            "or `gpu_memory_utilization_gb`."\n',
                "KV cache auto-fit error",
                '"or `gpu_memory_utilization_gb`."\n',
            ),
        ],
        "vllm/v1/worker/gpu_model_runner.py": [
            (
                '                    "`max_num_seqs` or `gpu_memory_utilization` when "\n'
                '                    "initializing the engine."\n',
                '                    "`max_num_seqs`, `gpu_memory_utilization`, or "\n'
                '                    "`gpu_memory_utilization_gb` when initializing the engine."\n',
                "GPU model runner sampler OOM message",
                '                    "`max_num_seqs`, `gpu_memory_utilization`, or "\n'
                '                    "`gpu_memory_utilization_gb` when initializing the engine."\n',
            ),
            (
                '                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "\n'
                '                    "initializing the engine."\n',
                '                    "lowering `max_num_seqs`, `gpu_memory_utilization`, or "\n'
                '                    "`gpu_memory_utilization_gb` when initializing the engine."\n',
                "GPU model runner pooler OOM message",
                '                    "lowering `max_num_seqs`, `gpu_memory_utilization`, or "\n'
                '                    "`gpu_memory_utilization_gb` when initializing the engine."\n',
            ),
        ],
    }

    for rel, items in replacements.items():
        path, text = read(rel)
        original = text
        for old, new, description, already in items:
            text, _ = replace_once(text, old, new, description, already=already)
        write(path, original, text)


def patch_usage_stats() -> None:
    path, text = read("vllm/v1/utils.py")
    original = text

    text, _ = replace_once(
        text,
        '            "gpu_memory_utilization": vllm_config.cache_config.gpu_memory_utilization,\n',
        '            "gpu_memory_utilization": vllm_config.cache_config.gpu_memory_utilization,\n'
        '            "gpu_memory_utilization_gb": (\n'
        "                vllm_config.cache_config.gpu_memory_utilization_gb\n"
        "            ),\n",
        "usage stats gpu_memory_utilization_gb",
        already='"gpu_memory_utilization_gb": (',
    )

    write(path, original, text)


patch_cache_config()
patch_engine_args()
patch_llm_entrypoint()
patch_request_memory()
patch_gpu_worker()
patch_messages()
patch_usage_stats()

for path in sorted(changed_paths):
    try:
        ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as exc:
        die(f"Syntax check failed for {path}: {exc}")

if changed_paths:
    for path in sorted(changed_paths):
        print(f"[gpu-mem-util-gb] Patched {path.relative_to(root)}")
else:
    print("[gpu-mem-util-gb] --gpu-memory-utilization-gb support is already applied; skipping.")
PY

echo "=====> You can now use --gpu-memory-utilization-gb to specify reserved memory in GiB"
