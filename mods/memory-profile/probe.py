"""Startup instrumentation and a standalone, CPU-only host sampler.

Imported inside vLLM as vllm._spark_memory_profile. The sampler runs this file
directly so observing the server never imports another copy of vLLM or Torch.
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import ctypes
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading
import time
import weakref


_manifest = None
_worker = None
_role = "process"
_lock = threading.Lock()
_warned = set()

CONFIG_FIELDS = {
    "model_config": ("model", "revision", "dtype", "quantization", "max_model_len", "enforce_eager"),
    "parallel_config": ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "decode_context_parallel_size", "prefill_context_parallel_size", "world_size", "nnodes", "distributed_executor_backend"),
    "cache_config": ("gpu_memory_utilization", "kv_cache_memory_bytes", "cache_dtype", "block_size", "enable_prefix_caching", "mamba_cache_mode", "mamba_ssm_cache_dtype", "cpu_offload_gb"),
    "scheduler_config": ("max_num_seqs", "max_num_batched_tokens", "max_num_encoder_input_tokens", "encoder_cache_size", "enable_chunked_prefill"),
    "speculative_config": ("method", "num_speculative_tokens", "model"),
    "compilation_config": ("mode", "backend", "cudagraph_mode", "cudagraph_capture_sizes", "max_cudagraph_capture_size"),
    "kernel_config": ("moe_backend", "linear_backend", "enable_b12x_autotune"),
    "load_config": ("load_format",),
}
ENV_FIELDS = (
    "B12X_AUTOTUNE", "B12X_POLICY_MODE", "B12X_STATE_COMPILE_WORKERS",
    "B12X_WEIGHTS_COMPILE_WORKERS", "B12X_BIND_COMPILE_WORKERS",
    "VLLM_PLE_TABLE_MEMORY", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_AOT_COMPILE",
    "VLLM_USE_MEGA_AOT_ARTIFACT", "VLLM_SSM_CONV_STATE_LAYOUT", "VLLM_MXFP8_LM_HEAD",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "CUDA_MODULE_LOADING",
)


def scalar(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): scalar(item) for key, item in value.items()}
    # Config enums and torch.dtype are safe; never serialize arbitrary objects.
    if hasattr(value, "value"):
        return scalar(value.value)
    return str(value) if type(value).__name__ == "dtype" else None


def proc_counters(path):
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(":")
            words = value.split()
            if words and words[0].isdigit():
                result[key] = int(words[0]) * (1024 if words[-1] == "kB" else 1)
    except (OSError, ValueError):
        pass
    return result


def host_memory():
    mem = proc_counters("/proc/meminfo")
    keys = ("MemTotal", "MemAvailable", "MemFree", "Cached", "SReclaimable", "Shmem", "SwapTotal", "SwapFree")
    result = {key: mem.get(key) for key in keys}
    result["unavailable_bytes"] = (
        mem["MemTotal"] - mem["MemAvailable"]
        if "MemTotal" in mem and "MemAvailable" in mem else None
    )
    return result


def process_memory(pid):
    result = proc_counters(f"/proc/{pid}/smaps_rollup")
    return {key: result.get(key) for key in (
        "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Rss", "Anonymous", "Swap", "Locked"
    )}


def process_start(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


class Mallinfo2(ctypes.Structure):
    _fields_ = [(key, ctypes.c_size_t) for key in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
        "fsmblks", "uordblks", "fordblks", "keepcost",
    )]


def native_heap():
    try:
        query = ctypes.CDLL(None).mallinfo2
        query.argtypes = []
        query.restype = Mallinfo2
        info = query()
        return {key: getattr(info, key) for key in ("arena", "uordblks", "fordblks", "hblkhd")}
    except (AttributeError, OSError):
        return None


def manifest():
    global _manifest
    if _manifest is None:
        _manifest = json.loads(Path(__file__).with_name("_spark_memory_profile.json").read_text())
    return _manifest


def stamp(config):
    return {"time_unix": time.time(), "elapsed_seconds": time.monotonic() - config["start_monotonic"]}


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def observe(function, *args, **kwargs):
    """Instrumentation errors must not replace model results or exceptions."""
    try:
        return function(*args, **kwargs)
    except Exception as error:
        key = (getattr(function, "__name__", type(function).__name__), type(error).__name__)
        if key not in _warned:
            _warned.add(key)
            print(f"[memory-profile] {key[0]} failed ({key[1]}); profile may be incomplete", file=sys.stderr, flush=True)
        return None


def cuda_memory(device=None):
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return None
    config = manifest()
    if config["synchronize"]:
        torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    result = {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "device_free_bytes": free, "device_total_bytes": total,
        "peak_allocated_since_vllm_reset_bytes": torch.cuda.max_memory_allocated(device),
        "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
    }
    if hasattr(torch.cuda.memory, "host_memory_stats"):
        result["pinned_host_allocator"] = torch.cuda.memory.host_memory_stats()
    return result


def metadata(worker):
    config = worker.vllm_config
    fields = {name: {key: scalar(getattr(getattr(config, name, None), key, None)) for key in keys}
              for name, keys in CONFIG_FIELDS.items()}
    mm = getattr(config.model_config, "multimodal_config", None)
    fields["multimodal_config"] = {key: scalar(getattr(mm, key, None)) for key in (
        "language_model_only", "mm_processor_cache_gb", "mm_processor_cache_type", "skip_mm_profiling", "limit_per_prompt",
    )}
    draft = getattr(getattr(config, "speculative_config", None), "draft_model_config", None)
    fields["speculative_config"]["draft_model"] = scalar(getattr(draft, "model", None))
    fields["environment"] = {key: os.environ[key] for key in ENV_FIELDS if key in os.environ}
    versions = {}
    for name in ("vllm", "torch", "b12x", "flashinfer-python", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    torch = sys.modules["torch"]
    device = getattr(worker, "device", None)
    hardware = {}
    if device is not None and torch.cuda.is_initialized():
        props = torch.cuda.get_device_properties(device)
        from vllm.platforms import current_platform
        hardware = {"name": props.name, "total_memory_bytes": props.total_memory,
                    "compute_capability": [props.major, props.minor],
                    "integrated": current_platform.is_integrated_gpu(device.index),
                    "cuda_runtime": torch.version.cuda}
    hf_config = getattr(config.model_config, "hf_config", None)
    return {"configuration": fields, "configuration_sha256": hashlib.sha256(
                json.dumps(fields, sort_keys=True).encode()).hexdigest(),
            "model_revision_resolved": getattr(hf_config, "_commit_hash", None),
            "versions": versions, "hardware": hardware}


def storage_inventory(values):
    storages = {}
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif hasattr(value, "untyped_storage") and value.device.type != "meta":
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    visit(values)
    return {"cuda_storage_bytes": sum(size for (device, _), size in storages.items() if device.startswith("cuda")),
            "cpu_storage_bytes": sum(size for (device, _), size in storages.items() if device == "cpu"),
            "unique_storage_count": len(storages)}


def model_inventory(worker):
    runner = worker.model_runner
    models = [getattr(runner, "model", None)]
    for name in ("speculator", "drafter"):
        models.append(getattr(getattr(runner, name, None), "model", None))
    tensors = []
    for model in models:
        if model is not None and hasattr(model, "parameters"):
            tensors.extend(model.parameters())
            tensors.extend(model.buffers())
    return storage_inventory(tensors) if tensors else None


def kv_inventory(worker, config):
    from vllm.v1.core import kv_cache_utils
    cache = getattr(worker.model_runner, "kv_caches", None)
    inventory = storage_inventory(cache) if cache is not None else None
    tokens = concurrency = None
    if hasattr(kv_cache_utils, "get_kv_cache_capacity"):
        tokens, concurrency = kv_cache_utils.get_kv_cache_capacity(worker.vllm_config, config)
    groups = []
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        groups.append({"spec_type": type(spec).__name__, "layers": len(group.layer_names),
                       "block_size": spec.block_size, "page_size_bytes": spec.page_size_bytes,
                       "draft_group": getattr(group, "is_eagle_group", False),
                       "host_resident": getattr(group, "host_resident", False)})
    allocated = inventory["cuda_storage_bytes"] if inventory else None
    return {"storage": inventory, "num_blocks": config.num_blocks, "groups": groups,
            "configured_tensor_bytes": sum(t.size for t in config.kv_cache_tensors),
            "equivalent_capacity_tokens": tokens, "max_concurrency": concurrency,
            "effective_bytes_per_1000_capacity_tokens": allocated * 1000 / tokens if allocated is not None and tokens else None,
            "capacity_ratio_is_not_a_universal_token_slope": True}


def record(phase, worker=None, **extra):
    config = manifest()
    if worker is None and _worker is not None:
        worker = _worker()
    row = {**stamp(config), "phase": phase, "pid": os.getpid(),
           "process_start_ticks": process_start(os.getpid()),
           "role": "worker" if worker is not None else _role,
           "host_id": config["host_id"], "run_id": config["run_id"],
           "host": host_memory(), "cpu": process_memory(os.getpid()),
           "native_heap": native_heap(), **extra}
    row["cuda"] = observe(cuda_memory, getattr(worker, "device", None))
    if worker is not None:
        row["rank"] = worker.rank
        row["data_parallel_rank"] = getattr(worker.parallel_config, "data_parallel_rank", 0)
        row["rank_key"] = f"dp{row['data_parallel_rank']}/rank{worker.rank}"
        row["worker_memory"] = {key: scalar(getattr(worker, key, None)) for key in (
            "requested_memory", "total_consumed", "peak_activation_memory",
            "cudagraph_memory_estimate", "available_kv_cache_memory_bytes",
        )}
        row["model_loader_reported_bytes"] = getattr(getattr(worker, "model_runner", None), "model_memory_usage", None)
    if _warned:
        row["instrumentation_errors"] = [list(key) for key in sorted(_warned)]
    with _lock:
        append(Path(config["host_directory"]) / f"events-{os.getpid()}.jsonl", row)
    if phase in ("worker_ready", "api_ready"):
        print(f"[memory-profile] {phase} {row.get('rank_key', '')}: measurements in {config['host_directory']}", flush=True)


def install_worker(namespace):
    cls = namespace["Worker"]
    original_request = namespace["request_memory"]
    @functools.wraps(original_request)
    def request(snapshot, cache_config, *args, **kwargs):
        # Persist the exact snapshot passed to vLLM, before its admission check.
        observe(record, "utilization_check", snapshot={key: scalar(value) for key, value in vars(snapshot).items() if isinstance(value, (int, float, bool))},
                gpu_memory_utilization=cache_config.gpu_memory_utilization)
        result = original_request(snapshot, cache_config, *args, **kwargs)
        observe(record, "utilization_check_passed", requested_memory_bytes=result)
        return result
    namespace["request_memory"] = request

    def wrap(name, before, after):
        original = getattr(cls, name)
        @functools.wraps(original)
        def measured(self, *args, **kwargs):
            global _worker
            _worker = weakref.ref(self)
            observe(record, before, self)
            try:
                result = original(self, *args, **kwargs)
            except BaseException as error:
                observe(record, "worker_failed", self, failed_phase=name, exception_type=type(error).__name__)
                raise
            extra = {}
            if name == "init_device":
                extra["metadata"] = observe(metadata, self)
            elif name == "load_model":
                extra["model_storage"] = observe(model_inventory, self)
                observe(install_runner, self)
            elif name == "determine_available_memory":
                extra["kv_budget_bytes"] = result
            elif name == "initialize_from_config":
                self._spark_profile_kv_config = args[0] if args else kwargs["kv_cache_config"]
                extra["kv_cache"] = observe(kv_inventory, self, self._spark_profile_kv_config)
            elif name == "compile_or_warm_up_model":
                extra["model_storage"] = observe(model_inventory, self)
                extra["kv_cache"] = observe(kv_inventory, self, self._spark_profile_kv_config)
                extra["metadata"] = observe(metadata, self)
            observe(record, after, self, **extra)
            return result
        setattr(cls, name, measured)
    for spec in (
        ("init_device", "before_device_init", "device_initialized"),
        ("load_model", "before_model_load", "model_loaded"),
        ("determine_available_memory", "before_memory_profile", "kv_budget_decision"),
        ("initialize_from_config", "before_kv_allocation", "kv_allocated"),
        ("compile_or_warm_up_model", "before_compile_warmup", "worker_ready"),
    ):
        wrap(*spec)


def install_runner(worker):
    runner = worker.model_runner
    for name in ("profile_run", "profile_cudagraph_memory", "capture_model"):
        if not hasattr(runner, name):
            continue
        def wrap(original, method):
            @functools.wraps(original)
            def measured(*args, **kwargs):
                observe(record, "before_" + method, worker)
                result = original(*args, **kwargs)
                observe(record, "after_" + method, worker,
                        returned_bytes=result if type(result) in (int, float) else None)
                return result
            return measured
        setattr(runner, name, wrap(getattr(runner, name), name))


def wrap_gc(original):
    @functools.wraps(original)
    def measured(*args, **kwargs):
        observe(record, "before_startup_gc")
        result = original(*args, **kwargs)
        observe(record, "after_startup_gc")
        return result
    return measured


def wrap_lifespan(original):
    @asynccontextmanager
    async def measured(app):
        global _role
        _role = "api"
        observe(record, "api_startup")
        ready = False
        try:
            async with original(app):
                observe(record, "api_ready")
                ready = True
                yield
        except BaseException as error:
            observe(record, "api_shutdown_failed" if ready else "api_failed", exception_type=type(error).__name__)
            raise
    return measured


def sample_host(config, processes=False):
    row = {**stamp(config), "host": host_memory()}
    cgroup = {}
    for name in ("memory.current", "memory.peak", "memory.swap.current"):
        try:
            cgroup[name] = int(Path("/sys/fs/cgroup", name).read_text())
        except (OSError, ValueError):
            pass
    row["cgroup"] = cgroup
    if processes:
        rows = []
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                counters = process_memory(int(path.name))
                if counters["Pss"] is not None:
                    rows.append({"pid": int(path.name), **counters})
        row["processes"] = rows
        row["process_namespace_pss_bytes"] = sum(p["Pss"] for p in rows)
    append(Path(config["host_directory"]) / "host.jsonl", row)


def monitor(config):
    import fcntl
    from profile_card import write_card
    directory = Path(config["host_directory"])
    with (directory / "monitor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        end = time.monotonic() + config["duration_seconds"]
        next_process = next_card = 0
        while time.monotonic() < end and not (directory / "STOP").exists():
            now = time.monotonic()
            observe(sample_host, config, processes=now >= next_process)
            if now >= next_process:
                next_process = now + config["process_interval_seconds"]
            if now >= next_card:
                observe(write_card, [directory], directory / "profile.yaml", local=True)
                next_card = now + 5
            time.sleep(config["sample_interval_seconds"])
        observe(write_card, [directory], directory / "profile.yaml", local=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    if args.baseline:
        sample_host(config, processes=True)
    else:
        monitor(config)
