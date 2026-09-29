#!/usr/bin/env python3
"""Install the opt-in memory profiler without importing vLLM or initializing CUDA."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time
import uuid


MARKER = "# spark-vllm mod: memory-profile v1"
SCHEMA = "spark-vllm-memory-profile/v1"
WORKER_METHODS = ("init_device", "load_model", "determine_available_memory", "initialize_from_config", "compile_or_warm_up_model")


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("Run and host IDs must be 1–96 letters, digits, dots, underscores or hyphens, starting with a letter/digit")
    return value


def patched(source, kind):
    suffixes = {
        "worker": "from vllm._spark_memory_profile import install_worker as _spark_install_memory_profile\n_spark_install_memory_profile(globals())\n",
        "gc": "from vllm._spark_memory_profile import wrap_gc as _spark_profile_gc\nfreeze_gc_heap = _spark_profile_gc(freeze_gc_heap)\n",
        "api": "from vllm._spark_memory_profile import wrap_lifespan as _spark_profile_lifespan\nlifespan = _spark_profile_lifespan(lifespan)\n",
    }
    suffix = "\n" + MARKER + "\n" + suffixes[kind]
    if MARKER in source:
        if source.count(MARKER) != 1 or not source.endswith(suffix):
            raise ValueError(f"Incomplete memory-profile patch in {kind}")
        return source
    tree = ast.parse(source)
    if kind == "worker":
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker"]
        if len(classes) != 1:
            raise ValueError("Expected one GPU Worker class")
        methods = {node.name: node for node in classes[0].body if isinstance(node, ast.FunctionDef)}
        if not set(WORKER_METHODS) <= methods.keys():
            raise ValueError("Unsupported Worker startup methods")
        requests = [node for node in ast.walk(methods["init_device"]) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "request_memory"]
        if len(requests) != 1 or len(requests[0].args) != 2:
            raise ValueError("Expected request_memory(snapshot, cache_config) in init_device")
    else:
        name = "freeze_gc_heap" if kind == "gc" else "lifespan"
        cls = ast.FunctionDef if kind == "gc" else ast.AsyncFunctionDef
        functions = [node for node in tree.body if isinstance(node, cls) and node.name == name]
        if len(functions) != 1:
            raise ValueError(f"Expected one {name} function")
        if kind == "api" and not any(isinstance(node, ast.Name) and node.id == "asynccontextmanager" for node in functions[0].decorator_list):
            raise ValueError("Expected an asynccontextmanager lifespan")
    result = source.rstrip() + "\n" + suffix
    compile(result, f"memory-profile-{kind}", "exec")
    return result


def install(root):
    import yaml  # Validate the card writer's dependency before modifying vLLM.
    del yaml
    targets = {"worker": root / "v1/worker/gpu_worker.py", "gc": root / "utils/gc_utils.py"}
    for relative in ("entrypoints/launchers/utils/server_utils.py", "entrypoints/openai/api_server.py"):
        candidate = root / relative
        if candidate.exists() and any(isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan" for node in ast.parse(candidate.read_text()).body):
            targets["api"] = candidate
            break
    if "api" not in targets:
        raise ValueError("Supported API lifespan hook not found")
    originals = {kind: path.read_text() for kind, path in targets.items()}
    changes = {kind: patched(source, kind) for kind, source in originals.items()}
    manifest_path = root / "_spark_memory_profile.json"
    present = [MARKER in source for source in originals.values()]
    if any(present):
        if not all(present) or not manifest_path.is_file() or not (root / "_spark_memory_profile.py").is_file():
            raise ValueError("Incomplete installed profiler; use a fresh container")
        config = json.loads(manifest_path.read_text())
        requested = os.environ.get("VLLM_MEMORY_PROFILE_RUN_ID")
        if requested and requested != config["run_id"]:
            raise ValueError("Profiler already installed for another run; use a fresh container")
        print(f"[memory-profile] Already installed: {config['host_directory']}")
        return config
    run_id = identifier(os.environ.get("VLLM_MEMORY_PROFILE_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    try:
        host_identity = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        host_identity = socket.gethostname()
    host_id = identifier(os.environ.get("VLLM_MEMORY_PROFILE_HOST_ID") or hashlib.sha256(host_identity.encode()).hexdigest()[:16])
    base = Path(os.environ.get("VLLM_MEMORY_PROFILE_DIR", "/memory-profiles"))
    if not base.is_absolute():
        raise ValueError("VLLM_MEMORY_PROFILE_DIR must be absolute")
    directory = base / run_id / host_id
    config = {"schema": SCHEMA, "run_id": run_id, "host_id": host_id,
              "hostname": socket.gethostname(), "recipe": os.environ.get("VLLM_MEMORY_PROFILE_RECIPE"),
              "image": os.environ.get("VLLM_MEMORY_PROFILE_IMAGE"),
              "host_directory": str(directory), "start_monotonic": time.monotonic(),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "heap_trim_patch_detected": "spark-vllm-docker: trim unused CPU heap pages after startup GC" in originals["gc"],
              "source_sha256": {str(targets[k].relative_to(root)): hashlib.sha256(v.encode()).hexdigest() for k, v in originals.items()}}
    for name, env, default in (("sample_interval_seconds", "INTERVAL", "0.5"),
                               ("process_interval_seconds", "PROCESS_INTERVAL", "5"),
                               ("duration_seconds", "DURATION", "3600")):
        config[name] = float(os.environ.get("VLLM_MEMORY_PROFILE_" + env, default))
        if not math.isfinite(config[name]) or config[name] < 0.1:
            raise ValueError(f"VLLM_MEMORY_PROFILE_{env} must be finite and >= 0.1 seconds")
    sync = os.environ.get("VLLM_MEMORY_PROFILE_SYNC", "1")
    if sync not in ("0", "1"):
        raise ValueError("VLLM_MEMORY_PROFILE_SYNC must be 0 or 1")
    config["synchronize"] = sync == "1"
    try:
        config["driver_version"] = Path("/proc/driver/nvidia/version").read_text().strip()
    except OSError:
        config["driver_version"] = None
    probe = Path(__file__).with_name("probe.py").read_text()
    compile(probe, "probe.py", "exec")
    config["probe_sha256"] = hashlib.sha256(probe.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(config, sort_keys=True, indent=2) + "\n"
    (directory / "manifest.json").write_text(encoded)
    (root / "_spark_memory_profile.py").write_text(probe)
    manifest_path.write_text(encoded)
    # All source layouts and output permissions were checked before these writes.
    for kind, path in targets.items():
        path.write_text(changes[kind])
    print(f"[memory-profile] Installed; output: {directory}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_root", type=Path)
    args = parser.parse_args()
    try:
        install(args.package_root)
    except (OSError, ValueError, SyntaxError) as error:
        parser.exit(1, f"memory-profile: {error}\n")
