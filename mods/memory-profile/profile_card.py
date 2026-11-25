#!/usr/bin/env python3
"""Merge memory-profile run directories into a model/recipe YAML profile card."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import yaml


SCHEMA = "spark-vllm-memory-profile/v1"


def jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            # A concurrent writer may not have finished its final record yet.
            if not line.endswith("\n"):
                break
            rows.append(json.loads(line))
    return rows


def host_directories(inputs):
    directories = set()
    for path in map(Path, inputs):
        if (path / "manifest.json").is_file():
            directories.add(path.resolve())
        else:
            directories.update(p.parent.resolve() for p in path.rglob("manifest.json"))
    if not directories:
        raise ValueError("No memory-profile manifest.json files found")
    return sorted(directories)


def last(rows, phase):
    return next((row for row in reversed(rows) if row["phase"] == phase), {})


def cuda_summary(cuda):
    if not cuda:
        return None
    result = {key: value for key, value in cuda.items() if key != "pinned_host_allocator"}
    pinned = cuda.get("pinned_host_allocator")
    if pinned is not None:
        result["pinned_host_allocator"] = {key: value for key, value in pinned.items()
                                           if key.endswith((".current", ".peak"))}
    return result


def checkpoint(row):
    cuda, cpu = row.get("cuda") or {}, row.get("cpu") or {}
    return {"phase": row["phase"], "elapsed_seconds": round(row["elapsed_seconds"], 3),
            "host_unavailable_bytes": row.get("host", {}).get("unavailable_bytes"),
            "cpu_pss_bytes": cpu.get("Pss"),
            "cuda_allocated_bytes": cuda.get("allocated_bytes"),
            "cuda_reserved_bytes": cuda.get("reserved_bytes")}


def kv_summary(kv):
    if kv is None:
        return None
    # Tensor descriptors may alias the same physical pool. Keep their sum in
    # the raw trace only; the card's footprint is the deduplicated storage.
    result = {key: value for key, value in kv.items() if key not in ("groups", "configured_tensor_bytes")}
    layouts = Counter(json.dumps(group, sort_keys=True) for group in kv.get("groups", []))
    result["group_layouts"] = [{"count": count, **json.loads(group)} for group, count in sorted(layouts.items())]
    return result


def rank_card(events):
    ready = last(events, "worker_ready")
    kv = ready.get("kv_cache") or last(events, "kv_allocated").get("kv_cache")
    model = ready.get("model_storage")
    cuda = ready.get("cuda")
    non_kv = None
    if cuda and kv and kv.get("storage"):
        size = kv["storage"]["cuda_storage_bytes"]
        non_kv = {
            "allocated_bytes": cuda["allocated_bytes"] - size,
            "reserved_bytes": cuda["reserved_bytes"] - size,
            "registered_model_storage_bytes": model["cuda_storage_bytes"] if model else None,
            "other_allocated_bytes_including_graphs": cuda["allocated_bytes"] - size - model["cuda_storage_bytes"] if model else None,
            "attribution": "Derived residual of simultaneous ready counters; not a complete native/driver inventory.",
        }
    captures = []
    before = None
    kv_time = last(events, "kv_allocated").get("elapsed_seconds", float("inf"))
    for row in events:
        if row["phase"] == "before_capture_model":
            before = row
        elif row["phase"] == "after_capture_model" and before:
            start, end = before.get("cuda"), row.get("cuda")
            captures.append({
                "phase": "serving" if row["elapsed_seconds"] > kv_time else "profiling",
                "reported_free_memory_delta_bytes": row.get("returned_bytes"),
                "torch_allocated_delta_bytes": end["allocated_bytes"] - start["allocated_bytes"] if start and end else None,
                "torch_reserved_delta_bytes": end["reserved_bytes"] - start["reserved_bytes"] if start and end else None,
            })
            before = None
    metadata = ready.get("metadata") or last(events, "device_initialized").get("metadata")
    return {
        "rank_key": events[0]["rank_key"], "rank": events[0]["rank"],
        "data_parallel_rank": events[0]["data_parallel_rank"],
        "host_id": events[0]["host_id"], "pid": events[0]["pid"],
        "worker_ready": bool(ready), "metadata": metadata,
        "utilization_check": {
            "snapshot": last(events, "utilization_check").get("snapshot"),
            "gpu_memory_utilization": last(events, "utilization_check").get("gpu_memory_utilization"),
            "requested_memory_bytes": last(events, "utilization_check_passed").get("requested_memory_bytes"),
        },
        "kv_budget_bytes": last(events, "kv_budget_decision").get("kv_budget_bytes"),
        "kv_cache": kv_summary(kv), "model_storage_at_ready": model,
        "model_loader_reported_bytes": last(events, "model_loaded").get("model_loader_reported_bytes"),
        "cuda_allocator_at_ready": cuda_summary(cuda), "cpu_at_ready": ready.get("cpu"),
        "native_heap_at_ready": ready.get("native_heap"), "non_kv_torch_at_ready": non_kv,
        "graphs": {"captures": captures,
                   "profiling_estimate_bytes": last(events, "after_profile_cudagraph_memory").get("returned_bytes"),
                   "deltas_are_not_isolated_graph_object_sizes": True},
        "startup_gc": [{"phase": row["phase"], "cpu": row.get("cpu"), "native_heap": row.get("native_heap")}
                       for row in events if row["phase"] in ("before_startup_gc", "after_startup_gc")],
        "checkpoints": [checkpoint(row) for row in events],
        "failures": [row for row in events if row["phase"] == "worker_failed"],
    }


def build_card(inputs, *, local=False):
    hosts, ranks, apis, manifests, all_events = [], [], [], [], []
    for directory in host_directories(inputs):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported manifest schema in {directory}")
        manifests.append(manifest)
        events = sorted((row for path in sorted(directory.glob("events-*.jsonl")) for row in jsonl(path)),
                        key=lambda row: row["elapsed_seconds"])
        if any(row.get("host_id") != manifest["host_id"] or row.get("run_id") != manifest["run_id"] for row in events):
            raise ValueError(f"Mixed host/run events in {directory}")
        all_events.extend(events)
        workers = {}
        for row in events:
            if row["role"] == "worker":
                workers.setdefault((row["rank_key"], row["pid"]), []).append(row)
        ranks.extend(rank_card(rows) for rows in workers.values())
        for pid in sorted({row["pid"] for row in events if row["role"] == "api"}):
            rows = [row for row in events if row["pid"] == pid]
            ready = last(rows, "api_ready")
            apis.append({"host_id": manifest["host_id"], "pid": pid,
                         "ready": bool(ready), "cpu_at_ready": ready.get("cpu"),
                         "native_heap_at_ready": ready.get("native_heap"),
                         "cuda_at_ready": cuda_summary(ready.get("cuda")),
                         "startup_gc": [{"phase": row["phase"], "cpu": row.get("cpu"), "native_heap": row.get("native_heap")}
                                        for row in rows if row["phase"] in ("before_startup_gc", "after_startup_gc")],
                         "checkpoints": [checkpoint(row) for row in rows]})
        samples = jsonl(directory / "host.jsonl")
        baseline = samples[0] if samples else {}
        ready_times = [row["elapsed_seconds"] for row in events if row["phase"] in ("worker_ready", "api_ready")]
        all_workers_ready = bool(workers) and all(last(rows, "worker_ready") for rows in workers.values())
        # Headless workers have no API process. A later API marker extends this
        # local cutoff; no cross-host clock comparison or summation is performed.
        cutoff = max(ready_times) if ready_times and all_workers_ready else None
        startup_samples = [row for row in samples + events if cutoff is None or row["elapsed_seconds"] <= cutoff]
        baseline_used = baseline.get("host", {}).get("unavailable_bytes")
        usages = [row["host"]["unavailable_bytes"] for row in startup_samples if row.get("host", {}).get("unavailable_bytes") is not None]
        pss = [row["process_namespace_pss_bytes"] for row in startup_samples if "process_namespace_pss_bytes" in row]
        final = max((row for row in startup_samples if row.get("host")), key=lambda row: row["elapsed_seconds"], default={})
        peak = max(usages) if usages else None
        kv_start = min((row["elapsed_seconds"] for row in events if row["phase"] == "before_kv_allocation"), default=None)
        def phase_peak(before):
            if kv_start is None or baseline_used is None:
                return None
            values = [row["host"]["unavailable_bytes"] for row in startup_samples
                      if row.get("host", {}).get("unavailable_bytes") is not None
                      and (row["elapsed_seconds"] < kv_start) == before]
            return max(values) - baseline_used if values else None
        hosts.append({
            "host_id": manifest["host_id"], "hostname": manifest["hostname"],
            "baseline_host_memory": baseline.get("host"),
            "startup_peak_unavailable_bytes": peak,
            "startup_peak_increment_bytes": peak - baseline_used if peak is not None and baseline_used is not None else None,
            "before_first_serving_kv_allocation_peak_increment_bytes": phase_peak(True),
            "from_first_serving_kv_allocation_peak_increment_bytes": phase_peak(False),
            "startup_peak_process_namespace_pss_bytes": max(pss) if pss else None,
            "ready_host_memory": final.get("host") if cutoff is not None else None,
            "startup_cutoff_elapsed_seconds": cutoff,
            "observed_workers_ready": all_workers_ready,
            "peaks_are_sampled_lower_bounds": True,
            "sample_interval_seconds": manifest["sample_interval_seconds"],
            "sampling_covers_startup": bool(cutoff is not None and samples and samples[-1]["elapsed_seconds"] >= cutoff),
            "process_interval_seconds": manifest["process_interval_seconds"],
            "heap_trim_patch_detected": manifest["heap_trim_patch_detected"],
            "source_sha256": manifest["source_sha256"],
            "driver_version": manifest.get("driver_version"),
            "image": manifest.get("image"),
            "probe_sha256": manifest.get("probe_sha256"),
        })
    run_ids = {item["run_id"] for item in manifests}
    if len(run_ids) != 1:
        raise ValueError("Select one run ID; profiles from different runs cannot be merged")
    if len({host["host_id"] for host in hosts}) != len(hosts):
        raise ValueError("Duplicate physical host ID; do not count the same host twice")
    metadata = [rank["metadata"] for rank in ranks if rank["metadata"]]
    models = {m["configuration"]["model_config"]["model"] for m in metadata}
    if len(models) > 1:
        raise ValueError("Different models cannot share a run/profile card")
    expected = set()
    topologies = set()
    for m in metadata:
        parallel = m["configuration"]["parallel_config"]
        world, dp = parallel.get("world_size"), parallel.get("data_parallel_size") or 1
        if world:
            topologies.add((world, dp))
            expected.update(f"dp{d}/rank{r}" for d in range(dp) for r in range(world))
    if len(topologies) > 1:
        raise ValueError("Conflicting parallel configurations in one run")
    counts = Counter(rank["rank_key"] for rank in ranks)
    recorded = set(counts)
    ready = {rank["rank_key"] for rank in ranks if rank["worker_ready"]}
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    errors = sorted({tuple(error) for row in all_events for error in row.get("instrumentation_errors", [])})
    failures = [row for row in all_events if row["phase"] in ("worker_failed", "api_failed")]
    complete = bool(expected) and ready == expected and not duplicates and not failures
    observed_api = any(api["ready"] for api in apis)
    return {
        "profile_schema": SCHEMA, "run_id": next(iter(run_ids)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": next(iter(models), None),
        "recipe": manifests[0].get("recipe"),
        "scope": "host" if local else "run",
        "status": "startup_observed" if complete and observed_api and not errors and all(host["sampling_covers_startup"] for host in hosts) else "incomplete",
        "coverage": {
            "expected_ranks": sorted(expected), "recorded_ranks": sorted(recorded),
            "ready_ranks": sorted(ready), "missing_ranks": sorted(expected - recorded),
            "duplicate_ranks": duplicates, "all_workers_ready": complete,
            "api_readiness_observed": observed_api, "instrumentation_errors": [list(error) for error in errors],
            "workload": "startup_only; later host samples are retained in JSONL",
            "maximum_serving_workload_tested": False,
        },
        "hosts": hosts, "ranks": sorted(ranks, key=lambda rank: (rank["rank_key"], rank["pid"])),
        "api_processes": apis,
        "evaluation": {
            "automatic_admission_enabled": False,
            "reason": "Observed startup profile, not a guarantee for other hosts, settings or serving workloads.",
            "units": "bytes unless a field names another unit",
            "do_not_add_host_and_cuda_memory": True,
            "do_not_sum_host_measurements_across_ranks": True,
            "compatibility_fingerprint_complete": False,
            "notes": [
                "On UMA, CUDA and CPU allocations compete for the same host memory.",
                "MemTotal minus MemAvailable includes unrelated host activity; cgroup/PSS can omit GPU-backed pages.",
                "CUDA counters cover the PyTorch allocator; graph deltas and residuals are not exact native ownership.",
                "Registered model storage excludes unregistered backend weights and workspaces.",
                "KV bytes per 1000 equivalent capacity tokens is specific to this configuration, especially for hybrid models.",
                "Reserve startup peak headroom even when post-warmup cleanup reduces readiness usage.",
                "Reprofile changes in model revision, parallelism, kernels, quantization, graphs, batching or speculation.",
            ],
        },
    }


def write_card(inputs, output, *, local=False):
    card = build_card(inputs, local=local)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=".profile-", delete=False) as stream:
            temporary = Path(stream.name)
            # Containers commonly run as root while the bind mount is read by
            # the host user. NamedTemporaryFile otherwise leaves the card 0600.
            os.fchmod(stream.fileno(), 0o644)
            yaml.safe_dump(card, stream, sort_keys=False)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Run/host directories, including copies gathered from other nodes")
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()
    try:
        card = write_card(args.inputs, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"memory-profile: {error}\n")
    print(f"{card['status']}: {len(card['ranks'])} rank(s), {len(card['hosts'])} host(s); {args.output}")


if __name__ == "__main__":
    main()
