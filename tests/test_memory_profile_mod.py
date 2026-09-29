#!/usr/bin/env python3
"""CPU-only regression tests for startup observation and profile aggregation."""
import asyncio
from contextlib import asynccontextmanager
import fcntl
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

PROJECT = Path(__file__).resolve().parents[1]
MOD = PROJECT / "mods/memory-profile"
sys.path.insert(0, str(MOD))


def load(name):
    spec = importlib.util.spec_from_file_location("memory_profile_" + name, MOD / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCHER, PROBE, CARD, COLLECT, REPORT = [load(name) for name in ("patch", "probe", "profile_card", "collect", "report")]

WORKER = '''
from .utils import request_memory
class Worker:
    def init_device(self):
        self.requested_memory = request_memory(self.init_snapshot, self.cache_config)
    def load_model(self): pass
    def determine_available_memory(self): return 100
    def initialize_from_config(self, kv_cache_config): pass
    def compile_or_warm_up_model(self): return "timings"
'''
GC = 'def freeze_gc_heap():\n    gc.collect()\n    gc.freeze()\n'
API = '@asynccontextmanager\nasync def lifespan(app):\n    yield\n'


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "vllm"
        self.env = {"VLLM_MEMORY_PROFILE_DIR": str(self.root / "output"),
                    "VLLM_MEMORY_PROFILE_RUN_ID": "model-run", "VLLM_MEMORY_PROFILE_HOST_ID": "node-a"}
        for name, text in (("v1/worker/gpu_worker.py", WORKER), ("utils/gc_utils.py", GC),
                           ("entrypoints/launchers/utils/server_utils.py", API)):
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def test_install_is_idempotent_and_preserves_source(self):
        with patch.dict(os.environ, self.env):
            config = PATCHER.install(self.package)
            self.assertEqual(PATCHER.install(self.package), config)
        worker = (self.package / "v1/worker/gpu_worker.py").read_text()
        self.assertTrue(worker.startswith(WORKER.rstrip()))
        self.assertEqual(worker.count(PATCHER.MARKER), 1)
        self.assertEqual(json.loads((Path(config["host_directory"]) / "manifest.json").read_text()), config)

    def test_rejects_unsupported_source_before_any_write(self):
        target = self.package / "utils/gc_utils.py"
        target.write_text("def changed_gc(): pass\n")
        with patch.dict(os.environ, self.env), self.assertRaises(ValueError):
            PATCHER.install(self.package)
        self.assertEqual((self.package / "v1/worker/gpu_worker.py").read_text(), WORKER)
        self.assertFalse((self.root / "output").exists())

    def test_legacy_lifespan_location(self):
        original = self.package / "entrypoints/launchers/utils/server_utils.py"
        original.unlink()
        target = self.package / "entrypoints/openai/api_server.py"
        target.parent.mkdir(parents=True)
        target.write_text(API)
        with patch.dict(os.environ, self.env):
            PATCHER.install(self.package)
        self.assertIn(PATCHER.MARKER, target.read_text())

    def test_invalid_configuration_does_not_write_sources(self):
        for key, value in (("VLLM_MEMORY_PROFILE_RUN_ID", "../escape"),
                           ("VLLM_MEMORY_PROFILE_INTERVAL", "nan"),
                           ("VLLM_MEMORY_PROFILE_DIR", "relative")):
            with self.subTest(key=key), patch.dict(os.environ, {**self.env, key: value}), self.assertRaises(ValueError):
                PATCHER.install(self.package)
        self.assertEqual((self.package / "v1/worker/gpu_worker.py").read_text(), WORKER)

    def test_run_script_uses_cpu_only_sampler_and_writes_card(self):
        (self.package / "__init__.py").write_text('raise RuntimeError("Do not import vLLM")\n')
        env = {**os.environ, **self.env, "VLLM_PACKAGE_ROOT": str(self.package), "VLLM_MEMORY_PROFILE_DURATION": "0.1"}
        result = subprocess.run(["bash", str(MOD / "run.sh")], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        directory = self.root / "output/model-run/node-a"
        self.assertTrue((directory / "host.jsonl").exists())
        # Run a bounded monitor in the foreground too; the file lock makes
        # duplicate invocations harmless. No Torch/vLLM import is necessary.
        result = subprocess.run([sys.executable, str(MOD / "probe.py"), "--manifest", str(self.package / "_spark_memory_profile.json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        # Card generation is tested directly below; the background process may
        # still own the lock at this instant.
        card = CARD.write_card([directory], directory / "checked.yaml")
        self.assertEqual(card["status"], "incomplete")
        (directory / "STOP").touch()
        with (directory / "monitor.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)


class ProbeTests(unittest.TestCase):
    def tearDown(self):
        PROBE._worker = None
        PROBE._role = "process"
        PROBE._warned.clear()

    def test_shared_storage_is_counted_once_across_views_and_models(self):
        def tensor(device, pointer, size):
            storage = SimpleNamespace(data_ptr=lambda: pointer, nbytes=lambda: size)
            return SimpleNamespace(device=SimpleNamespace(type=device, __str__=lambda: device), untyped_storage=lambda: storage)
        class Device:
            def __init__(self, kind): self.type = kind
            def __str__(self): return self.type
        a = tensor("cuda", 1, 4096)
        b = tensor("cuda", 2, 1024)
        c = tensor("cpu", 1, 32)
        for item in (a, b, c): item.device = Device(item.device.type)
        result = PROBE.storage_inventory({"layer1": [a, b], "layer2": a, "state": (a, c)})
        self.assertEqual(result, {"cuda_storage_bytes": 5120, "cpu_storage_bytes": 32, "unique_storage_count": 3})

    def test_worker_return_values_and_original_failures_are_preserved(self):
        class Worker:
            def init_device(self): return "device"
            def load_model(self): raise RuntimeError("original failure")
            def determine_available_memory(self): return 100
            def initialize_from_config(self, kv_cache_config): return kv_cache_config
            def compile_or_warm_up_model(self): return "timings"
        namespace = {"Worker": Worker, "request_memory": lambda snapshot, cache: 90}
        PROBE.install_worker(namespace)
        worker = Worker()
        with patch.object(PROBE, "record", side_effect=OSError("full output disk")), patch.object(PROBE, "metadata", return_value={}):
            self.assertEqual(worker.init_device(), "device")
            self.assertEqual(worker.determine_available_memory(), 100)
            with self.assertRaisesRegex(RuntimeError, "original failure"):
                worker.load_model()

    def test_exact_admission_snapshot_is_recorded_even_on_failure(self):
        class Worker:
            def init_device(self): pass
            def load_model(self): pass
            def determine_available_memory(self): pass
            def initialize_from_config(self, kv_cache_config): pass
            def compile_or_warm_up_model(self): pass
        request = Mock(side_effect=ValueError("insufficient memory"))
        namespace = {"Worker": Worker, "request_memory": request}
        PROBE.install_worker(namespace)
        snapshot = SimpleNamespace(free_memory=20, total_memory=100)
        with patch.object(PROBE, "record") as record, self.assertRaises(ValueError):
            namespace["request_memory"](snapshot, SimpleNamespace(gpu_memory_utilization=.8))
        self.assertEqual(record.call_args.kwargs["snapshot"], {"free_memory": 20, "total_memory": 100})
        request.assert_called_once()

    def test_gc_wrapper_observes_without_additional_cleanup(self):
        order = []
        original = lambda: order.append("original")
        with patch.object(PROBE, "record", side_effect=lambda phase: order.append(phase)):
            PROBE.wrap_gc(original)()
        self.assertEqual(order, ["before_startup_gc", "original", "after_startup_gc"])

    def test_metadata_does_not_serialize_credentials_or_arbitrary_config(self):
        config = SimpleNamespace(model_config=SimpleNamespace(model="org/model", hf_config=None),
                                 api_key="must-not-record", hf_token="must-not-record")
        worker = SimpleNamespace(vllm_config=config)
        with patch.dict(os.environ, {"HF_TOKEN": "must-not-record", "B12X_AUTOTUNE": "0"}), \
             patch.dict(sys.modules, {"torch": SimpleNamespace()}):
            result = PROBE.metadata(worker)
        encoded = json.dumps(result)
        self.assertNotIn("must-not-record", encoded)
        self.assertEqual(result["configuration"]["environment"]["B12X_AUTOTUNE"], "0")

    def test_api_readiness_is_only_recorded_after_startup(self):
        order = []
        @asynccontextmanager
        async def original(app):
            order.append("startup")
            yield
            order.append("shutdown")
        async def run():
            async with PROBE.wrap_lifespan(original)(None):
                order.append("serve")
        with patch.object(PROBE, "record", side_effect=lambda phase: order.append(phase)):
            asyncio.run(run())
        self.assertEqual(order, ["api_startup", "startup", "api_ready", "serve", "shutdown"])


class CardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def host(self, host_id, rank, *, world=2, ready=True, api=False, run_id="run", failures=False):
        directory = self.root / host_id
        directory.mkdir()
        manifest = {"schema": CARD.SCHEMA, "run_id": run_id, "host_id": host_id, "hostname": host_id,
                    "sample_interval_seconds": .5, "process_interval_seconds": 5,
                    "heap_trim_patch_detected": True, "source_sha256": {}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        samples = [{"elapsed_seconds": t, "host": {"MemTotal": 1000, "unavailable_bytes": amount},
                    "process_namespace_pss_bytes": amount // 10} for t, amount in ((0, 100), (1, 800), (4, 600), (10, 950))]
        (directory / "host.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
        metadata = {"configuration": {"model_config": {"model": "org/model"}, "parallel_config": {"world_size": world, "data_parallel_size": 1}}}
        base = {"pid": rank + 100, "rank": rank, "data_parallel_rank": 0, "rank_key": f"dp0/rank{rank}",
                "role": "worker", "host_id": host_id, "run_id": run_id, "host": {"unavailable_bytes": 600}, "cpu": {"Pss": 20}}
        events = [{**base, "phase": "device_initialized", "elapsed_seconds": .5, "metadata": metadata}]
        if ready:
            events.append({**base, "phase": "worker_ready", "elapsed_seconds": 3, "metadata": metadata,
                           "cuda": {"allocated_bytes": 400, "reserved_bytes": 450},
                           "model_storage": {"cuda_storage_bytes": 250},
                           "kv_cache": {"storage": {"cuda_storage_bytes": 100}}})
        if failures:
            events.append({**base, "phase": "worker_failed", "elapsed_seconds": 3.5, "exception_type": "RuntimeError"})
        if api:
            events.append({**base, "pid": 500, "role": "api", "phase": "api_ready", "elapsed_seconds": 4})
        (directory / "events.jsonl").write_text("")  # Unrelated filename is ignored.
        (directory / "events-100.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
        return directory

    def test_merges_all_ranks_without_double_counting_hosts_or_runtime_peak(self):
        first = self.host("host-a", 0, api=True)
        second = self.host("host-b", 1)
        result = CARD.write_card([first, second], self.root / "card.yaml")
        self.assertEqual(yaml.safe_load((self.root / "card.yaml").read_text()), result)
        self.assertEqual((self.root / "card.yaml").stat().st_mode & 0o777, 0o644)
        self.assertEqual(result["status"], "startup_observed")
        self.assertEqual(len(result["hosts"]), 2)
        self.assertEqual(result["hosts"][0]["startup_peak_increment_bytes"], 700)
        self.assertEqual(result["ranks"][0]["non_kv_torch_at_ready"]["other_allocated_bytes_including_graphs"], 50)
        self.assertFalse(result["evaluation"]["automatic_admission_enabled"])

    def test_missing_failed_and_duplicate_ranks_are_incomplete(self):
        first = self.host("host-a", 0, api=True)
        result = CARD.build_card([first])
        self.assertEqual(result["coverage"]["missing_ranks"], ["dp0/rank1"])
        self.assertEqual(result["status"], "incomplete")
        second = self.host("host-b", 1, ready=False, failures=True)
        self.assertEqual(CARD.build_card([first, second])["status"], "incomplete")
        third = self.host("host-c", 0)
        self.assertEqual(CARD.build_card([first, third])["coverage"]["duplicate_ranks"], ["dp0/rank0"])

    def test_kv_phase_peaks_exclude_later_serving_usage(self):
        directory = self.host("host-a", 0, world=1, api=True)
        path = directory / "events-100.jsonl"
        event = CARD.jsonl(path)[0]
        with path.open("a") as stream:
            stream.write(json.dumps({**event, "phase": "before_kv_allocation", "elapsed_seconds": 2}) + "\n")
        host = CARD.build_card([directory])["hosts"][0]
        self.assertEqual(host["before_first_serving_kv_allocation_peak_increment_bytes"], 700)
        self.assertEqual(host["from_first_serving_kv_allocation_peak_increment_bytes"], 500)

    def test_card_reports_physical_kv_storage_without_descriptor_alias_sum(self):
        group = {"spec_type": "FullAttentionSpec", "layers": 1, "block_size": 16, "page_size_bytes": 128}
        summary = CARD.kv_summary({"storage": {"cuda_storage_bytes": 1024},
                                   "configured_tensor_bytes": 2048, "groups": [group, group]})
        self.assertEqual(summary["storage"]["cuda_storage_bytes"], 1024)
        self.assertNotIn("configured_tensor_bytes", summary)
        self.assertEqual(summary["group_layouts"], [{"count": 2, **group}])

    def test_mixed_runs_are_rejected(self):
        first = self.host("host-a", 0)
        second = self.host("host-b", 1, run_id="other-run")
        with self.assertRaisesRegex(ValueError, "different runs"):
            CARD.build_card([first, second])

    def test_truncated_tail_is_ignored_but_corrupt_complete_records_fail(self):
        path = self.root / "events.jsonl"
        path.write_text('{"ok":1}\n{"incomplete":')
        self.assertEqual(CARD.jsonl(path), [{"ok": 1}])
        path.write_text('{"ok":1}\n{"bad":}\n')
        with self.assertRaises(ValueError):
            CARD.jsonl(path)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        gib = REPORT.GIB
        self.host = {"host_id": "host-a", "baseline_host_memory": {"unavailable_bytes": 2 * gib, "MemTotal": 16 * gib},
                     "ready_host_memory": {"unavailable_bytes": 8 * gib}, "startup_cutoff_elapsed_seconds": 10,
                     "startup_peak_unavailable_bytes": 9 * gib, "startup_peak_increment_bytes": 7 * gib}
        self.points = [{"phase": phase, "elapsed_seconds": time, "host_unavailable_bytes": host * gib,
                        "cpu_pss_bytes": cpu * gib if cpu is not None else None,
                        "cuda_allocated_bytes": cuda * gib if cuda is not None else None,
                        "cuda_reserved_bytes": cuda * gib if cuda is not None else None}
                       for phase, time, host, cpu, cuda in (("model_loaded", 2, 6, 1, 4), ("before_kv_allocation", 4, 7, None, None),
                                                           ("worker_ready", 10, 8, 2, 6), ("runtime", 11, 15, 3, 7))]
        self.rank = {"rank_key": "dp0/rank0", "pid": 100, "host_id": "host-a", "worker_ready": True,
                     "checkpoints": self.points, "cpu_at_ready": {"Pss": 2 * gib},
                     "kv_cache": {"storage": {"cuda_storage_bytes": gib}, "effective_bytes_per_1000_capacity_tokens": 16 * REPORT.MIB}}
        self.card = {"profile_schema": CARD.SCHEMA, "run_id": "test-run", "recipe": "test-recipe", "status": "incomplete",
                     "hosts": [self.host], "ranks": [self.rank], "api_processes": [],
                     "coverage": {"expected_ranks": ["dp0/rank0", "dp0/rank1"], "ready_ranks": ["dp0/rank0"], "missing_ranks": ["dp0/rank1"]}}
        self.path = self.root / "card.yaml"
        self.path.write_text(yaml.safe_dump(self.card))

    def test_text_report_units_missing_values_and_incomplete_coverage(self):
        report = REPORT.markdown_report(REPORT.load_card(self.path))
        self.assertIn("Ready ranks: 1/2", report)
        self.assertIn("**Missing ranks:**", report)
        self.assertIn("16.000 MiB", report)
        self.assertIn("8.000 GiB (50.0%)", report)
        self.assertIn("unavailable", report)
        self.assertNotIn("runtime", report)

    def test_relative_series_keep_unknown_intervals_and_real_baselines(self):
        points = REPORT.startup_points(self.rank, self.host)
        times, values = REPORT.series(points, "cpu_pss_bytes", relative=True)
        self.assertEqual(times, [2, 4, 10])
        self.assertEqual(values[0], 0)
        self.assertTrue(math.isnan(values[1]))
        self.assertEqual(values[2], 1)
        _, values = REPORT.series(points, "host_unavailable_bytes", relative=True, baseline=2 * REPORT.GIB)
        self.assertEqual(values, [4, 5, 6])

    def test_host_series_do_not_sum_ranks_or_mix_host_clocks(self):
        self.card["ranks"].append({**self.rank, "rank_key": "dp0/rank1", "pid": 101})
        self.card["hosts"].append({**self.host, "host_id": "host-b"})
        self.card["ranks"].append({**self.rank, "host_id": "host-b", "checkpoints": [
            {**self.points[0], "elapsed_seconds": 1, "host_unavailable_bytes": 100 * REPORT.GIB}]})
        points = REPORT.host_points(self.card, self.host)
        self.assertEqual([p["host_unavailable_bytes"] / REPORT.GIB for p in points], [6, 7, 8])

    def test_cpu_stack_waits_for_every_process_and_clears_missing_samples(self):
        api = {"host_id": "host-a", "checkpoints": [
            {"elapsed_seconds": 3, "cpu_pss_bytes": 3 * REPORT.GIB},
            {"elapsed_seconds": 9, "cpu_pss_bytes": 4 * REPORT.GIB}]}
        times, components, total = REPORT.cpu_stack_series([self.rank, api], self.host)
        self.assertEqual(times, [2, 3, 4, 9, 10])
        self.assertTrue(math.isnan(components[1][0]))  # API has no earlier sample.
        self.assertTrue(math.isnan(total[0]))
        self.assertEqual(total[1], 4)  # Worker 1 GiB plus API 3 GiB.
        self.assertTrue(math.isnan(total[2]))  # Explicitly missing worker PSS.
        self.assertTrue(math.isnan(total[3]))
        self.assertEqual(total[4], 6)  # Worker recovers; API's last sample carried forward.

    def test_ready_cpu_sum_is_host_local_and_missing_values_are_not_zero(self):
        self.card["api_processes"] = [{"host_id": "host-a", "pid": 200, "cpu_at_ready": {"Pss": 3 * REPORT.GIB}}]
        self.card["hosts"].append({"host_id": "host-b"})
        self.card["ranks"].append({**self.rank, "host_id": "host-b", "cpu_at_ready": None})
        report = REPORT.markdown_report(self.card)
        self.assertIn("| host-a | 2/2 | 5.000 GiB |", report)
        self.assertIn("| host-b | 0/1 | unavailable |", report)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting dependency not installed")
    def test_absolute_cpu_plot_stacks_processes_and_labels_the_estimated_sum(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.card["api_processes"] = [{"host_id": "host-a", "pid": 200, "checkpoints": [
            {"elapsed_seconds": 9, "cpu_pss_bytes": 3 * REPORT.GIB}]}]
        with patch.object(plt, "close"):
            REPORT.plot_card(self.card, self.root / "stacked.png")
            figure = plt.gcf()
        try:
            cpu = figure.axes[1]
            total = next(line for line in cpu.lines if line.get_label().startswith("Estimated total"))
            self.assertEqual(total.get_ydata()[-1], 5)
            self.assertTrue(all(math.isnan(value) for value in total.get_ydata()[:-1]))
            self.assertEqual(len(cpu.collections), 2)
            self.assertTrue(any("~5.00 GiB" in text.get_text() for text in cpu.texts))
        finally:
            plt.close(figure)

    def test_stdout_needs_no_matplotlib_and_input_is_never_overwritten(self):
        # A module that raises on import proves text-only operation is independent.
        (self.root / "matplotlib.py").write_text('raise RuntimeError("Must not import plotting")\n')
        env = {**os.environ, "PYTHONPATH": str(self.root)}
        result = subprocess.run([sys.executable, str(MOD / "report.py"), str(self.path)], capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith("# vLLM startup memory report"))
        before = self.path.read_bytes()
        result = subprocess.run([sys.executable, str(MOD / "report.py"), str(self.path), "-o", str(self.path), "--no-plot"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_schema_and_unknown_host_are_rejected(self):
        for data in ({**self.card, "profile_schema": "future-schema"}, {**self.card, "hosts": []}):
            self.path.write_text(yaml.safe_dump(data))
            with self.assertRaises(ValueError):
                REPORT.load_card(self.path)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting dependency not installed")
    def test_real_multi_host_plot_preserves_missing_data_and_separate_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.card["hosts"].append({"host_id": "host-b", "baseline_host_memory": None})
        with patch.object(plt, "close"):
            REPORT.plot_card(self.card, self.root / "chart.png", relative=True)
            figure = plt.gcf()
        try:
            self.assertEqual(len(figure.axes), 6)
            self.assertEqual(list(figure.axes[0].lines[0].get_ydata()), [4, 5, 6])
            cpu = figure.axes[1].lines[0]
            self.assertEqual(list(cpu.get_xdata()), [2, 4, 10])
            self.assertTrue(math.isnan(cpu.get_ydata()[1]))
            self.assertEqual(len(figure.axes[3].lines), 0)
            self.assertTrue((self.root / "chart.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        finally:
            plt.close(figure)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional plotting dependency not installed")
    def test_cli_writes_report_with_relative_encoded_chart_link(self):
        output = self.root / "reports/report.md"
        chart = self.root / "plots/my chart.svg"
        result = subprocess.run([sys.executable, str(MOD / "report.py"), str(self.path), "-o", str(output),
                                 "--plot", str(chart)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("../plots/my%20chart.svg", output.read_text())
        self.assertIn("<svg", chart.read_text())


class CollectionTests(unittest.TestCase):
    def archive(self, name, *, symlink=False):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            info = tarfile.TarInfo(name)
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/target"
            else:
                info.size = 2
            archive.addfile(info, None if symlink else io.BytesIO(b"{}"))
        data.seek(0)
        return data

    def test_collection_extracts_regular_files_and_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            COLLECT.extract(self.archive("node/manifest.json"), root)
            self.assertEqual((root / "node/manifest.json").read_text(), "{}")
            for name, symlink in (("../outside", False), ("/absolute", False), ("link", True)):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    COLLECT.extract(self.archive(name, symlink=symlink), root)


if __name__ == "__main__":
    unittest.main()
