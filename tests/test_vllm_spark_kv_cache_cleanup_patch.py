#!/usr/bin/env python3
"""CPU-only checks that released heap pages actually reach the KV budget."""

import ctypes
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCH_PATH = PROJECT_DIR / "docker/patch_vllm_spark_kv_cache_cleanup.py"
TARGET_REL = Path("vllm/v1/worker/gpu_worker.py")
GIB = 1024**3
SOURCE = '''import torch


class Worker:
    def determine_available_memory(self):
        profile_result = self.profile_result
        self.model_runner.profile_run()
        free_gpu_memory = profile_result.after_profile.free_memory
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )
        return self.available_kv_cache_memory_bytes

    def initialize_from_config(self, kv_cache_config):
        """Allocate the KV cache."""
        self.model_runner.initialize_kv_cache(kv_cache_config)
'''
B12X_SOURCE = SOURCE.replace(
    "        free_gpu_memory = profile_result.after_profile.free_memory\n",
    """        final_profile_snapshot = MemorySnapshot(device=self.device)
        late_persistent_memory = max(
            profile_result.after_profile.free_memory
            - final_profile_snapshot.free_memory,
            0,
        )
        free_gpu_memory = final_profile_snapshot.free_memory
""",
).replace(
    "self.requested_memory - profile_result.non_kv_cache_memory",
    "self.requested_memory - profile_result.non_kv_cache_memory - late_persistent_memory",
)


class KvSizingHeapTrimTests(unittest.TestCase):
    def apply_patch(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / TARGET_REL
            target.parent.mkdir(parents=True)
            target.write_text(source)
            result = subprocess.run(
                [sys.executable, str(PATCH_PATH), str(root)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return target.read_text()

    def run_worker(self, source, *, modern=True, trim_bytes=4 * GIB,
                   missing=False, load_error=None, device_type="cuda"):
        events = []
        state = {"free": 20 * GIB}

        class Snapshot:
            free_memory = 20 * GIB

            def measure(self):
                events.append("measure")
                self.free_memory = state["free"]

            def __sub__(self, before):
                return SimpleNamespace(
                    non_torch_memory=before.free_memory - self.free_memory - 60 * GIB,
                )

        profile = SimpleNamespace(
            before_create=SimpleNamespace(free_memory=100 * GIB),
            after_profile=Snapshot(),
            weights_memory=60 * GIB,
            torch_peak_increase=3 * GIB,
            non_kv_cache_memory=83 * GIB,
        )
        if modern:
            profile.total_consumed = 80 * GIB
            profile.transient_peak_headroom = 3 * GIB

        def empty_cache():
            events.append("empty_cache")
            state["free"] += 2 * GIB

        def trim_heap(pad):
            self.assertEqual(pad, 0)
            events.append("trim")
            state["free"] += trim_bytes
            return int(trim_bytes > 0)

        def final_snapshot(*, device):
            self.assertEqual(device, "cuda:0")
            events.append("final_snapshot")
            # Model a late retained allocation to catch double counting or a
            # trim placed after B12X's final device-wide snapshot.
            return SimpleNamespace(free_memory=state["free"] - GIB)

        trim = Mock(side_effect=trim_heap)
        library = SimpleNamespace() if missing else SimpleNamespace(malloc_trim=trim)
        fake_ctypes = SimpleNamespace(
            CDLL=Mock(return_value=library, side_effect=load_error),
            c_size_t=ctypes.c_size_t,
            c_int=ctypes.c_int,
        )
        # No freeze/reset_peak APIs: accidental calls would fail the test.
        fake_gc = SimpleNamespace(collect=lambda: events.append("collect"))
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            synchronize=lambda device: events.append("synchronize"),
            empty_cache=empty_cache,
        ))
        namespace = {"MemorySnapshot": final_snapshot,
                     "logger": SimpleNamespace(info_once=Mock())}
        with patch.dict(sys.modules, {"ctypes": fake_ctypes, "gc": fake_gc,
                                      "torch": fake_torch}):
            exec(compile(self.apply_patch(source), str(TARGET_REL), "exec"), namespace)
            worker = namespace["Worker"]()
            worker.device_config = SimpleNamespace(device_type=device_type)
            worker.device = "cuda:0"
            worker.model_runner = SimpleNamespace(
                profile_run=lambda: events.append("profile"),
                _cleanup_profiling_kv_cache=lambda: events.append("cleanup_kv"),
            )
            worker.requested_memory = 90 * GIB
            worker.profile_result = profile
            budget = worker.determine_available_memory()
        return budget, events, profile, trim, fake_ctypes

    def test_trim_is_charged_back_to_regular_legacy_and_b12x_kv_budgets(self):
        for source, modern in ((SOURCE, False), (SOURCE, True), (B12X_SOURCE, True)):
            with self.subTest(b12x=source == B12X_SOURCE, modern=modern):
                budget, events, profile, trim, libc = self.run_worker(source, modern=modern)
                b12x = source == B12X_SOURCE
                self.assertEqual(budget, (12 if b12x else 13) * GIB)
                self.assertEqual(profile.non_kv_cache_memory, 77 * GIB)
                self.assertEqual(events, ["profile", "cleanup_kv", "collect", "synchronize",
                                          "empty_cache", "trim", "measure"]
                                 + (["final_snapshot"] if b12x else []))
                self.assertEqual(trim.argtypes, [ctypes.c_size_t])
                self.assertIs(trim.restype, ctypes.c_int)
                libc.CDLL.assert_called_once_with(None)
                if modern:
                    self.assertEqual(profile.total_consumed, 74 * GIB)
                    self.assertEqual(profile.transient_peak_headroom, 3 * GIB)
                else:
                    self.assertEqual(profile.non_torch_increase, 14 * GIB)

    def test_unsupported_allocators_preserve_cuda_cleanup_and_budget(self):
        for source in (SOURCE, B12X_SOURCE):
            for kwargs in ({"missing": True}, {"load_error": OSError("unavailable")},
                           {"trim_bytes": 0}):
                with self.subTest(b12x=source == B12X_SOURCE, kwargs=kwargs):
                    budget, events, _, trim, _ = self.run_worker(source, **kwargs)
                    self.assertEqual(budget, (8 if source == B12X_SOURCE else 9) * GIB)
                    self.assertIn("empty_cache", events)
                    self.assertIn("measure", events)
                    if kwargs.get("missing") or kwargs.get("load_error"):
                        trim.assert_not_called()

    def test_non_cuda_path_does_not_load_libc_or_run_cleanup(self):
        budget, events, _, trim, libc = self.run_worker(SOURCE, device_type="xpu")
        self.assertEqual(budget, 7 * GIB)
        self.assertEqual(events, ["profile"])
        trim.assert_not_called()
        libc.CDLL.assert_not_called()

    def test_repeat_application_and_equivalent_existing_cleanup(self):
        for source in (SOURCE, B12X_SOURCE):
            with self.subTest(b12x=source == B12X_SOURCE):
                patched = self.apply_patch(source)
                self.assertEqual(self.apply_patch(patched), patched)
                # Equivalent upstream cleanup must still receive CPU trimming.
                start = patched.index("            # spark-vllm-docker: trim CPU heap")
                end = patched.index("            profile_result.after_profile.measure()")
                without_trim = patched[:start] + patched[end:]
                self.assertEqual(self.apply_patch(without_trim), patched)

    def test_unknown_layout_fails_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / TARGET_REL
            target.parent.mkdir(parents=True)
            source = SOURCE.replace("free_gpu_memory = profile_result.after_profile.free_memory",
                                    "free_gpu_memory = unknown_snapshot()")
            target.write_text(source)
            result = subprocess.run([sys.executable, str(PATCH_PATH), str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_text(), source)

    def test_both_lanes_apply_shared_patch_before_wheel_build(self):
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        builder = dockerfile.split("FROM base AS vllm-builder\n", 1)[1]
        build, runner = builder.split("FROM ${CUDA_IMAGE} AS runner\n", 1)
        command = f"RUN python3 /tmp/vllm-patches/{PATCH_PATH.name} .\n"
        # A standalone RUN without a B12X guard applies to either selected repo.
        self.assertIn(command, build)
        self.assertLess(build.index(command), build.index("uv build --no-build-isolation --wheel ."))
        self.assertNotIn(PATCH_PATH.name, runner)


if __name__ == "__main__":
    unittest.main()
