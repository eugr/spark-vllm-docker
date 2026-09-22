#!/usr/bin/env python3
"""CPU-only checks for InstantTensor's vLLM memory-accounting patch."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCHER_PATH = PROJECT_DIR / "docker/patch_instanttensor_vllm_memory.py"
SPEC = importlib.util.spec_from_file_location("instanttensor_memory_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# Budget calculation and rejection from InstantTensor 0.2.0, with unrelated
# backend selection and I/O configuration omitted. No GPU packages are imported.
SOURCE = '''class safe_open:
    def _determine_io_params(self, config):
        max_free_mem_usage = config.max_free_mem_usage
        if max_free_mem_usage is None:
            max_free_mem_usage = 0.5

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        avail_bytes = int(free_bytes * max_free_mem_usage)

        if self.process_group is not None:
            avail_bytes_tensor = torch.tensor([avail_bytes], device=self.device)
            dist.all_reduce(avail_bytes_tensor, op=dist.ReduceOp.MIN, group=self.process_group)
            avail_bytes = avail_bytes_tensor.item()

        self._device_memory_budget = avail_bytes

    def _finalize_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > self._device_memory_budget:
            raise RuntimeError(
                f"buffer_size ({self.buffer_size} B) exceeds device memory "
                f"budget ({self._device_memory_budget} B)"
            )
'''
NEIGHBOR = '''

def unrelated_memory_query():
    return torch.cuda.mem_get_info()
'''


class InstantTensorVllmMemoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root / "instanttensor/_impl.py"
        self.target.parent.mkdir(parents=True)
        self.target.write_text(SOURCE + NEIGHBOR)
        for package in ("instanttensor", "vllm", "torch"):
            directory = self.root / package
            directory.mkdir(exist_ok=True)
            (directory / "__init__.py").write_text(
                f'raise RuntimeError("Patching must not import {package}")\n'
            )

    def run_patch(self, *, installed=False):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root)
        return subprocess.run(
            [sys.executable, str(PATCHER_PATH)]
            + (["--installed"] if installed else [str(self.target)]),
            cwd=PROJECT_DIR, env=env, capture_output=True, text=True,
        )

    def make_loader(self, *, available, fraction=None, remote_budget=None):
        snapshot = Mock(return_value=SimpleNamespace(free_memory=available))
        module = ModuleType("vllm.utils.mem_utils")
        module.MemorySnapshot = snapshot
        modules = {
            "vllm": ModuleType("vllm"),
            "vllm.utils": ModuleType("vllm.utils"),
            "vllm.utils.mem_utils": module,
        }

        # A mutable scalar stands in for the CUDA tensor used by all_reduce.
        result = SimpleNamespace(value=0)

        def make_tensor(values, *, device):
            result.value = values[0]
            return SimpleNamespace(item=lambda: result.value)

        def all_reduce(value, *, op, group):
            self.assertIs(op, dist.ReduceOp.MIN)
            self.assertIs(group, loader.process_group)
            result.value = min(result.value, remote_budget)

        torch = SimpleNamespace(
            cuda=SimpleNamespace(mem_get_info=Mock(
                side_effect=AssertionError("Must use vLLM's memory policy")
            )),
            tensor=Mock(side_effect=make_tensor),
        )
        dist = SimpleNamespace(
            ReduceOp=SimpleNamespace(MIN=object()),
            all_reduce=Mock(side_effect=all_reduce),
        )
        namespace = {"torch": torch, "dist": dist}
        exec(PATCHER.patch_memory_query(SOURCE), namespace)
        loader = namespace["safe_open"]()
        loader.device = SimpleNamespace(type="cuda", index=1)
        loader.process_group = object() if remote_budget is not None else None
        with patch.dict(sys.modules, modules):
            loader._determine_io_params(SimpleNamespace(max_free_mem_usage=fraction))
        snapshot.assert_called_once_with(device=loader.device)
        torch.cuda.mem_get_info.assert_not_called()
        if remote_budget is None:
            dist.all_reduce.assert_not_called()
        else:
            dist.all_reduce.assert_called_once()
            torch.tensor.assert_called_once_with(
                [int(available * (0.5 if fraction is None else fraction))],
                device=loader.device,
            )
        return loader

    def test_vllm_available_memory_allows_large_tensor_despite_small_cuda_reading(self):
        # Native UMA can have ample reclaimable memory even when CUDA reports
        # only 1,356,980,224 bytes free (the reported 678,490,112-byte budget).
        available = 16 * 1024**3
        loader = self.make_loader(available=available)
        self.assertEqual(loader._device_memory_budget, available // 2)
        loader._finalize_buffer_size(1_059_061_760)

    def test_low_vllm_availability_still_rejects_oversized_buffers(self):
        # If vLLM selects the low CUDA reading, e.g. under WSL, the patch must
        # honor it instead of substituting host RAM or bypassing the guard.
        loader = self.make_loader(available=1_356_980_224)
        self.assertEqual(loader._device_memory_budget, 678_490_112)
        with self.assertRaisesRegex(RuntimeError, "exceeds device memory budget"):
            loader._finalize_buffer_size(1_059_061_760)

    def test_explicit_budget_fraction_and_boundary_are_preserved(self):
        loader = self.make_loader(available=1001, fraction=0.25)
        self.assertEqual(loader._device_memory_budget, 250)
        loader._finalize_buffer_size(250)
        with self.assertRaises(RuntimeError):
            loader._finalize_buffer_size(251)

    def test_distributed_budget_is_still_the_minimum_across_ranks(self):
        for remote_budget in (300, 900):
            with self.subTest(remote_budget=remote_budget):
                loader = self.make_loader(available=1000, remote_budget=remote_budget)
                self.assertEqual(loader._device_memory_budget, min(500, remote_budget))
                with self.assertRaises(RuntimeError):
                    loader._finalize_buffer_size(loader._device_memory_budget + 1)

    def test_patch_is_scoped_and_idempotent_without_importing_packages(self):
        for installed in (False, True):
            with self.subTest(installed=installed):
                self.target.write_text(SOURCE + NEIGHBOR)
                result = self.run_patch(installed=installed)
                self.assertEqual(result.returncode, 0, result.stderr)
                patched = self.target.read_text()
                self.assertNotEqual(patched, SOURCE + NEIGHBOR)
                self.assertTrue(patched.endswith(NEIGHBOR))
                self.assertEqual(PATCHER.patch_memory_query(patched), patched)
                result = self.run_patch(installed=installed)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("already patched", result.stdout)
                self.assertEqual(self.target.read_text(), patched)

    def test_unknown_or_ambiguous_sources_fail_without_writing(self):
        for source in (
            NEIGHBOR,
            SOURCE + SOURCE,
            SOURCE.replace("_determine_io_params", "_select_io_params"),
            SOURCE.replace("torch.cuda.mem_get_info()", "torch.cuda.mem_get_info(self.device)"),
            SOURCE.replace("int(free_bytes * max_free_mem_usage)", "free_bytes"),
            SOURCE.replace(PATCHER.ORIGINAL, PATCHER.ORIGINAL * 2),
            SOURCE.replace("class safe_open:", "class safe_open("),
        ):
            with self.subTest(source=source):
                self.target.write_text(source)
                result = self.run_patch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unable to patch", result.stderr)
                self.assertEqual(self.target.read_text(), source)

    def test_runner_patches_after_dependency_installation(self):
        runner = (PROJECT_DIR / "Dockerfile").read_text().split(
            "FROM ${CUDA_IMAGE} AS runner\n", 1
        )[1]
        script = "patch_instanttensor_vllm_memory.py"
        self.assertIn(f"COPY docker/{script} /tmp/instanttensor-patches/{script}", runner)
        patch_position = runner.index(
            f"RUN python3 /tmp/instanttensor-patches/{script} --installed"
        )
        self.assertGreater(patch_position, runner.rindex("uv pip install"))


if __name__ == "__main__":
    unittest.main()
