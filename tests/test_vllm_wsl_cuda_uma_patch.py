#!/usr/bin/env python3

import ast
import importlib.util
import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCHER_PATH = PROJECT_DIR / "docker/patch_vllm_wsl_cuda_uma.py"
SPEC = importlib.util.spec_from_file_location("wsl_cuda_uma_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


ENVS = '''
import os

environment_variables = {
    "VLLM_WSL2_ENABLE_PIN_MEMORY": lambda: bool(
        int(os.getenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "0"))
    ),
}
'''

MEM_UTILS = '''
import psutil
import torch

from vllm.platforms import current_platform


def release_device_memory_under_pressure(device: torch.device) -> bool:
    if device.type != "cuda" or not current_platform.is_integrated_gpu(device.index):
        return False
    return True


class MemorySnapshot:
    def measure(self) -> None:
        device = self.device_
        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        if current_platform.is_integrated_gpu(device.index):
            # Use host availability to include reclaimable OS memory on UMA.
            self.free_memory = psutil.virtual_memory().available
        self.cuda_memory = self.total_memory - self.free_memory
'''


class WslCudaUmaPatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root / "vllm/utils/mem_utils.py"
        self.target.parent.mkdir(parents=True)
        self.target.write_text(MEM_UTILS)
        self.envs = self.root / "vllm/envs.py"
        self.envs.write_text(ENVS)
        (self.root / "vllm/__init__.py").write_text(
            'raise RuntimeError("Patching must not import vLLM")\n'
        )

    def run_patch(self, *, installed=False):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root)
        return subprocess.run(
            [sys.executable, str(PATCHER_PATH)]
            + (["--installed"] if installed else [str(self.root)]),
            cwd=PROJECT_DIR,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_source_patch_is_idempotent_and_preserves_other_policies(self):
        result = self.run_patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        patched = self.target.read_text()
        self.assertNotEqual(patched, MEM_UTILS)
        self.assertEqual(self.envs.read_text(), ENVS)

        def release_function(source):
            return next(
                ast.dump(node)
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef)
                and node.name == "release_device_memory_under_pressure"
            )

        self.assertEqual(release_function(patched), release_function(MEM_UTILS))
        result = self.run_patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already patched", result.stdout)
        self.assertEqual(self.target.read_text(), patched)

    def test_installed_wheel_is_patched_without_importing_vllm(self):
        result = self.run_patch(installed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        patched = self.target.read_text()
        self.assertNotEqual(patched, MEM_UTILS)
        result = self.run_patch(installed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already patched", result.stdout)
        self.assertEqual(self.target.read_text(), patched)

    def test_only_cuda_wsl_uma_bypasses_host_memory(self):
        tree = ast.parse(PATCHER.patch_mem_utils(MEM_UTILS))
        # Execute the patched snapshot with fake memory APIs; no GPU is needed.
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        code = compile(tree, "patched_snapshot", "exec")
        for integrated, cuda, wsl, host_free in itertools.product(
            (True, False), (True, False), (True, False), (20, 100)
        ):
            with self.subTest(
                integrated=integrated, cuda=cuda, wsl=wsl, host_free=host_free
            ):
                platform = Mock()
                platform.is_integrated_gpu.return_value = integrated
                platform.is_cuda.return_value = cuda
                memory_info = Mock(return_value=(40, 120))
                host_info = Mock(return_value=SimpleNamespace(available=host_free))
                namespace = {
                    "current_platform": platform,
                    "in_wsl": lambda: wsl,
                    "torch": SimpleNamespace(
                        accelerator=SimpleNamespace(get_memory_info=memory_info)
                    ),
                    "psutil": SimpleNamespace(virtual_memory=host_info),
                }
                exec(code, namespace)
                snapshot = namespace["MemorySnapshot"]()
                # ROCm also uses torch's cuda device type; backend is decisive.
                snapshot.device_ = SimpleNamespace(type="cuda", index=1)
                snapshot.measure()
                use_host = integrated and not (cuda and wsl)
                expected_free = host_free if use_host else 40
                self.assertEqual(snapshot.free_memory, expected_free)
                self.assertEqual(snapshot.total_memory, 120)
                self.assertEqual(snapshot.cuda_memory, 120 - expected_free)
                self.assertEqual(host_info.call_count, int(use_host))
                memory_info.assert_called_once_with(snapshot.device_)
                platform.is_integrated_gpu.assert_called_once_with(1)

    def test_existing_import_is_not_duplicated(self):
        source = MEM_UTILS.replace(
            "from vllm.platforms import current_platform",
            "from vllm.platforms import current_platform\n"
            "from vllm.platforms.interface import in_wsl, Platform",
        )
        patched = PATCHER.patch_mem_utils(source)
        self.assertEqual(patched.count("from vllm.platforms.interface import"), 1)
        self.assertEqual(PATCHER.patch_mem_utils(patched), patched)

    def test_parenthesized_original_guard(self):
        source = MEM_UTILS.replace(
            "if current_platform.is_integrated_gpu(device.index):",
            "if (\n            current_platform.is_integrated_gpu(device.index)\n        ):",
        )
        patched = PATCHER.patch_mem_utils(source)
        self.assertEqual(
            ast.dump(ast.parse(patched)),
            ast.dump(ast.parse(PATCHER.patch_mem_utils(MEM_UTILS))),
        )

    def test_unknown_guard_fails_without_partial_writes(self):
        source = MEM_UTILS.replace(
            "if current_platform.is_integrated_gpu(device.index):",
            "if current_platform.is_integrated_gpu(device.index) and unknown_policy():",
        )
        self.target.write_text(source)
        result = self.run_patch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected exactly one known MemorySnapshot UMA guard", result.stderr)
        self.assertEqual(self.target.read_text(), source)
        self.assertEqual(self.envs.read_text(), ENVS)

    def test_runner_patches_after_installing_cached_wheels(self):
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        runner = dockerfile.split("FROM ${CUDA_IMAGE} AS runner\n", 1)[1]
        script = "patch_vllm_wsl_cuda_uma.py"
        self.assertIn(f"COPY docker/{script} /tmp/vllm-patches/{script}", runner)
        self.assertGreater(
            runner.index(f"RUN python3 /tmp/vllm-patches/{script} --installed"),
            runner.index("uv pip install /workspace/flashinfer-wheels/*.whl"),
        )


if __name__ == "__main__":
    unittest.main()
