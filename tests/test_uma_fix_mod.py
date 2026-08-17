#!/usr/bin/env python3

import ast
import hashlib
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MOD = PROJECT_DIR / "mods/uma-fix/run.sh"

MEM_UTILS = '''
from functools import cache

import psutil
import torch
import torch.types

from vllm.platforms import current_platform

from .mem_constants import GiB_bytes, KiB_bytes, MiB_bytes


def format_gib(b: int) -> str:
    return f"{round(b / GiB_bytes, 2)}"


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    return gpu


def release_device_memory_under_pressure(device: torch.device) -> bool:
    if device.type != "cuda" or not current_platform.is_integrated_gpu(device.index):
        return False
    return True


class MemorySnapshot:
    def measure(self) -> None:
        device = self.device_
        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        if current_platform.is_integrated_gpu(device.index):
            # On UMA (Unified Memory Architecture) platforms where CPU and
            # GPU share physical memory (e.g. GH200, DGX Spark, Jetson Orin),
            # cudaMemGetInfo underreports free memory because it does not
            # account for reclaimable OS memory (page cache, buffers).
            # Use psutil to get the true available memory.
            self.free_memory = psutil.virtual_memory().available
        self.cuda_memory = self.total_memory - self.free_memory
'''

CALL_SITES = {
    "vllm/model_executor/models/gemma4_mm.py": '''
import torch

from vllm.platforms import current_platform


def profile():
    first = torch.accelerator.get_memory_info()
    second = torch.accelerator.get_memory_info()
    return first, second
''',
    "vllm/v1/worker/gpu/model_runner.py": '''
import torch

from vllm.logger import init_logger


class GPUModelRunner:
    def capture(self):
        start = torch.accelerator.get_memory_info()[0]
        end = torch.accelerator.get_memory_info()[0]
        return start - end
''',
    "vllm/v1/worker/gpu/spec_decode/eagle/utils.py": '''
import torch

from vllm.config import VllmConfig


def should_share(w):
    return torch.accelerator.get_memory_info(w.device)[0]
''',
    "vllm/v1/worker/gpu_model_runner.py": '''
import torch

from vllm.logger import init_logger


class GPUModelRunner:
    def capture(self):
        before = torch.accelerator.get_memory_info()[0]
        after = torch.accelerator.get_memory_info()[0]
        return before - after
''',
    "vllm/v1/worker/gpu_worker.py": '''
import torch

from vllm.logger import init_logger


class Worker:
    def sleep(self):
        before = torch.accelerator.get_memory_info()[0]
        after, total = torch.accelerator.get_memory_info()
        return before, after, total
''',
}


class UmaFixModTests(unittest.TestCase):
    def test_latest_accelerator_api_is_patched_idempotently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "site-packages"
            mem_utils = root / "vllm/utils/mem_utils.py"
            mem_utils.parent.mkdir(parents=True)
            mem_utils.write_text(textwrap.dedent(MEM_UTILS).lstrip())

            targets = [mem_utils]
            for relative_path, source in CALL_SITES.items():
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(textwrap.dedent(source).lstrip())
                targets.append(target)

            # Keep this fixture focused on the dynamic memory patcher; the UVA
            # patch is validated separately against current upstream source.
            bin_dir = Path(temp_dir) / "bin"
            bin_dir.mkdir()
            git = bin_dir / "git"
            git.write_text("#!/bin/sh\nexit 0\n")
            git.chmod(0o755)

            env = os.environ.copy()
            env["PYTHON_ROOT"] = str(root)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            first = subprocess.run(
                ["bash", str(MOD)],
                cwd=PROJECT_DIR,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("Patched vllm/utils/mem_utils.py.", first.stdout)
            for relative_path in CALL_SITES:
                self.assertIn(f"Patched {relative_path}.", first.stdout)

            patched_hashes = {
                path: hashlib.sha256(path.read_bytes()).digest() for path in targets
            }
            second = subprocess.run(
                ["bash", str(MOD)],
                cwd=PROJECT_DIR,
                env=env,
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn(
                "vllm/utils/mem_utils.py is already patched; skipping.",
                second.stdout,
            )
            self.assertEqual(
                patched_hashes,
                {path: hashlib.sha256(path.read_bytes()).digest() for path in targets},
            )

            mem_source = mem_utils.read_text()
            ast.parse(mem_source)
            self.assertIn('"visible_device_id_to_physical_device_id"', mem_source)
            self.assertIn(
                'getattr(torch, "accelerator", None), "get_memory_info", None',
                mem_source,
            )
            self.assertIn(
                "cuda_free_memory, cuda_total_memory = cuda_mem_get_info(device)",
                mem_source,
            )
            self.assertNotIn(
                "cuda_free_memory, cuda_total_memory = current_platform.mem_get_info(device)",
                mem_source,
            )
            self.assertIn(
                "if not is_wsl and current_platform.is_integrated_gpu(device_id):",
                mem_source,
            )
            self.assertIn("# On UMA (Unified Memory Architecture)", mem_source)

            for target in targets[1:]:
                source = target.read_text()
                ast.parse(source)
                self.assertIn(
                    "from vllm.utils.mem_utils import cuda_mem_get_info", source
                )
                self.assertNotIn("torch.accelerator.get_memory_info(", source)


if __name__ == "__main__":
    unittest.main()
