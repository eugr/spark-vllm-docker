#!/usr/bin/env python3

import ast
import hashlib
import os
import runpy
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
MOD = PROJECT_DIR / "mods/uma-fix/run.sh"
PIN_MEMORY_ENV = "VLLM_WSL2_ENABLE_PIN_MEMORY"

ENVS = '''
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    VLLM_WSL2_ENABLE_PIN_MEMORY: bool = False

environment_variables = {
    "VLLM_WSL2_ENABLE_PIN_MEMORY": lambda: bool(
        int(os.getenv("VLLM_WSL2_ENABLE_PIN_MEMORY", "0"))
    ),
}
'''

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

UNRELATED_CALL_SITE = '''
import torch


def profile(device):
    return torch.accelerator.get_memory_info(device)
'''


class UmaFixModTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name) / "site-packages"

        self.envs = self.root / "vllm/envs.py"
        self.envs.parent.mkdir(parents=True)
        self.envs.write_text(textwrap.dedent(ENVS).lstrip())

        self.mem_utils = self.root / "vllm/utils/mem_utils.py"
        self.mem_utils.parent.mkdir(parents=True)
        self.mem_utils.write_text(textwrap.dedent(MEM_UTILS).lstrip())

        self.call_site = self.root / "vllm/v1/worker/gpu_worker.py"
        self.call_site.parent.mkdir(parents=True)
        self.call_site.write_text(textwrap.dedent(UNRELATED_CALL_SITE).lstrip())

        self.env = os.environ.copy()
        self.env["PYTHON_ROOT"] = str(self.root)

    def run_mod(self, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(MOD)],
            cwd=PROJECT_DIR,
            env=self.env,
            check=check,
            text=True,
            capture_output=True,
        )

    def test_defaults_pin_memory_and_preserves_raw_cuda_reporting(self):
        call_site_hash = hashlib.sha256(self.call_site.read_bytes()).digest()

        first = self.run_mod()
        self.assertIn(
            f"Set {PIN_MEMORY_ENV}=1 as the vLLM default.", first.stdout
        )
        self.assertIn(
            "Disabled host UMA accounting under WSL", first.stdout
        )

        env_source = self.envs.read_text()
        ast.parse(env_source)
        self.assertIn(f"{PIN_MEMORY_ENV}: bool = True", env_source)
        self.assertIn(
            f'os.getenv("{PIN_MEMORY_ENV}", "1")', env_source
        )

        namespace = runpy.run_path(str(self.envs))
        get_pin_memory = namespace["environment_variables"][PIN_MEMORY_ENV]
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(PIN_MEMORY_ENV, None)
            self.assertTrue(get_pin_memory())
        with patch.dict(os.environ, {PIN_MEMORY_ENV: "0"}):
            self.assertFalse(get_pin_memory())
        with patch.dict(os.environ, {PIN_MEMORY_ENV: "1"}):
            self.assertTrue(get_pin_memory())

        mem_source = self.mem_utils.read_text()
        ast.parse(mem_source)
        self.assertIn(
            "from vllm.platforms.interface import in_wsl", mem_source
        )
        self.assertIn("or in_wsl()", mem_source)
        self.assertIn("not in_wsl()", mem_source)
        self.assertIn(
            "self.free_memory, self.total_memory = "
            "torch.accelerator.get_memory_info(device)",
            mem_source,
        )
        self.assertNotIn("cuda_mem_get_info", mem_source)
        self.assertNotIn("import_pynvml", mem_source)
        self.assertEqual(
            call_site_hash,
            hashlib.sha256(self.call_site.read_bytes()).digest(),
        )

    def test_application_is_idempotent(self):
        self.run_mod()
        patched_hashes = {
            path: hashlib.sha256(path.read_bytes()).digest()
            for path in (self.envs, self.mem_utils, self.call_site)
        }

        second = self.run_mod()
        self.assertIn(
            f"{PIN_MEMORY_ENV} already defaults to 1; skipping.",
            second.stdout,
        )
        self.assertIn(
            "WSL memory accounting is already patched; skipping.",
            second.stdout,
        )
        self.assertEqual(
            patched_hashes,
            {
                path: hashlib.sha256(path.read_bytes()).digest()
                for path in (self.envs, self.mem_utils, self.call_site)
            },
        )

    def test_missing_upstream_pin_memory_support_fails_clearly(self):
        self.envs.write_text(
            "import os\n\nenvironment_variables = {}\n"
        )

        result = self.run_mod(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            f"{PIN_MEMORY_ENV} runtime default was not found",
            result.stderr,
        )

if __name__ == "__main__":
    unittest.main()
