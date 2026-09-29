#!/usr/bin/env python3
"""Exercise dependency build commands without Docker, downloads, or GPUs."""

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class DependencyBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.log = self.root / "commands.jsonl"
        self.env = {
            **os.environ,
            "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
            "COMMAND_LOG": str(self.log),
            "FLASHINFER_JIT_CACHE_PROVIDER_ARCHS": "12.1a",
            "B12X_REPO": "",
            "B12X_REF": "",
            "B12X_FROM_PYPI": "0",
            "B12X_CACHEBUST": "test-refresh",
            "FAIL_UV": "0",
        }
        mock = self.bin_dir / "uv"
        mock.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            "with open(os.environ['COMMAND_LOG'], 'a') as log:\n"
            "    log.write(json.dumps({'args': sys.argv[1:],\n"
            "        'arch': os.environ.get('FLASHINFER_JIT_CACHE_PROVIDER_ARCH')}) + '\\n')\n"
            "if os.environ['FAIL_UV'] == '1':\n"
            "    sys.exit(17)\n"
            "if sys.argv[1] == 'build':\n"
            "    for name in ('build', 'flashinfer_jit_cache_provider/jit_cache'):\n"
            "        path = pathlib.Path(name)\n"
            "        assert not path.exists(), f'Stale provider output: {path}'\n"
            "        path.mkdir(parents=True)\n"
            "        (path / 'stale.so').touch()\n"
        )
        mock.chmod(0o755)

    def commands(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def run_providers(self):
        return subprocess.run(
            [
                "bash",
                str(PROJECT_DIR / "docker/build_flashinfer_jit_providers.sh"),
                "/prepared/python3",
                str(self.root / "wheel output"),
            ],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
        )

    def test_builds_each_provider_with_prepared_python_and_clean_output(self):
        (self.root / "flashinfer-jit-cache-provider").mkdir()
        self.env["FLASHINFER_JIT_CACHE_PROVIDER_ARCHS"] = "12.1a 12.0f\n9.0a"
        result = self.run_providers()
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertEqual([cmd["arch"] for cmd in commands], ["12.1a", "12.0f", "9.0a"])
        for command in commands:
            self.assertEqual(
                command["args"],
                [
                    "build", "--python", "/prepared/python3", "--no-build-isolation",
                    "--wheel", ".", f"--out-dir={self.root / 'wheel output'}", "-v",
                ],
            )

    def test_monolithic_ref_skips_provider_builds(self):
        self.env.pop("FLASHINFER_JIT_CACHE_PROVIDER_ARCHS")
        result = self.run_providers()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), [])

    def test_failed_provider_stops_build(self):
        (self.root / "flashinfer-jit-cache-provider").mkdir()
        self.env["FLASHINFER_JIT_CACHE_PROVIDER_ARCHS"] = "12.1a 12.0f"
        self.env["FAIL_UV"] = "1"
        result = self.run_providers()
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(len(self.commands()), 1)

    def test_provider_requires_architectures(self):
        (self.root / "flashinfer-jit-cache-provider").mkdir()
        self.env.pop("FLASHINFER_JIT_CACHE_PROVIDER_ARCHS")
        result = self.run_providers()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.commands(), [])

    def run_b12x_install(self):
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        run = next(
            block for block in re.split(r"(?m)^RUN ", dockerfile)
            if block.startswith("--mount=") and '\n    if [ -n "$B12X_REPO" ]' in block
        ).split("\n\n", 1)[0]
        # Execute the Dockerfile's actual shell block with package installs and
        # import verification mocked. Any unexpected git clone fails the test.
        run = re.sub(r"^--mount=\S+\s*\\\n", "", run)
        for name in ("python3", "git"):
            mock = self.bin_dir / name
            mock.write_text("#!/bin/sh\nexit " + ("0" if name == "python3" else "91") + "\n")
            mock.chmod(0o755)
        # Use a fixed interpreter so the uv mock does not use the python3 stub.
        uv_mock = self.bin_dir / "uv"
        uv_mock.write_text(uv_mock.read_text().replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        return subprocess.run(
            ["sh", "-c", run], cwd=self.root, env=self.env, text=True, capture_output=True
        )

    def test_pypi_refreshes_latest_without_changing_dependencies(self):
        self.env["B12X_FROM_PYPI"] = "1"
        result = self.run_b12x_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), [{
            "args": [
                "pip", "install", "--upgrade", "--refresh-package", "b12x",
                "--no-deps", "--index-url", "https://pypi.org/simple", "b12x",
            ],
            "arch": None,
        }])

    def test_pypi_failure_is_not_silently_skipped(self):
        self.env["B12X_FROM_PYPI"] = "1"
        self.env["FAIL_UV"] = "1"
        result = self.run_b12x_install()
        self.assertEqual(result.returncode, 17, result.stderr)

    def test_unselected_b12x_install_is_skipped(self):
        result = self.run_b12x_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands(), [])

    def write_torch_helper(self, path, exit_code=0):
        helper = self.root / path
        helper.parent.mkdir(parents=True, exist_ok=True)
        helper.write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "assert len(sys.argv) == 1\n"
            "with Path('torch-helper.log').open('a') as log:\n"
            "    log.write(sys.argv[0] + '\\n')\n"
            f"if {exit_code}:\n"
            f"    sys.exit({exit_code})\n"
            # The upstream helper reads relative to the checkout root.
            "path = Path('requirements/build/cuda.txt')\n"
            "path.write_text(path.read_text().replace('torch==0.0.0\\n', ''))\n"
        )

    def run_vllm_requirements(self):
        for path, content in {
            "requirements/build/cuda.txt": "torch==0.0.0\npackaging\n",
            "requirements/cuda.txt": "nvidia-cutlass-dsl[cu13]==4.6.0\nflashinfer-python\n",
            "requirements/test/cuda.txt": "triton\nfastsafetensors\npytest\n",
        }.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        # Installing before the helper strips Torch must fail the test.
        with (self.bin_dir / "uv").open("a") as mock:
            mock.write(
                "assert pathlib.Path('requirements/build/cuda.txt').read_text() "
                "== 'packaging\\n', 'Torch pin reached the package installer'\n"
            )
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        run = next(
            block for block in re.split(r"(?m)^RUN ", dockerfile)
            if block.startswith("--mount=") and "use_existing_torch.py" in block
        ).split("\n\n", 1)[0]
        run = re.sub(r"^--mount=\S+\s*\\\n", "", run)
        run = run.replace(
            "/tmp/vllm-patches/pin_cutlass_dsl.py",
            shlex.quote(str(PROJECT_DIR / "docker/pin_cutlass_dsl.py")),
        )
        return subprocess.run(
            ["sh", "-c", run], cwd=self.root,
            env={**self.env, "CUTLASS_DSL_VERSION": "4.7.0"},
            text=True, capture_output=True,
        )

    def assert_vllm_requirements_prepared(self, helper):
        result = self.run_vllm_requirements()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "torch-helper.log").read_text(), helper + "\n")
        self.assertEqual(self.commands(), [{
            "args": ["pip", "install", "-r", "requirements/build/cuda.txt", "setuptools-rust>=1.9.0"],
            "arch": None,
        }])

    def test_vllm_uses_relocated_torch_helper(self):
        self.write_torch_helper("tools/use_existing_torch.py")
        self.assert_vllm_requirements_prepared("tools/use_existing_torch.py")

    def test_vllm_supports_root_level_torch_helper(self):
        self.write_torch_helper("use_existing_torch.py")
        self.assert_vllm_requirements_prepared("use_existing_torch.py")

    def test_vllm_prefers_tools_helper_when_both_exist(self):
        self.write_torch_helper("tools/use_existing_torch.py")
        self.write_torch_helper("use_existing_torch.py", exit_code=19)
        self.assert_vllm_requirements_prepared("tools/use_existing_torch.py")

    def test_vllm_missing_torch_helper_stops_before_install(self):
        result = self.run_vllm_requirements()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing tools/use_existing_torch.py and use_existing_torch.py", result.stderr)
        self.assertEqual(self.commands(), [])

    def test_vllm_failed_torch_helper_stops_without_fallback_or_install(self):
        self.write_torch_helper("tools/use_existing_torch.py", exit_code=19)
        self.write_torch_helper("use_existing_torch.py")
        result = self.run_vllm_requirements()
        self.assertEqual(result.returncode, 19, result.stderr)
        self.assertEqual((self.root / "torch-helper.log").read_text(), "tools/use_existing_torch.py\n")
        self.assertEqual(self.commands(), [])


if __name__ == "__main__":
    unittest.main()
