#!/usr/bin/env python3
"""Exercise dependency build commands without Docker, downloads, or GPUs."""

import json
import os
from pathlib import Path
import re
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


if __name__ == "__main__":
    unittest.main()
