#!/usr/bin/env python3
"""Exercise dependency build commands without Docker, downloads, or GPUs."""

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
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
            "FAIL_UV": "0",
        }
        mock = self.bin_dir / "uv"
        mock.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, sys\n"
            "with open(os.environ['COMMAND_LOG'], 'a') as log:\n"
            "    log.write(json.dumps({'args': sys.argv[1:],\n"
            "        'arch': os.environ.get('FLASHINFER_JIT_CACHE_PROVIDER_ARCH')}) + '\\n')\n"
            "if os.environ['FAIL_UV'] == '1' or pathlib.Path.cwd().name == os.environ.get('FAIL_UV_DIR'):\n"
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

    def docker_run_block(self, marker):
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        run = next(
            block for block in re.split(r"(?m)^RUN ", dockerfile)
            if block.startswith("--mount=") and marker in block
        ).split("\n\n", 1)[0]
        return re.sub(r"(?m)^\s*--mount=\S+\s*\\\n", "", run)

    def run_flashinfer_wheels(self, cubin):
        (self.root / "flashinfer-jit-cache").mkdir()
        if cubin:
            (self.root / "flashinfer-cubin").mkdir()
        (self.root / "pyproject.toml").write_text('license = "Apache-2.0"\n')
        wheel_dir = self.root / "wheels"
        wheel_dir.mkdir()
        for name, body in (("prepared-python", "exit 0"), ("git", "echo test-commit")):
            mock = self.bin_dir / name
            mock.write_text(f"#!/bin/sh\n{body}\n")
            mock.chmod(0o755)
        run = self.docker_run_block("# flashinfer-python")
        run = run.replace("/workspace/wheels", str(wheel_dir)).replace(
            "/tmp/build_flashinfer_jit_providers.sh",
            str(PROJECT_DIR / "docker/build_flashinfer_jit_providers.sh"),
        )
        return subprocess.run(
            ["sh", "-c", run], cwd=self.root,
            env={**self.env, "FLASHINFER_BUILD_CUBIN": str(int(cubin)),
                 "FLASHINFER_BUILD_PYTHON": str(self.bin_dir / "prepared-python"),
                 "FLASHINFER_CUDA_ARCH_LIST": "12.1a"},
            text=True, capture_output=True,
        )

    def test_regular_flashinfer_builds_cubin_and_jit_wheels(self):
        result = self.run_flashinfer_wheels(cubin=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.commands()), 3)
        self.assertTrue((self.root / "flashinfer-cubin/build").is_dir())
        self.assertTrue((self.root / "flashinfer-jit-cache/build").is_dir())

    def test_cubin_failure_stops_before_jit_build_and_provenance(self):
        self.env["FAIL_UV_DIR"] = "flashinfer-cubin"
        result = self.run_flashinfer_wheels(cubin=True)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(len(self.commands()), 2)
        self.assertFalse((self.root / "wheels/.flashinfer-commit").exists())

    def test_b12x_build_skips_cubins_but_keeps_jit_and_provenance(self):
        result = self.run_flashinfer_wheels(cubin=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.commands()), 2)
        self.assertTrue((self.root / "flashinfer-jit-cache/build").is_dir())
        self.assertEqual((self.root / "wheels/.flashinfer-commit").read_text(), "test-commit\n")
        self.assertEqual((self.root / "wheels/.flashinfer-arch").read_text(), "12.1a\n")

    def test_flashinfer_git_caches_are_isolated_and_refresh_selected_refs(self):
        def git(*args, cwd):
            result = subprocess.run(["git", *args], cwd=cwd, env=self.env,
                                    text=True, capture_output=True, check=True)
            return result.stdout.strip()

        sources = []
        for name in ("upstream", "fork"):
            source = self.root / name
            source.mkdir()
            git("init", "-b", "main", cwd=source)
            git("config", "user.name", "Test", cwd=source)
            git("config", "user.email", "test@example.invalid", cwd=source)
            git("config", "commit.gpgsign", "false", cwd=source)
            (source / "identity").write_text(name)
            git("add", ".", cwd=source)
            git("commit", "-m", "initial", cwd=source)
            sources.append(source)

        target = self.root / "checkout"
        run = self.docker_run_block('echo "CACHEBUST_FLASHINFER=')
        run = run.replace("/repo-cache", str(self.root / "repo-cache"))
        run = run.replace("/workspace/flashinfer", str(target))
        for source in (*sources, sources[0], sources[1]):
            if target.exists():
                shutil.rmtree(target)
            (source / "identity").write_text(source.name + " updated")
            git("add", ".", cwd=source)
            git("commit", "--allow-empty", "-m", "update", cwd=source)
            result = subprocess.run(
                ["sh", "-c", run], cwd=self.root,
                env={**self.env, "FLASHINFER_REPO": str(source),
                     "FLASHINFER_REF": "main", "CACHEBUST_FLASHINFER": "test"},
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(git("rev-parse", "HEAD", cwd=target), git("rev-parse", "HEAD", cwd=source))
            self.assertEqual(git("remote", "get-url", "origin", cwd=target), str(source))
        self.assertEqual(len(list((self.root / "repo-cache").iterdir())), 2)

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
