#!/usr/bin/env python3
"""Check build-time installation and run the upstream CPU cache regressions."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "docker/patch_b12x_cache_integrity.py"
PATCH = ROOT / "docker/b12x-cache-integrity.patch"
FIXTURE = ROOT / "tests/fixtures/b12x-cache-integrity"


class B12xCachePatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(FIXTURE, self.root / "b12x/_lib")
        (self.root / "b12x/__init__.py").write_text(
            'raise RuntimeError("Build-time patching must not import B12X")\n'
        )

    def run_patcher(self, installed=False):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root)
        return subprocess.run(
            [sys.executable, str(PATCHER)] +
            (["--installed"] if installed else [str(self.root)]),
            capture_output=True, text=True, env=env,
        )

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in self.root.rglob("*") if p.is_file()}

    def test_patch_installed_and_source_packages_idempotently(self):
        result = self.run_patcher(installed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Patched B12X", result.stdout)
        before = self.snapshot()
        result = self.run_patcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already patched", result.stdout)
        self.assertEqual(before, self.snapshot())
        self.assertFalse((self.root / "tests").exists())
        for path in (self.root / "b12x/_lib").glob("*.py"):
            compile(path.read_text(), str(path), "exec")

    def test_unknown_sources_fail_before_any_file_changes(self):
        target = self.root / "b12x/_lib/compile_plan.py"
        target.write_text(target.read_text().replace(
            "return _cache_object_path(program.key).is_file()", "return True",
        ))
        before = self.snapshot()
        result = self.run_patcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source differs", result.stderr)
        self.assertEqual(before, self.snapshot())

    def test_pypi_layout_without_preparation_planner(self):
        (self.root / "b12x/_lib/compile_plan.py").unlink()
        target = self.root / "b12x/_lib/compiler.py"
        target.write_text(target.read_text().replace(
            "from .compile_plan import _RETAINED_PROGRAMS\n", "",
        ))
        result = self.run_patcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("valid_object(staged_object,", target.read_text())
        self.assertIn("atomic_write_bytes(object_path, object_bytes)", target.read_text())
        self.assertFalse((self.root / "b12x/_lib/compile_plan.py").exists())
        self.assertEqual(self.run_patcher().returncode, 0)

    def test_upstream_patch_includes_passing_behavior_tests(self):
        result = subprocess.run(
            ["git", "apply", str(PATCH)], cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, str(self.root / "tests/_lib/test_compile_cache_integrity.py")],
            cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Ran 11 tests", result.stderr)

    def test_runner_patches_after_both_b12x_install_paths(self):
        runner = (ROOT / "Dockerfile").read_text().split("FROM ${CUDA_IMAGE} AS runner\n", 1)[1]
        position = runner.index("RUN python3 /tmp/b12x-patches/patch_b12x_cache_integrity.py --installed")
        for install in ("uv pip install --reinstall --no-deps /tmp/b12x-source",
                        "uv pip install --upgrade --refresh-package b12x"):
            self.assertLess(runner.index(install), position)
        self.assertIn("COPY docker/patch_b12x_cache_integrity.py docker/b12x-cache-integrity.patch /tmp/b12x-patches/", runner)


if __name__ == "__main__":
    unittest.main()
