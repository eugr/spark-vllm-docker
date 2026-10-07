#!/usr/bin/env python3
"""Check the loader workaround with a CPU-only C compilation reproducer."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCH_NAME = "patch_flashinfer_b12x_loader_include_order.py"


class LoaderIncludeOrderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.loader = self.root / "flashinfer/experimental/b12x/loader"

    def write_fixture(self, includes):
        self.loader.mkdir(parents=True)
        # Preserve the real dependency chain without requiring Python/CUDA headers.
        (self.loader / "_direct.c").write_text(
            "#define IO_ALIGNMENT 4096\n"
            "static int validate_direct_range(int offset) { return offset >= 0; }\n"
        )
        (self.loader / "_bounce.c").write_text(
            "static int bounce_execute(void) {\n"
            "    return validate_direct_range(IO_ALIGNMENT);\n"
            "}\n"
        )
        (self.loader / "_batch.c").write_text('#include "_bounce.c"\n')
        target = self.loader / "_storage.c"
        target.write_text(includes + "int main(void) { return !bounce_execute(); }\n")
        return target

    def run_patch(self):
        # Exercise the actual Dockerfile command after PR application and before
        # wheel packaging, using the host Python in place of the build Python.
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        builder = dockerfile.split("FROM base AS flashinfer-builder\n", 1)[1]
        builder = builder.split("FROM scratch AS flashinfer-export\n", 1)[0]
        command = next(
            line.removeprefix("RUN ") for line in builder.splitlines()
            if line.startswith("RUN ") and PATCH_NAME in line
        )
        self.assertLess(builder.index('if [ -n "$FLASHINFER_PRS" ]'), builder.index(command))
        self.assertLess(builder.index(command), builder.index("# flashinfer-python"))
        command = command.replace(
            f"/tmp/{PATCH_NAME}", shlex.quote(str(PROJECT_DIR / "docker" / PATCH_NAME))
        )
        return subprocess.run(
            ["sh", "-c", command], cwd=self.root, capture_output=True, text=True,
            env={**os.environ, "FLASHINFER_BUILD_PYTHON": sys.executable},
        )

    @unittest.skipUnless(shutil.which("cc"), "C compiler required for the reproducer")
    def test_broken_loader_fails_compilation_then_compiles_after_patch(self):
        target = self.write_fixture('#include "_batch.c"\n#include "_direct.c"\n')
        command = ["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(target)]
        before = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(before.returncode, 0)
        self.assertIn("IO_ALIGNMENT", before.stderr)
        self.assertIn("validate_direct_range", before.stderr)

        result = self.run_patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        after = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(after.returncode, 0, after.stderr)

    def test_upstream_fixed_source_is_untouched(self):
        target = self.write_fixture('#include "_direct.c"\n#include "_batch.c"\n')
        before = target.read_bytes(), target.stat().st_mtime_ns
        result = self.run_patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skipping", result.stdout)
        self.assertEqual((target.read_bytes(), target.stat().st_mtime_ns), before)

    def test_ref_without_b12x_loader_is_untouched(self):
        result = self.run_patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not applicable", result.stdout)
        self.assertFalse(self.loader.exists())


if __name__ == "__main__":
    unittest.main()
