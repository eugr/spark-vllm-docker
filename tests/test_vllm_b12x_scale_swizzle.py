#!/usr/bin/env python3
"""CPU-only regression coverage for the B12X intrinsics package split."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


PROJECT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "docker" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PATCHER = load_script("patch_vllm_b12x_scale_swizzle")
OLD = PATCHER.OLD_MODULE
REFERENCE = PATCHER.REFERENCE_MODULE

# Accessor and registry shape from local-inference-lab/vllm@8244d1c44348.
# Kernel imports are represented by the importer supplied by each test.
SOURCE = '''from types import ModuleType

_B12X_SUBMODULES = {
    module_name: _import_submodule(module_name)
    for module_name in (
        "b12x.gemm.blockscaled",
        "b12x._lib.intrinsics",
    )
}


def _get_submodule(module_name: str) -> ModuleType | None:
    return _B12X_SUBMODULES.get(module_name)


def get_b12x_intrinsics() -> ModuleType | None:
    return _get_submodule("b12x._lib.intrinsics")
'''


class ScaleSwizzleTests(unittest.TestCase):
    def setUp(self):
        self.export = SimpleNamespace(swizzle_block_scale=lambda tensor: ("swizzled", tensor))
        self.empty = SimpleNamespace()

    def accessor(self, source, modules):
        namespace = {"_import_submodule": modules.get}
        exec(compile(source, "fixture_b12x.py", "exec"), namespace)
        return namespace["get_b12x_intrinsics"]

    def test_reproduces_failure_and_fixes_shared_accessor(self):
        modules = {OLD: self.empty, REFERENCE: self.export}
        with self.assertRaises(AttributeError):
            self.accessor(SOURCE, modules)().swizzle_block_scale("scale")
        accessor = self.accessor(PATCHER.patch_source(SOURCE), modules)
        # Both NVFP4 and MXFP4 weight processing use this accessor.
        for scale in ("nvfp4-scale", "mxfp4-scale"):
            self.assertEqual(accessor().swizzle_block_scale(scale), ("swizzled", scale))

    def test_original_flashinfer_module_layout_still_works(self):
        accessor = self.accessor(PATCHER.patch_source(SOURCE), {OLD: self.export})
        self.assertIs(accessor(), self.export)

    def test_reference_export_is_preferred(self):
        accessor = self.accessor(PATCHER.patch_source(SOURCE), {
            OLD: SimpleNamespace(swizzle_block_scale=lambda x: "old"),
            REFERENCE: self.export,
        })
        self.assertIs(accessor(), self.export)

    def test_missing_or_noncallable_exports_are_unavailable(self):
        for modules in ({}, {OLD: self.empty, REFERENCE: self.empty},
                        {REFERENCE: SimpleNamespace(swizzle_block_scale=42)}):
            with self.subTest(modules=modules):
                self.assertIsNone(self.accessor(PATCHER.patch_source(SOURCE), modules)())

    def test_patch_is_idempotent_and_preserves_unrelated_code(self):
        extra = "\ndef unrelated():\n    return 123\n"
        patched = PATCHER.patch_source(SOURCE + extra)
        self.assertTrue(patched.endswith(extra))
        self.assertEqual(PATCHER.patch_source(patched), patched)

    def test_already_fixed_direct_accessor_is_preserved(self):
        fixed = SOURCE.replace(OLD, REFERENCE)
        self.assertEqual(PATCHER.patch_source(fixed), fixed)

    def test_absent_accessor_is_unchanged(self):
        self.assertEqual(PATCHER.patch_source("x = 1\n"), "x = 1\n")

    def test_partial_patch_and_unknown_layout_fail_without_writing(self):
        broken_sources = [
            SOURCE.replace('return _get_submodule(', 'return different_loader('),
            SOURCE.replace('        "b12x._lib.intrinsics",\n', ''),
            PATCHER.patch_source(SOURCE).replace(f'        "{REFERENCE}",\n', ''),
        ]
        for source in broken_sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                PATCHER.patch_source(source)

    def test_cli_skips_regular_source_and_patches_before_wheel_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, str(PROJECT / "docker/patch_vllm_b12x_scale_swizzle.py"), tmp]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            target = root / PATCHER.TARGET_REL
            target.parent.mkdir(parents=True)
            target.write_text(SOURCE)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_text(), PATCHER.patch_source(SOURCE))
        dockerfile = (PROJECT / "Dockerfile").read_text()
        source_patch = "RUN python3 /tmp/vllm-patches/patch_vllm_b12x_scale_swizzle.py ."
        self.assertLess(dockerfile.index(source_patch), dockerfile.index("# Final Compilation"))
        self.assertNotIn(source_patch, dockerfile.split("AS runner", 1)[1])


if __name__ == "__main__":
    unittest.main()
