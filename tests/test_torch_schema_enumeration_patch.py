#!/usr/bin/env python3
"""CPU-only checks for Torch schema enumeration; no Torch installation needed."""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCHER_PATH = PROJECT_DIR / "docker/patch_torch_schema_enumeration.py"
SPEC = importlib.util.spec_from_file_location("torch_schema_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# Relevant PyTorch torch/_library/utils.py source, including a neighboring loop
# that must remain untouched by the GLM-5.3 patch.
FILL_DEFAULTS = '''def fill_defaults(schema, args, kwargs):
    new_args = []
    new_kwargs = {}
    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
            if info.name in kwargs:
                new_kwargs[info.name] = kwargs[info.name]
            else:
                new_kwargs[info.name] = info.default_value
        else:
            if i < len(args):
                new_args.append(args[i])
            else:
                new_args.append(info.default_value)
    return tuple(new_args), new_kwargs
'''
NEIGHBOR = '''

def zip_schema(schema, args, kwargs):
    for i in range(len(schema.arguments)):
        info = schema.arguments[i]
        if info.kwarg_only:
            if info.name in kwargs:
                yield info, kwargs[info.name]
            continue
        if i < len(args):
            yield info, args[i]
'''
SOURCE = FILL_DEFAULTS + NEIGHBOR


class Schema:
    def __init__(self, arguments):
        self._arguments = arguments
        self.reads = 0

    @property
    def arguments(self):
        self.reads += 1
        return list(self._arguments)


class TorchSchemaEnumerationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.target = self.root / "torch/_library/utils.py"
        self.target.parent.mkdir(parents=True)
        self.target.write_text(SOURCE)
        (self.root / "torch/__init__.py").write_text(
            'raise RuntimeError("Patching must not import Torch")\n'
        )

    def run_patch(self, *, installed=False):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.root)
        return subprocess.run(
            [sys.executable, str(PATCHER_PATH)]
            + (["--installed"] if installed else [str(self.target)]),
            cwd=PROJECT_DIR,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_patch_is_scoped_and_idempotent(self):
        patched = PATCHER.patch_fill_defaults(SOURCE)
        self.assertNotEqual(patched, SOURCE)
        self.assertTrue(patched.endswith(NEIGHBOR))
        self.assertEqual(PATCHER.patch_fill_defaults(patched), patched)

    def test_defaults_and_argument_identity_are_preserved_with_one_schema_read(self):
        original, patched = {}, {}
        exec(SOURCE, original)
        exec(PATCHER.patch_fill_defaults(SOURCE), patched)
        supplied, default = object(), []
        arguments = [
            SimpleNamespace(name="x", kwarg_only=False, default_value=None),
            SimpleNamespace(name="y", kwarg_only=False, default_value=default),
            SimpleNamespace(name="scale", kwarg_only=True, default_value=1),
            SimpleNamespace(name="out", kwarg_only=True, default_value=None),
        ]
        cases = [
            ([], (), {}, ((), {})),
            (arguments, (supplied,), {}, ((supplied, default), {"scale": 1, "out": None})),
            (arguments, (supplied, 7), {"scale": 2, "out": supplied},
             ((supplied, 7), {"scale": 2, "out": supplied})),
            (arguments, (), {"scale": 0}, ((None, default), {"scale": 0, "out": None})),
        ]
        for declarations, args, kwargs, expected in cases:
            with self.subTest(args=args, kwargs=kwargs):
                before, after = Schema(declarations), Schema(declarations)
                saved_kwargs = kwargs.copy()
                result = patched["fill_defaults"](after, args, kwargs)
                self.assertEqual(original["fill_defaults"](before, args, kwargs), result)
                self.assertEqual(result, expected)
                for actual_arg, expected_arg in zip(result[0], expected[0]):
                    self.assertIs(actual_arg, expected_arg)
                self.assertEqual(kwargs, saved_kwargs)
                self.assertEqual(before.reads, len(declarations) + 1)
                self.assertEqual(after.reads, 1)

    def test_installed_package_is_patched_without_importing_torch(self):
        for installed in (False, True):
            with self.subTest(installed=installed):
                self.target.write_text(SOURCE)
                result = self.run_patch(installed=installed)
                self.assertEqual(result.returncode, 0, result.stderr)
                patched = self.target.read_text()
                self.assertEqual(patched, PATCHER.patch_fill_defaults(SOURCE))
                result = self.run_patch(installed=installed)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("already patched", result.stdout)
                self.assertEqual(self.target.read_text(), patched)

    def test_unknown_or_ambiguous_sources_fail_without_writing(self):
        for source in (
            NEIGHBOR,
            FILL_DEFAULTS + SOURCE,
            SOURCE.replace("range(len(schema.arguments))", "range(schema.size)"),
            SOURCE.replace("info = schema.arguments[i]", "info = other.arguments[i]"),
            SOURCE.replace("for i in range(len(schema.arguments)):",
                           "for i, info in enumerate(schema.arguments):"),
            SOURCE.replace("def fill_defaults(", "def fill_defaults(" + "("),
        ):
            with self.subTest(source=source):
                self.target.write_text(source)
                result = self.run_patch()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unable to patch", result.stderr)
                self.assertEqual(self.target.read_text(), source)

    def test_missing_target_fails_clearly(self):
        self.target.unlink()
        result = self.run_patch(installed=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unable to patch", result.stderr)

    def test_runner_applies_patch_after_all_package_installations(self):
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        runner = dockerfile.split("FROM ${CUDA_IMAGE} AS runner\n", 1)[1]
        script = "patch_torch_schema_enumeration.py"
        self.assertIn(f"COPY docker/{script} /tmp/torch-patches/{script}", runner)
        self.assertGreater(
            runner.index(f"RUN python3 /tmp/torch-patches/{script} --installed"),
            runner.rindex("uv pip install"),
        )


if __name__ == "__main__":
    unittest.main()
