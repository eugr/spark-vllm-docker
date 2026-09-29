#!/usr/bin/env python3

import ast
import ctypes
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCH_PATH = PROJECT_DIR / "docker/patch_vllm_startup_heap_trim.py"
SPEC = importlib.util.spec_from_file_location("startup_heap_trim_patch", PATCH_PATH)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

SOURCE = '''import gc


def freeze_gc_for_cudagraph_capture():
    gc.collect()
    gc.freeze()


def freeze_gc_heap() -> None:
    """Freeze the startup heap after all warmup has finished."""
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()


def unfreeze_heap():
    gc.unfreeze()
'''


class StartupHeapTrimPatchTests(unittest.TestCase):
    def runtime(self, source=SOURCE, *, trim_result=1, load_error=None, missing=False):
        events = []
        fake_gc = SimpleNamespace(
            collect=lambda *args: events.append(("collect", args)),
            freeze=lambda: events.append(("freeze",)),
        )
        trim = Mock(side_effect=lambda pad: events.append(("trim", pad)) or trim_result)
        library = SimpleNamespace() if missing else SimpleNamespace(malloc_trim=trim)
        fake_ctypes = SimpleNamespace(
            CDLL=Mock(return_value=library, side_effect=load_error),
            c_size_t=ctypes.c_size_t,
            c_int=ctypes.c_int,
        )
        tree = ast.parse(PATCHER.patch_source(source))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {"gc": fake_gc}
        exec(compile(tree, "patched_gc_utils.py", "exec"), namespace)
        with patch.dict(sys.modules, {"ctypes": fake_ctypes}):
            self.assertIsNone(namespace["freeze_gc_heap"]())
        return events, trim, fake_ctypes

    def test_trim_runs_after_full_collection_and_freeze_with_correct_abi(self):
        expected = [("collect", (0,)), ("collect", (1,)), ("collect", (2,)), ("freeze",)]
        for trim_result in (0, 1):
            with self.subTest(trim_result=trim_result):
                events, trim, fake_ctypes = self.runtime(trim_result=trim_result)
                self.assertEqual(events, expected + [("trim", 0)])
                fake_ctypes.CDLL.assert_called_once_with(None)
                self.assertEqual(trim.argtypes, [ctypes.c_size_t])
                self.assertIs(trim.restype, ctypes.c_int)

    def test_missing_libc_or_trim_symbol_does_not_break_startup(self):
        for kwargs in ({"load_error": OSError("unavailable")}, {"missing": True}):
            with self.subTest(kwargs=kwargs):
                events, trim, _ = self.runtime(**kwargs)
                self.assertEqual(events[-1], ("freeze",))
                trim.assert_not_called()

    def test_single_full_gc_call_is_supported(self):
        source = SOURCE.replace("gc.collect(0)\n    gc.collect(1)\n    gc.collect(2)", "gc.collect()")
        events, _, _ = self.runtime(source)
        self.assertEqual(events, [("collect", ()), ("freeze",), ("trim", 0)])

    def test_patch_preserves_capture_and_other_functions_and_is_idempotent(self):
        patched = PATCHER.patch_source(SOURCE)
        self.assertEqual(PATCHER.patch_source(patched), patched)
        def other_functions(source):
            return [ast.dump(node) for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name != "freeze_gc_heap"]
        self.assertEqual(other_functions(SOURCE), other_functions(patched))

    def test_function_at_end_without_newline(self):
        source = SOURCE[:SOURCE.index("\n\ndef unfreeze_heap")].rstrip()
        patched = PATCHER.patch_source(source)
        self.assertEqual(PATCHER.patch_source(patched), patched)

    def test_unknown_and_partial_layouts_are_rejected(self):
        for source in (
            SOURCE.replace("def freeze_gc_heap", "def new_freeze_gc_heap"),
            SOURCE.replace("    gc.collect(2)\n", ""),
            SOURCE.replace("gc.freeze()", "gc.freeze()\n    return"),
            PATCHER.patch_source(SOURCE).replace("malloc_trim(0)", "malloc_trim(1)"),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                PATCHER.patch_source(source)

    def test_source_entry_point_and_failure_without_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / PATCHER.TARGET_REL
            target.parent.mkdir(parents=True)
            (root / "vllm/__init__.py").write_text('raise RuntimeError("Do not import vLLM")\n')
            for args in ([str(root)], []):
                target.write_text(SOURCE)
                result = subprocess.run([sys.executable, str(PATCH_PATH), *args],
                                        cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(target.read_text(), PATCHER.patch_source(SOURCE))
            unknown = SOURCE.replace("gc.freeze()", "freeze_something_else()")
            target.write_text(unknown)
            result = subprocess.run([sys.executable, str(PATCH_PATH), str(root)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_text(), unknown)

    def test_dockerfile_patches_source_before_building_wheels(self):
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        build, runner = dockerfile.split("FROM ${CUDA_IMAGE} AS runner\n", 1)
        command = f"RUN python3 /tmp/vllm-patches/{PATCH_PATH.name} .\n"
        self.assertIn(command, build)
        self.assertLess(build.index(command), build.index("uv build --no-build-isolation --wheel ."))
        self.assertNotIn(PATCH_PATH.name, runner)


if __name__ == "__main__":
    unittest.main()
