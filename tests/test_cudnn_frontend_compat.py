#!/usr/bin/env python3
"""CPU regression tests for cuDNN Frontend 1.33 in both source builds."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "docker/patch_vllm_cudnn_frontend.py"
SPEC = importlib.util.spec_from_file_location("patch_cudnn", SCRIPT)
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)


class CudnnCompatibilityTests(unittest.TestCase):
    def test_reproduces_extra_output_failure_and_preserves_output_identity(self):
        for family in ("gdn", "gdn2", "gdp", "kda"):
            for count in (3, 4):
                with self.subTest(family=family, count=count):
                    outputs = [object() for _ in range(count)]
                    def build(**kwargs):
                        self.assertEqual(kwargs, {"q": "query", "output_final_state": True})
                        return outputs
                    env = dict(graph=SimpleNamespace(**{family: build}), family=family,
                               ports={"q": "query"}, attrs={"output_final_state": True})
                    if count == 4:
                        with self.assertRaises(ValueError):
                            exec(PATCH.OLD, env)
                    exec(PATCH.patch_flashinfer(PATCH.OLD), env)
                    self.assertIs(env["O"], outputs[0])
                    self.assertIs(env["fs"], outputs[1])

    def test_source_patches_are_idempotent_and_preserve_unrelated_requirements(self):
        source = "# preamble\n    " + PATCH.OLD + "\n# tail\n"
        fixed = PATCH.patch_flashinfer(source)
        self.assertEqual(fixed, source.replace(PATCH.OLD, PATCH.FIXED))
        self.assertEqual(PATCH.patch_flashinfer(fixed), fixed)
        requirements = "torch==2.14.0\nnvidia-cudnn-frontend>=1.31.0,<1.33\nnvtx==0.2.15\n"
        expected = requirements.replace(",<1.33", "")
        self.assertEqual(PATCH.patch_vllm(requirements), expected)
        self.assertEqual(PATCH.patch_vllm(expected), expected)
        # Future upstream policy remains authoritative.
        future = requirements.replace("<1.33", "<1.34")
        self.assertEqual(PATCH.patch_vllm(future), future)

    def test_source_cli_patches_before_wheel_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode, relative, original, expected in (
                ("flashinfer", PATCH.TARGET, PATCH.OLD, PATCH.FIXED),
                ("vllm", "requirements/cuda.txt", "nvidia-cudnn-frontend>=1.31.0,<1.33\n",
                 "nvidia-cudnn-frontend>=1.31.0\n"),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(original)
                for _ in range(2):
                    subprocess.run([sys.executable, str(SCRIPT), mode, str(root)],
                                   check=True, capture_output=True)
                    self.assertEqual(path.read_text(), expected)

    def test_dockerfile_patches_sources_before_building_wheels(self):
        dockerfile = (PROJECT / "Dockerfile").read_text()
        fi = dockerfile.split("FROM base AS flashinfer-builder", 1)[1].split(
            "FROM scratch AS flashinfer-export", 1)[0]
        self.assertLess(fi.index("cudnn_frontend.py flashinfer ."), fi.index("uv build"))
        vllm = dockerfile.split("FROM base AS vllm-builder", 1)[1].split(
            "FROM scratch AS vllm-export", 1)[0]
        self.assertLess(vllm.index("cudnn_frontend.py vllm ."), vllm.index("uv build"))


if __name__ == "__main__":
    unittest.main()
