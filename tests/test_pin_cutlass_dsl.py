#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PINNER_PATH = PROJECT_DIR / "docker/pin_cutlass_dsl.py"
SPEC = importlib.util.spec_from_file_location("cutlass_dsl_pinner", PINNER_PATH)
assert SPEC is not None and SPEC.loader is not None
PINNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PINNER)


class CutlassDslPinTests(unittest.TestCase):
    def test_vllm_cuda_requirement_is_updated(self):
        source = "nvidia-cutlass-dsl[cu13]==4.6.2\nquack-kernels==0.6.4\n"
        updated, count = PINNER.pin_text(source, "4.7.0")

        self.assertEqual(count, 1)
        self.assertIn("nvidia-cutlass-dsl[cu13]==4.7.0", updated)
        self.assertIn("quack-kernels==0.6.4", updated)

    def test_b12x_package_and_companion_libraries_are_updated(self):
        source = '''dependencies = [
  "nvidia-cutlass-dsl==4.6.0",
  "nvidia-cutlass-dsl-libs-base==4.6.0",
  "nvidia-cutlass-dsl-libs-core==4.6.0",
  "nvidia-cutlass-dsl-libs-cu12==4.6.0",
  "nvidia-cutlass-dsl-libs-cu13==4.6.0",
]
'''
        updated, count = PINNER.pin_text(source, "4.7.0")

        self.assertEqual(count, 5)
        self.assertNotIn("==4.6.0", updated)
        self.assertEqual(updated.count("==4.7.0"), 5)

    def test_non_exact_requirement_is_not_silently_rewritten(self):
        updated, count = PINNER.pin_text(
            "nvidia-cutlass-dsl[cu13]>=4.6.0\n", "4.7.0"
        )

        self.assertEqual(count, 0)
        self.assertEqual(updated, "nvidia-cutlass-dsl[cu13]>=4.6.0\n")


if __name__ == "__main__":
    unittest.main()
