import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PATCH = Path(__file__).resolve().parents[1] / "docker/patch_vllm_preserve_sm12x_target.py"
SIGNATURE = "function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)"


class TargetUnionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "CMakeLists.txt").write_text('set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")\n')
        self.helper = self.root / "cmake/utils.cmake"
        self.helper.parent.mkdir()
        # Minimal selector with the upstream failure: consume the first 12.x device only.
        self.original = SIGNATURE + """
  set(result)
  foreach(t IN LISTS TGT_CUDA_ARCHS)
    if(SRC_CUDA_ARCHS STREQUAL "8.0+PTX")
      list(APPEND result "8.0+PTX")
    elseif(t MATCHES "^12" AND SRC_CUDA_ARCHS STREQUAL "12.0f")
      list(APPEND result "${t}")
      break()
    elseif(t IN_LIST SRC_CUDA_ARCHS)
      list(APPEND result "${t}")
    endif()
  endforeach()
  set(${OUT_CUDA_ARCHS} "${result}" PARENT_SCOPE)
endfunction()
"""
        self.helper.write_text(self.original)

    def patch(self, arches):
        subprocess.run([sys.executable, str(PATCH), str(self.root)], check=True,
                       env={**os.environ, "VLLM_PRESERVE_SM12X_TARGET": "1", "TORCH_CUDA_ARCH_LIST": arches})

    def test_single_target_and_idempotent_multi_target(self):
        self.patch("12.1a")
        self.assertEqual(self.helper.read_text(), self.original)
        self.patch("10.3a 12.0a 12.1a")
        first = self.helper.read_text()
        self.assertNotEqual(first, self.original)
        self.patch("10.3a;12.0a;12.1a")
        self.assertEqual(first, self.helper.read_text())

    @unittest.skipUnless(shutil.which("cmake"), "requires CMake")
    def test_per_device_selection_preserves_exact_and_ptx_constraints(self):
        self.patch("10.3a 12.0a 12.1a")
        test = self.root / "verify.cmake"
        test.write_text("""
cmake_minimum_required(VERSION 3.20)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
function(expect src targets expected)
  cuda_archs_loose_intersection(actual "${src}" "${targets}")
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR "${src} on ${targets}: got '${actual}', expected '${expected}'")
  endif()
endfunction()
expect("12.0f" "10.3a;12.0a;12.1a" "12.0a;12.1a")
expect("12.0f" "12.1a;10.3a;12.0a" "12.1a;12.0a")
expect("10.3a" "10.3a;12.0a;12.1a" "10.3a")
expect("10.3a" "12.0a;12.1a" "")
expect("8.0+PTX" "10.3a;12.0a;12.1a" "8.0+PTX")
""")
        subprocess.run(["cmake", "-P", str(test)], check=True)


if __name__ == "__main__":
    unittest.main()
