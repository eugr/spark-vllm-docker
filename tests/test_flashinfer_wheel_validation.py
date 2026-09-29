#!/usr/bin/env python3
"""Exercise both build-script validation gates with synthetic wheel metadata."""

from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile


PROJECT_DIR = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = (PROJECT_DIR / "build-and-copy.sh").read_text()
# Load only the real validators, never the operational body of the build script.
VALIDATORS = BUILD_SCRIPT[
    BUILD_SCRIPT.index("validate_flashinfer_wheel_set() {"):
    BUILD_SCRIPT.index("promote_wheel_set() {")
]


class FlashInferWheelValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fi = self.root / "flashinfer wheels"
        self.vllm = self.root / "vllm wheels"
        self.fi.mkdir()
        self.vllm.mkdir()
        self.architectures = "12.1a"
        self.write_wheel("flashinfer-cubin")
        self.write_wheel("flashinfer-python")
        self.jit = self.write_wheel("flashinfer-jit-cache")
        (self.fi / ".flashinfer-commit").write_text("test-commit\n")
        (self.fi / ".flashinfer-arch").write_text("12.1a\n")
        (self.vllm / "vllm-test.whl").touch()

    def write_wheel(self, name, version="0.7.0", requirements=(), path=None):
        stem = f"{name.replace('-', '_')}-{version}"
        path = path or self.fi / f"{stem}-py3-none-any.whl"
        metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        metadata += "".join(f"Requires-Dist: {req}\n" for req in requirements)
        with ZipFile(path, "w") as archive:
            archive.writestr(f"{stem}.dist-info/METADATA", metadata + "\n")
        return path

    def shim(self, *requirements):
        self.write_wheel("flashinfer-jit-cache", requirements=requirements, path=self.jit)

    def check(self, error=None, gates=("export", "runner")):
        commands = {
            "export": ["validate_exported_wheel_set", "flashinfer", str(self.fi)],
            "runner": ["validate_runner_wheel_inputs", str(self.fi), str(self.vllm)],
        }
        for gate in gates:
            with self.subTest(gate=gate):
                script = (
                    "set -e\n" + VALIDATORS + "\nGPU_ARCH_LIST="
                    + shlex.quote(self.architectures) + "\n" + shlex.join(commands[gate])
                )
                result = subprocess.run(
                    ["bash", "-c", script], cwd=PROJECT_DIR, capture_output=True, text=True
                )
                output = result.stdout + result.stderr
                if error is None:
                    self.assertEqual(result.returncode, 0, output)
                else:
                    self.assertNotEqual(result.returncode, 0, output)
                    self.assertIn(error, output)
                    self.assertIn("--rebuild-flashinfer", output)

    def test_monolithic_uses_metadata_not_version_cutoff(self):
        for version in ("0.6.0", "0.7.0"):
            with self.subTest(version=version):
                self.write_wheel("flashinfer-jit-cache", version, path=self.jit)
                self.check()

    def test_historical_monolithic_cache_needs_no_arch_marker_for_runner(self):
        (self.fi / ".flashinfer-arch").unlink()
        self.check(gates=("runner",))

    def test_non_provider_dependency_does_not_require_providers(self):
        self.shim("filelock>=3.0")
        self.check()

    def test_missing_provider_is_rejected(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0")
        self.check("provider wheel for arch sm121a (flashinfer_jit_cache_sm121a-*.whl) is missing")

    def test_wrong_arch_or_missing_feature_suffix_does_not_satisfy_requirement(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0")
        self.write_wheel("flashinfer-jit-cache-sm120a")
        self.write_wheel("flashinfer-jit-cache-sm121")
        self.check("provider wheel for arch sm121a")

    def test_complete_provider_set(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0")
        self.write_wheel("flashinfer-jit-cache-sm121a")
        self.check()

    def test_wrong_version_is_rejected(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0")
        self.write_wheel("flashinfer-jit-cache-sm121a", "0.6.0")
        self.check("does not satisfy flashinfer-jit-cache-sm121a==0.7.0")

    def test_provider_metadata_must_match_name_and_version(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0")
        wheel = self.fi / "flashinfer_jit_cache_sm121a-0.7.0-py3-none-any.whl"
        for name, version in (("flashinfer-jit-cache-sm120a", "0.7.0"),
                              ("flashinfer-jit-cache-sm121a", "0.6.0")):
            with self.subTest(name=name, version=version):
                self.write_wheel(name, version, path=wheel)
                self.check("does not satisfy flashinfer-jit-cache-sm121a==0.7.0")

    def test_duplicate_provider_versions_are_rejected(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0")
        self.write_wheel("flashinfer-jit-cache-sm121a")
        self.write_wheel("flashinfer-jit-cache-sm121a", "0.6.0")
        self.check("Expected exactly one FlashInfer provider wheel")

    def test_every_declared_provider_is_required(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0", "flashinfer-jit-cache-sm120f==0.7.0")
        self.write_wheel("flashinfer-jit-cache-sm121a")
        # Even a single selected target cannot ignore another shim dependency.
        self.check("provider wheel for arch sm120f")
        self.write_wheel("flashinfer-jit-cache-sm120f")
        self.architectures = "12.1a 12.0f"
        self.check()

    def test_shim_for_another_architecture_is_rejected(self):
        self.shim("flashinfer-jit-cache-sm120f==0.7.0")
        self.write_wheel("flashinfer-jit-cache-sm120f")
        self.check("does not declare the provider for GPU arch 12.1a")
        self.architectures = "12.0f"
        self.check()

    def test_normalized_names_and_parenthesized_pins(self):
        self.shim("FlashInfer_JIT_Cache_sm121a (==0.7.0)")
        self.write_wheel("flashinfer-jit-cache-sm121a")
        self.check()

    def test_public_version_pin_accepts_local_build(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0")
        self.write_wheel("flashinfer-jit-cache-sm121a", "0.7.0+cu134")
        self.check()

    def test_local_version_pin_requires_matching_build(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0.dev20260924+cu134")
        wheel = self.write_wheel("flashinfer-jit-cache-sm121a", "0.7.0.dev20260924+cu134")
        self.check()
        self.write_wheel("flashinfer-jit-cache-sm121a", "0.7.0.dev20260924+cu130", path=wheel)
        self.check("does not satisfy flashinfer-jit-cache-sm121a==0.7.0.dev20260924+cu134")

    def test_corrupt_wheel_is_not_treated_as_monolithic(self):
        self.jit.write_bytes(b"not a wheel")
        self.check("Cannot read wheel metadata")

    def test_missing_metadata_is_not_treated_as_monolithic(self):
        with ZipFile(self.jit, "w") as archive:
            archive.writestr("unrelated.txt", "")
        self.check("expected exactly one .dist-info/METADATA entry")

    def test_unrecognized_provider_requirement_is_not_skipped(self):
        self.shim("flashinfer-jit-cache-sm121a>=0.7.0")
        self.check("Unsupported FlashInfer provider requirement")

    def test_publication_listing_defaults_to_sm121a_and_excludes_unrelated_wheels(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0")
        provider = self.write_wheel("flashinfer-jit-cache-sm121a")
        self.write_wheel("flashinfer-jit-cache-sm120a")
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "docker/validate_flashinfer_wheels.py"),
             str(self.jit), "--list-provider-wheels"], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [str(provider)])

    def test_failed_publication_listing_emits_no_partial_list(self):
        self.shim("flashinfer-jit-cache-sm121a==0.7.0", "flashinfer-jit-cache-sm120f==0.7.0")
        self.write_wheel("flashinfer-jit-cache-sm121a")
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "docker/validate_flashinfer_wheels.py"),
             str(self.jit), "--list-provider-wheels"], capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
