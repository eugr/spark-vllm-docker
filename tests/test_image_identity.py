#!/usr/bin/env python3
"""Content comparison across Docker stores, without a Docker daemon."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "docker/image_identity.py"
FIXTURES = ROOT / "tests/fixtures/image-identity"
SPEC = importlib.util.spec_from_file_location("image_identity", HELPER)
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


class ImageIdentityTests(unittest.TestCase):
    def setUp(self):
        self.classic = json.loads((FIXTURES / "classic.json").read_text())
        self.containerd = json.loads((FIXTURES / "containerd.json").read_text())

    def test_save_load_across_stores_matches_without_registry_digest(self):
        self.assertEqual(IDENTITY.fingerprint(self.classic),
                         IDENTITY.fingerprint(self.containerd))

    def test_layers_order_and_platform_must_match(self):
        for field, value in (
            ("Architecture", "amd64"), ("Os", "windows"),
            ("Variant", "v8"), ("OsVersion", "different"),
            ("RootFS", {"Type": "layers", "Layers": ["sha256:" + "3" * 64]}),
            ("RootFS", {"Type": "layers", "Layers": list(reversed(self.classic[0]["RootFS"]["Layers"]))}),
        ):
            with self.subTest(field=field, value=value):
                other = copy.deepcopy(self.classic)
                other[0][field] = value
                self.assertNotEqual(IDENTITY.fingerprint(self.classic),
                                    IDENTITY.fingerprint(other))

    def test_runtime_changes_mismatch_even_with_same_build_label_and_digest(self):
        self.classic[0]["Config"]["Labels"]["com.spark-vllm.build-id"] = "same-build"
        for field, value in (
            ("Env", ["VLLM_WORKER_MULTIPROC_METHOD=fork"]),
            ("Cmd", ["python3"]), ("Entrypoint", ["/entrypoint.sh"]),
            ("User", "1000"), ("WorkingDir", "/app"), ("StopSignal", "SIGINT"),
            ("Healthcheck", {"Test": ["CMD", "true"]}),
            ("ExposedPorts", {"9000/tcp": {}}), ("Volumes", {"/other": {}}),
            ("Labels", {"example.version": "2", "com.spark-vllm.build-id": "same-build"}),
        ):
            with self.subTest(field=field):
                other = copy.deepcopy(self.classic)
                other[0]["Config"][field] = value
                self.assertNotEqual(IDENTITY.fingerprint(self.classic),
                                    IDENTITY.fingerprint(other))

    def test_config_array_order_and_empty_map_entries_are_significant(self):
        for field, value in (
            ("Env", list(reversed(self.classic[0]["Config"]["Env"]))),
            ("Labels", {"example.version": "1"}),
            ("Volumes", {}), ("ExposedPorts", {}),
        ):
            with self.subTest(field=field):
                other = copy.deepcopy(self.classic)
                other[0]["Config"][field] = value
                self.assertNotEqual(IDENTITY.fingerprint(self.classic),
                                    IDENTITY.fingerprint(other))

    def test_unset_healthcheck_fields_are_normalized(self):
        self.classic[0]["Config"]["Healthcheck"] = {"Test": ["CMD", "true"], "Interval": 0}
        self.containerd[0]["Config"]["Healthcheck"] = {"Test": ["CMD", "true"]}
        self.assertEqual(IDENTITY.fingerprint(self.classic),
                         IDENTITY.fingerprint(self.containerd))

    def test_scratch_image_can_have_no_layers(self):
        self.classic[0]["RootFS"]["Layers"] = None
        self.containerd[0]["RootFS"].pop("Layers")
        self.assertEqual(IDENTITY.fingerprint(self.classic),
                         IDENTITY.fingerprint(self.containerd))

    def test_incomplete_or_invalid_metadata_is_rejected(self):
        bad_results = [[], {}, [None], self.classic * 2, [{"Id": "sha256:only-an-id"}]]
        for field, value in (("Config", None), ("RootFS", None), ("Architecture", ""),
                             ("RootFS", {"Type": "layers", "Layers": ""}),
                             ("RootFS", {"Type": "layers", "Layers": ["bad"]})):
            other = copy.deepcopy(self.classic)
            other[0][field] = value
            bad_results.append(other)
        for result in bad_results:
            with self.subTest(result=result), self.assertRaises(ValueError):
                IDENTITY.fingerprint(result)

    def test_cli_outputs_only_fingerprint_or_generic_error(self):
        for payload, success in ((json.dumps(self.classic), True),
                                 ("", False), ("sensitive-invalid-data", False),
                                 (json.dumps([{"Config": {"Env": ["TOKEN=private"]}}]), False)):
            with self.subTest(success=success):
                result = subprocess.run([sys.executable, str(HELPER)], input=payload,
                                        text=True, capture_output=True)
                if success:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertRegex(result.stdout, r"^sha256:[0-9a-f]{64}\n$")
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(result.stderr, "Could not fingerprint Docker image metadata.\n")


if __name__ == "__main__":
    unittest.main()
