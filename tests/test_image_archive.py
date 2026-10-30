#!/usr/bin/env python3
"""Validate Docker save archives without Docker or image downloads."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[1] / "docker/image_archive.py"
SPEC = importlib.util.spec_from_file_location("image_archive", HELPER)
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


class ImageArchiveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "image.tar"
        self.config = {"os": "linux", "architecture": "arm64",
                       "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "1" * 64]}}

    def write_archive(self, files, links=None):
        with tarfile.open(self.path, "w") as archive:
            for name, value in files.items():
                data = value if isinstance(value, bytes) else json.dumps(value).encode()
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
            for name, target in (links or {}).items():
                entry = tarfile.TarInfo(name)
                entry.type = tarfile.LNKTYPE
                entry.linkname = target
                archive.addfile(entry)

    def files(self, config_name="config.json", layer_name="layer/layer.tar"):
        return {"manifest.json": [{"Config": config_name, "Layers": [layer_name]}],
                config_name: copy.deepcopy(self.config), layer_name: b"layer payload"}

    def test_classic_and_containerd_archives(self):
        for config_name, layer_name in (
            ("config.json", "layer/layer.tar"),
            ("blobs/sha256/config", "blobs/sha256/layer"),
        ):
            with self.subTest(config=config_name):
                files = self.files(config_name, layer_name)
                # Other platform descriptors may intentionally be absent.
                files["index.json"] = {"manifests": [{"digest": "sha256:absent-platform"}]}
                self.write_archive(files)
                ARCHIVE.validate(self.path)

    def test_metadata_only_or_missing_layer_export(self):
        for absent in ("config.json", "layer/layer.tar"):
            with self.subTest(absent=absent):
                files = self.files()
                del files[absent]
                self.write_archive(files)
                with self.assertRaises(ValueError):
                    ARCHIVE.validate(self.path)

    def test_truncated_layer_payload(self):
        files = self.files()
        files["layer/layer.tar"] = b"x" * 2048
        self.write_archive(files)
        with tarfile.open(self.path) as archive:
            cutoff = archive.getmember("layer/layer.tar").offset_data + 1024
        with self.path.open("r+b") as archive:
            archive.truncate(cutoff)
        with self.assertRaises((ValueError, tarfile.TarError)):
            ARCHIVE.validate(self.path)

    def test_scratch_image_without_layers(self):
        for diff_ids in ([], None):
            with self.subTest(diff_ids=diff_ids):
                files = self.files()
                files["manifest.json"][0]["Layers"] = []
                files["config.json"]["rootfs"]["diff_ids"] = diff_ids
                del files["layer/layer.tar"]
                self.write_archive(files)
                ARCHIVE.validate(self.path)

    def test_hardlinked_layers(self):
        files = self.files()
        files["payload"] = files.pop("layer/layer.tar")
        self.write_archive(files, {"layer/layer.tar": "payload"})
        ARCHIVE.validate(self.path)

    def test_empty_or_invalid_manifest_and_config(self):
        for key, value in (
            ("manifest.json", []), ("manifest.json", b"invalid"),
            ("config.json", {"config": {"Env": ["TOKEN=private"]}}),
            ("config.json", {**self.config, "rootfs": {"type": "layers", "diff_ids": []}}),
        ):
            with self.subTest(key=key):
                files = self.files()
                files[key] = value
                self.write_archive(files)
                result = subprocess.run([sys.executable, str(HELPER), str(self.path)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, "")
                self.assertIn("Saved image archive is incomplete or invalid", result.stderr)
                self.assertNotIn("TOKEN", result.stderr)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
