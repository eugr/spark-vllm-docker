#!/usr/bin/env python3
"""Offline cache checks using tiny synthetic files; no Hub, Docker or GPU calls."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recipe_runner", ROOT / "run-recipe.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
RECIPE_PATH = ROOT / "recipes/qwen3.8-flash-next-mxfp4-fp8-r12.yaml"


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.home = self.temp / "home"
        self.stack.enter_context(patch.object(Path, "home", return_value=self.home))
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.process = self.stack.enter_context(patch.object(
            runner.subprocess, "run", side_effect=AssertionError("Unexpected subprocess")
        ))
        self.recipe = runner.load_recipe(RECIPE_PATH)
        self.model = self.recipe["model"]
        self.revision = self.recipe["model_revision"]
        self.required = self.recipe["model_required_files"]
        self.snapshot = self.snapshot_at(self.home / ".cache/huggingface")

    def snapshot_at(self, hf_home):
        return (hf_home / "hub" / f"models--{self.model.replace('/', '--')}"
                / "snapshots" / self.revision)

    def populate(self, snapshot=None):
        snapshot = snapshot or self.snapshot
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in self.required:
            (snapshot / name).write_text("fixture")
        (snapshot / "config.json").write_text('{"model_type": "fixture"}')
        self.write_index({"layer.weight": "weights.safetensors",
                          "ple.weight": "ple.safetensors"}, snapshot)
        (snapshot / "weights.safetensors").write_bytes(b"synthetic weights")
        (snapshot / "ple.safetensors").write_bytes(b"synthetic PLE")

    def write_index(self, weight_map, snapshot=None):
        ((snapshot or self.snapshot) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

    def exists(self):
        return runner.check_model_exists(self.model, self.revision, self.required)

    def run_main(self, *options):
        output = io.StringIO()
        argv = ["run-recipe.py", str(RECIPE_PATH), "--config", os.devnull, *options]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            result = runner.main()
        return result, output.getvalue()

    def test_recipe_asset_names(self):
        self.assertEqual(set(self.required), {
            "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
            "vocab.json", "merges.txt", "preprocessor_config.json",
            "video_preprocessor_config.json", "generation_config.json",
            "hybrid-conversion-manifest.json", "draft_lm_head_mxfp4.safetensors",
            "config.json", "model.safetensors.index.json",
        })

    def test_complete_snapshot(self):
        self.populate()
        self.assertTrue(self.exists())

    def test_absent_or_empty_snapshot(self):
        self.assertFalse(self.exists())
        self.snapshot.mkdir(parents=True)
        self.assertFalse(self.exists())

    def test_missing_shard_including_ple(self):
        for name in ("weights.safetensors", "ple.safetensors"):
            with self.subTest(name=name):
                self.populate()
                (self.snapshot / name).unlink()
                self.assertFalse(self.exists())

    def test_missing_auxiliary(self):
        for name in self.required:
            with self.subTest(name=name):
                self.populate()
                (self.snapshot / name).unlink()
                self.assertFalse(self.exists())

    def test_zero_byte_file(self):
        for name in [*self.required, "weights.safetensors", "ple.safetensors"]:
            with self.subTest(name=name):
                self.populate()
                (self.snapshot / name).write_bytes(b"")
                self.assertFalse(self.exists())

    def test_corrupt_index(self):
        self.populate()
        for data in (b"{broken", b"[]", b"null", b"{}", b"\xff", b'{"metadata":{}}'):
            with self.subTest(data=data):
                (self.snapshot / "model.safetensors.index.json").write_bytes(data)
                self.assertFalse(self.exists())

    def test_invalid_weight_map(self):
        self.populate()
        for mapping in ({}, [], None, {"tensor": 3}, {"": "weights.safetensors"}):
            with self.subTest(mapping=mapping):
                self.write_index(mapping)
                self.assertFalse(self.exists())

    def test_duplicate_keys_and_nonfinite_json(self):
        self.populate()
        for data in (
            '{"weight_map":{"t":"missing.safetensors","t":"weights.safetensors"}}',
            '{"weight_map":{"t":"missing.safetensors"},'
            '"weight_map":{"t":"weights.safetensors"}}',
            '{"metadata":{"size":NaN},"weight_map":{"t":"weights.safetensors"}}',
        ):
            with self.subTest(data=data):
                (self.snapshot / "model.safetensors.index.json").write_text(data)
                self.assertFalse(self.exists())

    def test_corrupt_config(self):
        self.populate()
        for data in ("{broken", "[]", "null", "{}"):
            with self.subTest(data=data):
                (self.snapshot / "config.json").write_text(data)
                self.assertFalse(self.exists())

    def test_valid_blob_symlinks_outside_snapshot(self):
        self.populate()
        blobs = self.snapshot.parent.parent / "blobs"
        blobs.mkdir()
        for number, path in enumerate(self.snapshot.iterdir()):
            target = blobs / f"blob-{number}"
            path.rename(target)
            path.symlink_to(Path("../../blobs") / target.name)
        self.assertTrue(self.exists())

    def test_dangling_symlink(self):
        for name in ("weights.safetensors", "tokenizer.json", "config.json"):
            with self.subTest(name=name):
                self.populate()
                path = self.snapshot / name
                path.unlink()
                path.symlink_to("../../blobs/not-downloaded")
                self.assertFalse(self.exists())
                path.unlink()

    def test_symlink_loop(self):
        self.populate()
        path = self.snapshot / "weights.safetensors"
        path.unlink()
        path.symlink_to(path.name)
        self.assertFalse(self.exists())

    def test_directory_is_not_a_shard(self):
        self.populate()
        path = self.snapshot / "weights.safetensors"
        path.unlink()
        path.mkdir()
        self.assertFalse(self.exists())

    def test_unsafe_index_paths(self):
        self.populate()
        for name in ("../outside", "/etc/passwd", "./weights.safetensors", "",
                     "sub/../../outside", "a\\b", "a\x00b", "a//b", "a" * 256):
            with self.subTest(name=name):
                self.write_index({"tensor": name})
                self.assertFalse(self.exists())

    def test_metadata_size_and_entry_bounds(self):
        self.populate()
        with patch.object(runner, "MODEL_INDEX_MAX_BYTES", 8):
            self.assertFalse(self.exists())
        with patch.object(runner, "MODEL_CONFIG_MAX_BYTES", 8):
            self.assertFalse(self.exists())
        with patch.object(runner, "MODEL_INDEX_MAX_ENTRIES", 1):
            self.assertFalse(self.exists())
        index = self.snapshot / "model.safetensors.index.json"
        index.write_text('{"weight_map":' + "[" * 2000 + "0" + "]" * 2000 + "}")
        self.assertFalse(self.exists())

    def test_alternate_hf_home_does_not_use_default_cache(self):
        self.populate()
        alternate = self.temp / "alternate"
        with patch.dict(os.environ, {"HF_HOME": str(alternate)}):
            self.assertFalse(self.exists())
            self.populate(self.snapshot_at(alternate))
            self.assertTrue(self.exists())

    def test_wrong_revision(self):
        self.populate()
        self.assertFalse(runner.check_model_exists(self.model, "a" * 40, self.required))
        for revision in (None, "../outside", "/tmp", "a/b"):
            self.assertFalse(runner.check_model_exists(self.model, revision, self.required))

    def test_legacy_recipe_behavior_unchanged(self):
        self.snapshot.mkdir(parents=True)
        with patch.dict(os.environ, {"HF_HOME": str(self.temp / "alternate")}):
            self.assertTrue(runner.check_model_exists(self.model, self.revision))
            self.assertTrue(runner.check_model_exists(self.model))

    def test_schema_rejects_bad_required_files(self):
        path = self.temp / "recipe.yaml"
        for value in (None, [], "config.json", [None], ["../outside"], ["a"] * 257):
            with self.subTest(value=value):
                recipe = dict(self.recipe, model_required_files=value)
                path.write_text(runner.yaml.safe_dump(recipe))
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    runner.load_recipe(path)

    def test_matching_hub_override_allowed(self):
        alternate = self.temp / "alternate"
        with patch.dict(os.environ, {"HF_HOME": str(alternate),
                                   "HF_HUB_CACHE": str(alternate / "hub")}):
            runner.validate_model_cache_environment()

    def test_conflicting_hub_override_before_any_side_effect(self):
        for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            for value in (str(self.temp / "wrong"), ""):
                with self.subTest(variable=variable, value=value), patch.dict(
                    os.environ, {variable: value}
                ), patch.object(runner, "run_autodiscover") as discover, patch.object(
                    runner, "check_image_exists"
                ) as image, patch.object(runner, "download_model") as download:
                    result, output = self.run_main("--setup", "--discover", "--solo")
                    self.assertEqual(result, 1)
                    self.assertIn(variable, output)
                    self.assertIn("HF_HOME", output)
                    discover.assert_not_called()
                    image.assert_not_called()
                    download.assert_not_called()
        self.process.assert_not_called()

    def test_complete_setup_never_calls_downloader(self):
        alternate = self.temp / "alternate"
        self.populate(self.snapshot_at(alternate))
        with patch.dict(os.environ, {"HF_HOME": str(alternate)}), patch.object(
            runner, "check_image_exists", return_value=True
        ), patch.object(runner, "download_model") as download:
            result, _ = self.run_main("--setup", "--download-only", "--solo")
            self.assertEqual(result, 0)
            download.assert_not_called()
        self.process.assert_not_called()

    def test_incomplete_setup_resumes_pinned_download(self):
        self.populate()
        (self.snapshot / "weights.safetensors").unlink()
        with patch.object(runner, "check_image_exists", return_value=True), patch.object(
            runner, "download_model", return_value=True
        ) as download:
            result, _ = self.run_main("--setup", "--download-only", "--solo")
            self.assertEqual(result, 0)
            download.assert_called_once_with(self.model, None, self.revision)
        self.process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
