#!/usr/bin/env python3
"""Focused tests for the mods/fix-qwen3.8-ple-dtype mod.

Exercises patch_config.py against fixture configs laid out like a real HF
snapshot (blob file + snapshot symlink) and run.sh with an HF_HOME override.
No container, GPU, docker, or network access is required.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MOD_DIR = PROJECT_DIR / "mods" / "fix-qwen3.8-ple-dtype"
PATCH_TOOL = MOD_DIR / "patch_config.py"
RUN_SCRIPT = MOD_DIR / "run.sh"

MODEL_ID = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
REPO_DIR_NAME = "models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4"

QUANT_ENTRY = {"quant_algo": "NVFP4", "group_size": 16}


def make_config(*, ple_dtype=None, with_quant_entry=True):
    """Build a minimal but shape-faithful checkpoint config dict."""
    text_config = {
        "hidden_size": 2560,
        "ple_embed_dim": 2560,
        "ngram_vocab_size_base": 20000000,
        "split_ngram_parts": 128,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "vocab_size": 248320,
    }
    if ple_dtype is not None:
        text_config["ple_embedding_dtype"] = ple_dtype
    config = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": text_config,
    }
    if with_quant_entry:
        config["quantization_config"] = {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.language_model.layers.0.mlp.experts": QUANT_ENTRY,
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": QUANT_ENTRY,
            },
        }
    return config


def json_bytes(config, indent=2):
    return (json.dumps(config, indent=indent) + "\n").encode("utf-8")


class PleDtypeModTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    # --- helpers -----------------------------------------------------------

    def write_snapshot(self, config, *, indent=2, revision="rev0"):
        """Lay out one snapshot as blob file + symlink, like the HF cache."""
        blobs = self.tmp / "hub" / REPO_DIR_NAME / "blobs"
        snapshots = self.tmp / "hub" / REPO_DIR_NAME / "snapshots" / revision
        blobs.mkdir(parents=True, exist_ok=True)
        snapshots.mkdir(parents=True, exist_ok=True)
        blob = blobs / f"configblob-{revision}"
        blob.write_bytes(json_bytes(config, indent=indent))
        config_json = snapshots / "config.json"
        config_json.symlink_to(blob)
        return config_json, blob

    def write_refs(self, ref="main", revision="rev0"):
        refs_dir = self.tmp / "hub" / REPO_DIR_NAME / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / ref).write_text(revision + "\n")

    def run_patch(self, *paths):
        return subprocess.run(
            [sys.executable, str(PATCH_TOOL), *(str(p) for p in paths)],
            capture_output=True,
            text=True,
        )

    def run_mod(self, *, hub_root):
        env = dict(HF_HOME=str(hub_root))
        return subprocess.run(
            ["bash", str(RUN_SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
        )

    def read_config(self, path):
        return json.loads(path.read_text(encoding="utf-8"))

    # --- patch_config.py ---------------------------------------------------

    def test_patch_adds_dtype_and_preserves_symlink(self):
        config_json, blob = self.write_snapshot(make_config())
        result = self.run_patch(config_json)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("set ple_embedding_dtype", result.stdout)

        self.assertTrue(config_json.is_symlink(), "snapshot symlink must survive")
        self.assertEqual(config_json.resolve(), blob.resolve())
        patched = self.read_config(blob)
        self.assertEqual(
            patched["text_config"]["ple_embedding_dtype"], "nvfp4"
        )
        # Nothing else may change.
        expected = make_config()
        patched["text_config"].pop("ple_embedding_dtype")
        self.assertEqual(patched, expected)

    def test_patch_is_idempotent(self):
        config_json, blob = self.write_snapshot(make_config())
        self.assertEqual(self.run_patch(config_json).returncode, 0)
        first = blob.read_bytes()
        second = self.run_patch(config_json)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("already set", second.stdout)
        self.assertEqual(blob.read_bytes(), first, "no-op run must not rewrite")

    def test_accepts_consistent_alternate_dtype_name(self):
        # "uint8" maps to the same vLLM nvfp4_group16 storage mode.
        config_json, blob = self.write_snapshot(make_config(ple_dtype="uint8"))
        result = self.run_patch(config_json)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already set", result.stdout)
        self.assertEqual(
            self.read_config(blob)["text_config"]["ple_embedding_dtype"], "uint8"
        )

    def test_refuses_without_quantization_entry(self):
        config_json, blob = self.write_snapshot(make_config(with_quant_entry=False))
        before = blob.read_bytes()
        result = self.run_patch(config_json)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to guess", result.stderr)
        self.assertEqual(blob.read_bytes(), before, "refused run must not write")

    def test_refuses_contradictory_dtype(self):
        config_json, blob = self.write_snapshot(make_config(ple_dtype="bfloat16"))
        before = blob.read_bytes()
        result = self.run_patch(config_json)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contradicts", result.stderr)
        self.assertEqual(blob.read_bytes(), before, "refused run must not write")

    def test_refuses_non_family_config(self):
        config = {
            "architectures": ["SomeOtherModel"],
            "text_config": {"hidden_size": 4096},
            "quantization_config": {
                "quantized_layers": {
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding": QUANT_ENTRY,
                },
            },
        }
        config_json, blob = self.write_snapshot(config)
        before = blob.read_bytes()
        result = self.run_patch(config_json)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a Qwen3.8-Flash-Next checkpoint", result.stderr)
        self.assertEqual(blob.read_bytes(), before)

    # --- run.sh ------------------------------------------------------------

    def test_run_script_patches_ref_revision(self):
        config_json, blob = self.write_snapshot(make_config())
        self.write_refs("main", "rev0")
        result = self.run_mod(hub_root=self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Patching revision rev0", result.stdout)
        self.assertIn("set ple_embedding_dtype", result.stdout)
        self.assertEqual(
            self.read_config(blob)["text_config"]["ple_embedding_dtype"], "nvfp4"
        )

    def test_run_script_skips_stale_snapshots(self):
        # Mirrors the real cache: refs/main target plus an older snapshot
        # revision with a different architecture and no PLE quant entry.
        config_json, blob = self.write_snapshot(make_config(), revision="rev0")
        stale_config, stale_blob = self.write_snapshot(
            {
                "architectures": ["Qwen3_8FlashNextForConditionalGeneration"],
                "text_config": dict(
                    make_config(with_quant_entry=False)["text_config"],
                    ple_embedding_dtype="nvfp4",
                ),
            },
            revision="rev_old",
        )
        self.write_refs("main", "rev0")
        stale_before = stale_blob.read_bytes()
        result = self.run_mod(hub_root=self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.read_config(blob)["text_config"]["ple_embedding_dtype"], "nvfp4"
        )
        self.assertEqual(
            stale_blob.read_bytes(), stale_before, "stale snapshot must be untouched"
        )
        self.assertNotIn("rev_old", result.stdout)

    def test_run_script_falls_back_without_refs(self):
        config_json, blob = self.write_snapshot(make_config())
        result = self.run_mod(hub_root=self.tmp)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.read_config(blob)["text_config"]["ple_embedding_dtype"], "nvfp4"
        )

    def test_run_script_fails_when_model_missing(self):
        result = self.run_mod(hub_root=self.tmp / "empty")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found under", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
