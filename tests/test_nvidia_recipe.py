#!/usr/bin/env python3
"""Tests for the NVIDIA recipe translator without launching containers."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("nvidia_recipe", ROOT / "run-nvidia-recipe.py")
translator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(translator)
runner_spec = importlib.util.spec_from_file_location("recipe_runner", ROOT / "run-recipe.py")
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


def source(image="vllm/vllm-openai:v0.31.0", nodes=2):
    return {
        "schema_version": 4,
        "backend": "vllm",
        "title": "Example Qwen",
        "runtime": {"kind": "container", "image": image},
        "checkpoint": {"target_label": "nvidia/Qwen"},
        "topology": {"node_count": nodes},
        "serve": {"command": """export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve nvidia/Qwen \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --speculative-config '{"method":"dflash","model":"z-lab/draft"}' \\
  --tensor-parallel-size 2 \\
  --nnodes=2 --node-rank $NODE_RANK \\
  --master-addr "$HEAD_IP" --master-port "$MASTER_PORT" \\
  --distributed-executor-backend mp --headless"""},
    }


class TranslationTests(unittest.TestCase):
    def test_translation_preserves_vllm_options_and_removes_orchestration(self):
        recipe = translator.convert(source(), None, False)
        self.assertEqual(recipe["container"], "vllm/vllm-openai:v0.31.0")
        self.assertEqual(recipe["model"], "nvidia/Qwen")
        self.assertTrue(recipe["cluster_only"])
        self.assertTrue(recipe["pull_image"])
        self.assertEqual(recipe["env"], {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        self.assertNotIn("--nnodes", recipe["command"])
        self.assertNotIn("--node-rank", recipe["command"])
        self.assertNotIn("--master-addr", recipe["command"])
        self.assertNotIn("--master-port", recipe["command"])
        self.assertNotIn("--headless", recipe["command"])
        script = runner.generate_launch_script(recipe, {"port": 9000}, is_solo=False)
        self.assertIn("--port 9000", script)
        self.assertIn('"method":"dflash"', script)
        self.assertIn("--tensor-parallel-size 2", script)
        self.assertNotIn("'\n'", script)

    def test_image_mapping_and_override(self):
        for image, expected in (
            ("eugr/spark-vllm:latest", "vllm-node"),
            ("eugr/spark-vllm-b12x:nightly", "vllm-node-b12x"),
            ("other/image:tag", "vllm-node"),
        ):
            with self.subTest(image=image):
                self.assertEqual(translator.convert(source(image), None, True)["container"], expected)
                self.assertEqual(translator.convert(source(image), "custom:tag", True)["container"], "custom:tag")
                self.assertNotIn("pull_image", translator.convert(source(image), "custom:tag", True))

    def test_export_does_not_launch_and_is_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvidia.yaml"
            exported = Path(directory) / "export.yaml"
            path.write_text(yaml.safe_dump(source()))
            with patch.object(translator.subprocess, "call") as call, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(translator.main([str(path), "--export", str(exported)]), 0)
            call.assert_not_called()
            recipe = runner.load_recipe(exported)
            self.assertEqual(recipe["container"], "vllm/vllm-openai:v0.31.0")
            self.assertEqual(translator.main([str(path), "--export", str(exported)]), 1)

    def test_export_without_filename_prints_only_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvidia.yaml"
            path.write_text(yaml.safe_dump(source()))
            output = io.StringIO()
            with patch.object(translator.subprocess, "call") as call, contextlib.redirect_stdout(output):
                self.assertEqual(translator.main([str(path), "--export"]), 0)
            call.assert_not_called()
            recipe = yaml.safe_load(output.getvalue())
            self.assertEqual(recipe["container"], "vllm/vllm-openai:v0.31.0")
            self.assertEqual(recipe["recipe_version"], "1")

    def test_solo_auto_mode_and_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nvidia.yaml"
            path.write_text(yaml.safe_dump(source(nodes=1)))
            def inspect(command):
                recipe = runner.load_recipe(Path(command[1]))
                self.assertTrue(recipe["solo_only"])
                self.assertEqual(command[2:], ["--solo", "--dry-run", "--port", "9001"])
                return 0
            with patch.object(translator.subprocess, "call", side_effect=inspect):
                self.assertEqual(translator.main([str(path), "--dry-run", "--port", "9001"]), 0)

    def test_github_blob_url_becomes_raw_url(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read(self):
                return b"backend: vllm\n"

        with patch.object(translator, "urlopen", return_value=Response()) as open_url:
            text = translator.read_source("https://github.com/NVIDIA/local-ai-inference-recipes/blob/main/example/recipe.yaml")
        self.assertEqual(text, "backend: vllm\n")
        self.assertEqual(open_url.call_args.args[0], "https://raw.githubusercontent.com/NVIDIA/local-ai-inference-recipes/main/example/recipe.yaml")

    def test_rejects_unsupported_backend(self):
        item = source()
        item["backend"] = "sglang"
        with self.assertRaisesRegex(ValueError, "NVIDIA vLLM"):
            translator.convert(item, None, False)

    def test_runner_pulls_external_image_and_copies_it_without_building(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "converted.yaml"
            path.write_text(yaml.safe_dump(translator.convert(source(), None, False)))
            argv = ["run-recipe.py", str(path), "--config", "/dev/null", "-n", "192.0.2.1,192.0.2.2", "--build-only"]
            with patch.object(sys, "argv", argv), \
                    patch.object(runner, "check_image_exists", return_value=False), \
                    patch.object(runner, "build_image", return_value=True) as build, \
                    patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
            run.assert_called_once_with(["docker", "pull", "vllm/vllm-openai:v0.31.0"])
            build.assert_called_once_with("vllm/vllm-openai:v0.31.0", ["192.0.2.2"], copy_only=True)


if __name__ == "__main__":
    unittest.main()
