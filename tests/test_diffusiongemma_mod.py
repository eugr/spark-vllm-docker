#!/usr/bin/env python3
"""Check DiffusionGemma capability detection without importing vLLM or GPUs."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
MOD = PROJECT / "mods/diffusiongemma/run.sh"


class DiffusionGemmaModTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "site-packages"
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.git_calls = self.root / "git-calls"
        git = self.bin / "git"
        git.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$GIT_CALLS"\nexit 1\n')
        git.chmod(0o755)
        self.write("model_executor/models/diffusion_gemma.py", (
            "from transformers import DiffusionGemmaConfig\n"
            "class DiffusionGemmaForConditionalGeneration:\n"
            "    self_conditioning_embeds = None\n"
        ))
        self.write("model_executor/models/registry.py", "DiffusionGemmaForBlockDiffusion\n")
        self.write("config/diffusion.py", "class DiffusionConfig:\n    pass\n")
        self.write("transformers_utils/config.py", "# Uses native Transformers configs.\n")
        self.write("v1/attention/backends/flash_attn.py", "dynamic_causal=dynamic_causal\n")
        self.write("vllm_flash_attn/flash_attn_interface.py", (
            'dynamic_causal: "torch.Tensor | None" = None\n'
        ))
        self.write("parser/gemma4.py", (
            "class Gemma4Parser(ParserEngine):\n"
            "    def _preprocess_feed(self): pass\n"
        ))
        self.write("parser/engine/registered_adapters.py", "Gemma4ParserReasoningAdapter\n")
        self.write("reasoning/gemma4_engine_reasoning_parser.py", "Gemma4ParserReasoningAdapter\n")
        self.write("tool_parsers/gemma4_engine_tool_parser.py", "Gemma4EngineToolParser\n")

    def write(self, name, text):
        path = self.package / "vllm" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def run_mod(self):
        return subprocess.run(
            ["bash", str(MOD)], cwd=self.workspace, text=True, capture_output=True,
            env={
                **os.environ,
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                "GIT_CALLS": str(self.git_calls),
                "PYTHON_ROOT": str(self.package),
                "WORKSPACE_DIR": str(self.workspace),
            },
        )

    def assert_skips_existing_support(self):
        # Run twice to verify a repeated mod application remains harmless.
        for _ in range(2):
            result = self.run_mod()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("already has DiffusionGemma support", result.stdout)
        self.assertFalse(self.git_calls.exists(), "Existing support must not be patched")
        self.assertEqual(
            (self.workspace / "fixed_chat_template.jinja").read_bytes(),
            (MOD.parent / "chat_template_no_think.jinja").read_bytes(),
        )

    def test_native_transformers_config_skips_legacy_patch(self):
        self.assert_skips_existing_support()

    def test_custom_config_registry_still_skips_legacy_patch(self):
        self.write("transformers_utils/config.py", 'diffusion_gemma="DiffusionGemmaConfig"\n')
        self.assert_skips_existing_support()

    def test_incomplete_support_still_requires_patch(self):
        for name in (
            "model_executor/models/diffusion_gemma.py",
            "model_executor/models/registry.py",
            "config/diffusion.py",
        ):
            with self.subTest(missing=name):
                path = self.package / "vllm" / name
                original = path.read_text()
                path.unlink()
                try:
                    result = self.run_mod()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Patch could not be applied", result.stderr)
                    self.assertIn("apply --check", self.git_calls.read_text())
                    self.assertFalse((self.workspace / "fixed_chat_template.jinja").exists())
                finally:
                    path.write_text(original)


if __name__ == "__main__":
    unittest.main()
