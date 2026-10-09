#!/usr/bin/env python3
"""Exercise source-series application and lane guards without Docker or GPUs."""

import ast
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
APPLIER = PROJECT / "docker/apply_patch_series.sh"
UPSTREAM = {
    "vllm": "https://github.com/vllm-project/vllm.git",
    "flashinfer": "https://github.com/flashinfer-ai/flashinfer.git",
}


class PatchSeriesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.series = self.root / "patches"
        self.series.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        (self.source / "feature").write_text("base\n")
        self.commit()
        self.base = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.source, text=True, stderr=subprocess.PIPE
        )

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")

    def patch(self, name, path, content):
        (self.source / path).parent.mkdir(parents=True, exist_ok=True)
        (self.source / path).write_text(content)
        self.git("add", ".")
        (self.series / name).write_text(
            self.git("diff", "--cached", "--binary", "--full-index")
        )
        self.commit()

    def apply(self, project="vllm", repository=None, enabled="1"):
        return subprocess.run(
            ["bash", str(APPLIER), project, repository or UPSTREAM[project],
             enabled, str(self.series)],
            cwd=self.source, text=True, capture_output=True,
        )

    def make_series(self):
        self.patch("first.patch", "feature", "first\n")
        self.patch("second.patch", "feature", "second\n")
        (self.series / "series").write_text("# Ordered dependencies\nfirst.patch\n\nsecond.patch\n")
        self.git("checkout", "-q", self.base)

    def test_applies_in_order_to_both_upstream_projects(self):
        self.make_series()
        for project in UPSTREAM:
            with self.subTest(project=project):
                self.git("checkout", "-q", self.base)
                result = self.apply(project)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((self.source / "feature").read_text(), "second\n")
                self.assertEqual(self.git("status", "--porcelain"), "")

    def test_b12x_repositories_skip_even_when_explicitly_enabled(self):
        # No series file exists: a fork must exit before even opening it.
        for project in UPSTREAM:
            for suffix in ("", ".git", ".git/"):
                with self.subTest(project=project, suffix=suffix):
                    result = self.apply(
                        project, f"https://github.com/local-inference-lab/{project}{suffix}"
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Skipping regular", result.stdout)
                    self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.base)

    def test_disabled_upstream_selection_does_not_read_series(self):
        for project in UPSTREAM:
            result = self.apply(project, enabled="0")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.git("rev-parse", "HEAD").strip(), self.base)

    def test_already_present_patch_is_skipped(self):
        self.patch("first.patch", "feature", "first\n")
        (self.series / "series").write_text("first.patch\n")
        head = self.git("rev-parse", "HEAD")
        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Already present", result.stdout)
        self.assertEqual(self.git("rev-parse", "HEAD"), head)

    def test_doc_conflicts_stop_before_later_patches(self):
        self.patch("first.patch", "feature", "first\n")
        self.patch("docs.patch", "README.md", "incoming documentation\n")
        self.patch("later.patch", "later", "must not be applied\n")
        self.git("checkout", "-q", self.base)
        (self.source / "README.md").write_text("existing documentation\n")
        self.commit()
        (self.series / "series").write_text("first.patch\ndocs.patch\nlater.patch\n")
        result = self.apply()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("README.md", self.git("diff", "--name-only", "--diff-filter=U"))
        self.assertFalse((self.source / "later").exists())

    def test_nvfp4_patch_preserves_upstream_quantized_input_test(self):
        # #59612 inserted a sibling test where #54614 originally added its test.
        # Replay the real vendored hunk against that upstream excerpt.
        path = "tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py"
        upstream = (
            PROJECT / "tests/fixtures/qwen38-patch-series/nvfp4_upstream.py"
        ).read_text()
        target = self.source / path
        target.parent.mkdir(parents=True)
        target.write_text(upstream)
        self.commit()
        patch = (PROJECT / "docker/patches/qwen3.8/vllm/07-pr54614.patch").read_text()
        sections = re.split(r"(?m)(?=^diff --git )", patch)
        section = next(s for s in sections if s.startswith(f"diff --git a/{path} "))
        (self.series / "nvfp4.patch").write_text(section)
        (self.series / "series").write_text("nvfp4.patch\n")

        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        updated = target.read_text()
        compile(updated, path, "exec")
        before = ast.parse(upstream)
        after = ast.parse(updated)
        added = "test_dynamic_precision_preserves_canonical_weights"
        functions = [node for node in after.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(sum(node.name == added for node in functions), 1)
        after.body = [node for node in after.body if not (
            isinstance(node, ast.FunctionDef) and node.name == added
        )]
        # The complete upstream test, its decorators, and helper stay intact.
        self.assertEqual(ast.dump(before), ast.dump(after))
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_docker_vllm_selection_respects_presets_and_repository(self):
        self.patch("first.patch", "feature", "first\n")
        (self.series / "series").write_text("first.patch\n")
        dockerfile = (PROJECT / "Dockerfile").read_text()
        block = next(
            block for block in re.split(r"(?m)^RUN ", dockerfile)
            if block.startswith("set -eux;") and 'VLLM_ALL_PRS=""' in block
        ).split("\n\n# Targeted production subset", 1)[0]
        block = block.replace("/tmp/apply_patch_series.sh", str(APPLIER)).replace(
            "/tmp/regular-vllm", str(self.series)
        )
        # Stop before the existing network-fetching custom PR loop.
        block = block.split("    for pr in $VLLM_SELECTED_PRESET_PRS", 1)[0] + " true\n"
        cases = [
            (UPSTREAM["vllm"], "main", "", "", True),
            (UPSTREAM["vllm"], "main", "0", "", False),
            (UPSTREAM["vllm"], "custom-ref", "auto", "", False),
            (UPSTREAM["vllm"], "main", "auto", "12345", False),
            (UPSTREAM["vllm"], "custom-ref", "1", "12345", True),
            ("https://github.com/local-inference-lab/vllm", "main", "1", "", False),
            ("https://github.com/local-inference-lab/vllm", "dev/karmic-kraken", "0", "", False),
        ]
        for repository, ref, preset, prs, expected in cases:
            with self.subTest(repository=repository, ref=ref, preset=preset, prs=prs):
                self.git("checkout", "-q", self.base)
                result = subprocess.run(
                    ["sh", "-c", block], cwd=self.source, text=True, capture_output=True,
                    env={**os.environ, "VLLM_REPO": repository, "VLLM_REF": ref,
                         "VLLM_APPLY_PRESET_PRS": preset, "VLLM_PRS": prs,
                         "VLLM_PRESET_PRS": ""},
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(
                    (self.source / "feature").read_text(), "first\n" if expected else "base\n"
                )


if __name__ == "__main__":
    unittest.main()
