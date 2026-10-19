#!/usr/bin/env python3
"""Test vLLM/DeepGEMM compatibility using local Git repos, without a GPU or Docker."""

import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_deepgemm", PROJECT_DIR / "docker/prepare_deepgemm.py"
)
deepgemm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepgemm)

# Dependency declarations at vLLM e161a0798 (stable ABI) and 6bbad6ac (legacy).
UPSTREAM_REPO = "https://github.com/vllm-project/DeepGEMM.git"
STABLE_REF = "1e1842a833699298f7afc02eefb2ed168fff6938"
OLD_REF = "e1f418c2a4f20818221f6b0e578b4c2f634d4c3f"


class DeepGemmSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vllm = self.root / "vllm"
        self.cmake = self.vllm / "cmake/external_projects/deepgemm.cmake"
        self.cmake.parent.mkdir(parents=True)
        self.cache = self.root / "cache"
        self.env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }

    def write_cmake(self, stable=True, repo=UPSTREAM_REPO, ref=None):
        ref = ref or (STABLE_REF if stable else OLD_REF)
        dependency = "deep_gemm/_C.py" if stable else "csrc/python_api.cpp"
        self.cmake.write_text(
            f'set(_DEEPGEMM_UPSTREAM_REPO "{repo}")\n'
            f'set(_DEEPGEMM_UPSTREAM_TAG "{ref}")\n'
            'add_custom_command(OUTPUT "${_dg_marker}"\n'
            f'  DEPENDS "${{deepgemm_SOURCE_DIR}}/{dependency}")\n'
        )

    def git(self, cwd, *args):
        return subprocess.run(
            ["git", *args], cwd=cwd, env=self.env, check=True,
            text=True, capture_output=True,
        ).stdout.strip()

    def make_repo(self, name, stable=True):
        repo = self.root / name
        repo.mkdir()
        self.git(repo, "init", "-b", "main")
        files = ["csrc/python_api.cpp", "deep_gemm/__init__.py"]
        if stable:
            files += ["deep_gemm/_C.py", "setup.py"]
        for name in files:
            path = repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-m", "initial")
        return repo

    def run_prepare(self, destination="checkout", repo="", ref=""):
        # Execute the actual Docker RUN command with temporary paths. This also
        # exercises its argument wiring and propagation of helper failures.
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        run = next(
            block for block in re.split(r"(?m)^RUN ", dockerfile)
            if block.startswith("--mount=") and "python3 /tmp/prepare_deepgemm.py" in block
        ).split("\n\n", 1)[0]
        run = re.sub(r"^--mount=\S+\s*\\\n", "", run)
        run = run.replace("/repo-cache", str(self.cache)).replace(
            "/tmp/prepare_deepgemm.py", str(PROJECT_DIR / "docker/prepare_deepgemm.py")
        )
        return subprocess.run(
            ["sh", "-c", run], cwd=self.vllm,
            env={**self.env, "DEEPGEMM_REPO": repo, "DEEPGEMM_REF": ref,
                 "DEEPGEMM_SRC_DIR": str(self.root / destination)},
            text=True, capture_output=True,
        )

    def test_stable_abi_uses_selected_vllm_dependency(self):
        self.write_cmake()
        self.assertEqual(deepgemm.select_source(self.vllm), (UPSTREAM_REPO, STABLE_REF, True))

    def test_older_regular_and_b12x_keep_sm121_workaround(self):
        for ref in (OLD_REF, "ad1f1726aa540a76c1d26d6a120effb8de21eaa4"):
            with self.subTest(ref=ref):
                self.write_cmake(stable=False, ref=ref)
                self.assertEqual(
                    deepgemm.select_source(self.vllm),
                    (deepgemm.LEGACY_REPO, deepgemm.LEGACY_REF, False),
                )

    def test_commented_shim_does_not_select_stable_abi(self):
        self.write_cmake(stable=False)
        with self.cmake.open("a") as cmake:
            cmake.write('# TODO: vendor deep_gemm/_C.py after stable-ABI migration\n')
        self.assertFalse(deepgemm.select_source(self.vllm)[2])

    def test_explicit_overrides_take_precedence(self):
        self.write_cmake()
        for repo, ref, expected in (
            ("custom-repo", "custom-ref", ("custom-repo", "custom-ref", True)),
            ("custom-repo", "", ("custom-repo", STABLE_REF, True)),
            ("", "custom-ref", (UPSTREAM_REPO, "custom-ref", True)),
        ):
            with self.subTest(repo=repo, ref=ref):
                self.assertEqual(deepgemm.select_source(self.vllm, repo, ref), expected)

    def test_unresolved_stable_pin_fails_instead_of_falling_back_to_legacy(self):
        for declaration in (
            "", 'set(_DEEPGEMM_UPSTREAM_TAG "${OTHER_PIN}")',
            'set(_DEEPGEMM_UPSTREAM_TAG "one")\nset(_DEEPGEMM_UPSTREAM_TAG "two")',
        ):
            with self.subTest(declaration=declaration):
                self.write_cmake()
                self.cmake.write_text(self.cmake.read_text().replace(
                    f'set(_DEEPGEMM_UPSTREAM_TAG "{STABLE_REF}")', declaration
                ))
                with self.assertRaisesRegex(ValueError, "_DEEPGEMM_UPSTREAM_TAG"):
                    deepgemm.select_source(self.vllm)

    def test_checkout_uses_pin_and_preserves_git_provenance(self):
        repo = self.make_repo("upstream")
        commit = self.git(repo, "rev-parse", "HEAD")
        self.git(repo, "commit", "--allow-empty", "-m", "newer unselected revision")
        self.write_cmake(repo=str(repo), ref=commit)
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.root / "checkout"
        self.assertTrue((target / "deep_gemm/_C.py").is_file())
        self.assertEqual(self.git(target, "rev-parse", "HEAD"), commit)

    def test_incompatible_legacy_override_fails_before_compilation(self):
        repo = self.make_repo("old-deepgemm", stable=False)
        self.write_cmake()
        result = self.run_prepare(repo=str(repo), ref="main")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incompatible with vLLM's stable-ABI build", result.stderr)
        self.assertIn("deep_gemm/_C.py", result.stderr)
        self.assertFalse((self.root / "checkout").exists())

    def test_missing_ref_is_not_replaced_with_head(self):
        repo = self.make_repo("upstream")
        self.write_cmake(repo=str(repo), ref="nonexistent-ref")
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "checkout").exists())

    def test_legacy_source_can_be_prepared_without_stable_shim(self):
        repo = self.make_repo("legacy", stable=False)
        self.write_cmake(stable=False)
        result = self.run_prepare(repo=str(repo), ref="main")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "checkout/deep_gemm/_C.py").exists())

    def test_repo_caches_are_isolated_and_branches_refresh(self):
        sources = [self.make_repo(name) for name in ("upstream", "fork")]
        old_cache = self.cache / "deepgemm"
        old_cache.mkdir(parents=True)
        sentinel = old_cache / "untouched"
        sentinel.write_text("old shared cache")
        for index, repo in enumerate((*sources, *sources)):
            (repo / "identity").write_text(f"{repo.name} revision {index}")
            self.git(repo, "add", ".")
            self.git(repo, "commit", "-m", "update")
            self.write_cmake(repo=str(repo), ref="main")
            result = self.run_prepare(destination=f"checkout-{index}")
            self.assertEqual(result.returncode, 0, result.stderr)
            target = self.root / f"checkout-{index}"
            self.assertEqual(self.git(target, "rev-parse", "HEAD"), self.git(repo, "rev-parse", "HEAD"))
            self.assertEqual(self.git(target, "remote", "get-url", "origin"), str(repo))
        self.assertEqual(len(list(self.cache.glob("deepgemm-*"))), 2)
        self.assertEqual(sentinel.read_text(), "old shared cache")

    def test_cached_checkout_does_not_hide_incompatible_revision(self):
        repo = self.make_repo("upstream")
        self.write_cmake(repo=str(repo), ref="main")
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.git(repo, "rm", "deep_gemm/_C.py")
        self.git(repo, "commit", "-m", "remove stable-ABI shim")
        result = self.run_prepare(destination="incompatible")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing deep_gemm/_C.py", result.stderr)
        self.assertFalse((self.root / "incompatible").exists())

    def test_submodules_are_initialized_and_copied_with_provenance(self):
        self.env["GIT_ALLOW_PROTOCOL"] = "file"
        dependency = self.make_repo("headers")
        repo = self.make_repo("upstream")
        self.git(repo, "submodule", "add", str(dependency), "third-party/headers")
        self.git(repo, "commit", "-m", "add dependency")
        self.write_cmake(repo=str(repo), ref="main")
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        target = self.root / "checkout/third-party/headers"
        self.assertTrue((target / "csrc/python_api.cpp").is_file())
        self.assertEqual(self.git(target, "rev-parse", "HEAD"), self.git(dependency, "rev-parse", "HEAD"))

    def test_docker_prepares_dependency_after_patches_before_compilation(self):
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        prepare = dockerfile.index("python3 /tmp/prepare_deepgemm.py")
        self.assertLess(dockerfile.index("Final vLLM source after PR application"), prepare)
        self.assertLess(dockerfile.index("RUN python3 /tmp/vllm-patches/patch_vllm_wsl_cuda_uma.py"), prepare)
        self.assertLess(prepare, dockerfile.index("# Final Compilation"))
        self.assertIn('ARG DEEPGEMM_REPO=""', dockerfile)
        self.assertIn('ARG DEEPGEMM_REF=""', dockerfile)


if __name__ == "__main__":
    unittest.main()
