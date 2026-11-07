#!/usr/bin/env python3
"""Exercise source-series application and lane guards without Docker or GPUs."""

import ast
import bisect
from contextlib import nullcontext
from functools import partial
import functools
import os
from pathlib import Path
import re
import subprocess
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


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

    def apply_upstream_excerpt(self, patch_name, path, fixture, hunk_terms=(), project="vllm"):
        target = self.source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        upstream = (PROJECT / "tests/fixtures/qwen38-patch-series" / fixture).read_text()
        target.write_text(upstream)
        self.commit()
        patch = (PROJECT / "docker/patches/qwen3.8" / project / patch_name).read_text()
        section = next(
            s for s in re.split(r"(?m)(?=^diff --git )", patch)
            if s.startswith(f"diff --git a/{path} ")
        )
        if hunk_terms:
            header, *hunks = re.split(r"(?m)(?=^@@ )", section)
            selected = [h for h in hunks if any(term in h for term in hunk_terms)]
            self.assertEqual(len(selected), len(hunk_terms))
            section = header + "".join(selected)
        (self.series / "excerpt.patch").write_text(section)
        (self.series / "series").write_text("excerpt.patch\n")
        result = self.apply(project=project)
        self.assertEqual(result.returncode, 0, result.stderr)
        updated = target.read_text()
        compile(updated, path, "exec")
        return upstream, updated

    def test_qsa_ring_patch_preserves_upstream_bf16_indexer_test(self):
        upstream, updated = self.apply_upstream_excerpt(
            "05-pr54912.patch", "tests/models/qwen4_exp/test_config.py",
            "qsa_config_upstream.py",
        )
        before, after = ast.parse(upstream), ast.parse(updated)
        added_tests = {
            "test_qsa_ring_capacity_divides_block_size",
            "test_qsa_ring_capacity_widens_within_bound",
            "test_qsa_ring_capacity_refuses_disproportionate_widening",
        }
        added_imports = {
            "vllm.models.qwen4_exp.common.qsa_indexer",
            "vllm.utils.math_utils",
        }
        functions = [n.name for n in after.body if isinstance(n, ast.FunctionDef)]
        for name in added_tests:
            self.assertEqual(functions.count(name), 1)
        self.assertEqual(
            {n.module: [alias.name for alias in n.names]
             for n in after.body if isinstance(n, ast.ImportFrom)
             and n.module in added_imports},
            {"vllm.models.qwen4_exp.common.qsa_indexer":
             ["QSA_RING_MAX_WIDENING", "qsa_ring_capacity"],
             "vllm.utils.math_utils": ["cdiv"]},
        )
        after.body = [n for n in after.body if not (
            isinstance(n, ast.FunctionDef) and n.name in added_tests
            or isinstance(n, ast.ImportFrom) and n.module in added_imports
        )]
        # Preserve the new model imports and BF16-indexer regression verbatim.
        self.assertEqual(ast.dump(before), ast.dump(after))
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_adaptive_mtp_preserves_upstream_module_and_prefix_cache_tests(self):
        upstream, updated = self.apply_upstream_excerpt(
            "04-pr60068.patch", "tests/v1/e2e/spec_decode/mtp/qwen3_5/test_mtp.py",
            "mtp_modules_upstream.py",
        )
        before, after = ast.parse(upstream), ast.parse(updated)
        added = "test_mtp_draft_confidence_stop"
        self.assertEqual(sum(getattr(n, "name", None) == added for n in after.body), 1)
        after.body = [n for n in after.body if getattr(n, "name", None) != added]
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_adaptive_depth_preserves_decode_graph_eligibility(self):
        _, updated = self.apply_upstream_excerpt(
            "04-pr60068.patch", "vllm/v1/worker/gpu/model_runner.py",
            "decode_graph_dispatch_upstream.py",
            ("+            max_query_len = int(batch_req_state",),
        )
        namespace = {}
        exec(updated, namespace)
        runner = namespace["Runner"]()
        runner.pcp_manager = None
        scheduler = SimpleNamespace(
            num_scheduled_tokens={"request": 5}, total_num_scheduled_tokens=5,
        )
        for eligible, expected in ((True, 2), (False, None)):
            state = SimpleNamespace(
                num_tokens=2, decode_graph_eligible=eligible,
                num_scheduled_tokens=SimpleNamespace(max=lambda: 2),
            )
            runner.gather_batch_req_state = lambda *args: (state, None)
            self.assertEqual(runner.prepare_dispatch(scheduler, False), (2, expected))
        runner.gather_batch_req_state = lambda *args: (None, None)
        self.assertEqual(runner.prepare_dispatch(scheduler, True), (5, 5))

    def test_loader_copy_preserves_upstream_mmap_prefaulting(self):
        _, updated = self.apply_upstream_excerpt(
            "17-flash-next-b12x-loader.patch", "vllm/model_executor/utils.py",
            "weight_copy_upstream.py",
        )
        events = []

        class Source:
            def numel(self):
                return self.size

            def is_contiguous(self):
                return self.contiguous

            def reshape(self, *args):
                return self

            def view(self, *args):
                return self

            def __getitem__(self, index):
                return self

            def sum(self):
                events.append("prefault")

        source = Source()
        destination = SimpleNamespace(dtype="bf16", device="cuda")

        def transfer(dst, src):
            self.assertIs(dst, destination)
            self.assertIs(src, source)
            events.append("transfer")
            return dst

        namespace = {
            "torch": SimpleNamespace(uint8="uint8"),
            "mmap": SimpleNamespace(PAGESIZE=4096),
        }
        exec("from __future__ import annotations\n" + updated, namespace)
        cases = [
            ("cpu", "bf16", 16, True, True, 2),
            ("cpu", "bf16", 1, True, True, 0),
            ("cpu", "fp32", 16, True, True, 0),
            ("cpu", "bf16", 16, False, True, 0),
            ("cpu", "bf16", 16, True, False, 0),
            ("meta", "bf16", 16, True, True, 0),
            ("cuda", "bf16", 16, True, True, 0),
        ]
        with patch.dict("sys.modules", {
            "vllm.model_executor.weight_transfer": SimpleNamespace(copy_weight=transfer),
        }):
            for device, dtype, size, contiguous, integrated, touches in cases:
                with self.subTest(device=device, dtype=dtype, size=size,
                                  contiguous=contiguous, integrated=integrated):
                    events.clear()
                    source.device = SimpleNamespace(type=device)
                    source.dtype, source.size, source.contiguous = dtype, size, contiguous
                    namespace["_is_integrated_gpu"] = lambda device: integrated
                    self.assertIs(namespace["copy_weight_"](destination, source), destination)
                    self.assertEqual(events, ["prefault"] * touches + ["transfer"])

    def test_mamba_graph_inputs_alias_buffers_and_preserve_fallback_padding(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU tensor checks require torch")
        _, updated = self.apply_upstream_excerpt(
            "18-pr65182.patch", "vllm/v1/attention/backends/mamba_attn.py",
            "mamba_graph_inputs_upstream.py",
            ("+    def bind_mrv2_graph_inputs(", "+            assert state_indices_tensor_d"),
        )
        namespace = {"torch": torch, "NULL_BLOCK_ID": -1}
        exec("from __future__ import annotations\n" + updated, namespace)
        cls = namespace["BaseMambaAttentionMetadataBuilder"]
        for depth in (0, 2):
            for bind in (False, True):
                with self.subTest(depth=depth, bind=bind):
                    builder = cls.__new__(cls)
                    builder.use_spec_decode = depth > 0
                    builder.decode_cudagraph_max_bs = 8
                    builder.compilation_config = SimpleNamespace(
                        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: True),
                    )
                    aligned = torch.arange(8 * (1 + depth), dtype=torch.int32).view(8, -1)
                    accepted = torch.arange(1, 9, dtype=torch.int32) if depth else None
                    builder.state_indices_tensor_d = torch.empty_like(aligned)
                    if depth:
                        builder.decode_num_accepted_tokens = torch.empty_like(accepted)
                    if bind:
                        builder.bind_mrv2_graph_inputs(aligned, accepted)
                        self.assertIs(builder.state_indices_tensor_d, aligned)
                        if depth:
                            self.assertIs(builder.decode_num_accepted_tokens, accepted)
                        with self.assertRaises(AssertionError):
                            builder.bind_mrv2_graph_inputs(aligned[:2], accepted)
                    expected = aligned[:2].clone()
                    # Aliased graph inputs must avoid even an in-place self-copy.
                    guard = (patch.object(torch.Tensor, "copy_", side_effect=AssertionError(
                        "graph inputs were copied")) if bind else nullcontext())
                    with guard:
                        indices, counts = builder.stage(
                            SimpleNamespace(num_prefills=0, num_decodes=2, num_reqs=4),
                            aligned[:2], None if accepted is None else accepted[:2],
                            torch.arange(5, dtype=torch.int32),
                        )
                    self.assertEqual(indices.data_ptr(), builder.state_indices_tensor_d.data_ptr())
                    torch.testing.assert_close(indices[:2], expected)
                    self.assertTrue(torch.all(indices[2:] == -1))
                    if depth:
                        self.assertEqual(counts.data_ptr(), builder.decode_num_accepted_tokens.data_ptr())
                        self.assertEqual(counts.tolist(), [1, 2, 1, 1])
                    else:
                        self.assertIsNone(counts)

    def test_mamba_persistent_buffers_preserve_replayssm_initialization(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU tensor checks require torch")
        _, updated = self.apply_upstream_excerpt(
            "18-pr65182.patch",
            "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
            "mamba_replayssm_upstream.py",
            ("+        self._mamba_num_accepted_tokens_batch = (",),
        )

        class DefaultModelState:
            def __init__(self, config, model, encoder_cache, device):
                self.max_num_reqs = 4
                self.device = device

        namespace = {"torch": torch, "DefaultModelState": DefaultModelState}
        exec("from __future__ import annotations\n" + updated, namespace)
        for mode in ("align", "none"):
            for depth in (0, 2):
                for replay in (False, True):
                    with self.subTest(mode=mode, depth=depth, replay=replay):
                        config = SimpleNamespace(
                            cache_config=SimpleNamespace(
                                mamba_cache_mode=mode, use_kda_recoverssm=False,
                            ),
                            num_speculative_tokens=depth,
                            use_flashinfer_replayssm=replay,
                        )
                        state = namespace["MambaHybridModelState"](
                            config, None, None, torch.device("cpu"),
                        )
                        self.assertIsNone(state._replayssm_query_start_loc)
                        self.assertEqual(state._use_flashinfer_replayssm, replay)
                        if replay:
                            torch.testing.assert_close(
                                state._is_prefilling_gpu, torch.zeros(4, dtype=torch.bool),
                            )
                        else:
                            self.assertFalse(hasattr(state, "_is_prefilling_gpu"))
                        if mode == "align" or replay:
                            self.assertIsNone(state._mamba_ctx)
                            self.assertEqual(state._mamba_group_ids, [])
                            self.assertIsNone(state._mamba_state_copy_funcs)
                        batch = state._mamba_num_accepted_tokens_batch
                        if mode == "align" and depth:
                            torch.testing.assert_close(batch, torch.ones(4, dtype=torch.int32))
                            # Batch ordering must not overwrite per-request acceptance counts.
                            self.assertNotEqual(batch.data_ptr(), state.num_accepted_tokens_gpu.data_ptr())
                            batch.fill_(3)
                            self.assertEqual(state.num_accepted_tokens_gpu.tolist(), [1] * 4)
                        else:
                            self.assertIsNone(batch)

    def test_qsa_ring_patch_applies_after_upstream_indexer_rename(self):
        # #64110 moved the cache module and renamed both cache classes.
        # Exercise the real runtime hunks and cache-spec method without GPUs.
        path = "vllm/models/qwen4_exp/common/qsa_indexer.py"
        _, updated = self.apply_upstream_excerpt(
            "05-pr54912.patch", path, "qsa_indexer_upstream.py",
        )
        tree = ast.parse(updated)
        cache_class = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                           and n.name == "QSARawKeyCache")
        method = next(n for n in cache_class.body if isinstance(n, ast.FunctionDef)
                      and n.name == "get_kv_cache_spec")
        selected = [n for n in tree.body if (
            isinstance(n, ast.FunctionDef) and n.name == "qsa_ring_capacity"
            or isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "QSA_RING_MAX_WIDENING"
                    for t in n.targets)
        )]
        module = ast.Module(body=selected + [method], type_ignores=[])
        namespace = {
            "cdiv": lambda a, b: (a + b - 1) // b,
            "logger": SimpleNamespace(info_once=Mock()),
            "VllmConfig": object, "KVCacheSpec": object,
            "CircularBufferSpec": SimpleNamespace,
        }
        exec(compile(module, path, "exec"), namespace)
        for block_size in (848, 1616):
            cache = SimpleNamespace(
                compress_ratio=4, cache_config=SimpleNamespace(block_size=block_size),
                head_size=128, dtype="bfloat16",
            )
            for depth in range(13):
                with self.subTest(block_size=block_size, depth=depth):
                    spec = namespace["get_kv_cache_spec"](
                        cache, SimpleNamespace(num_speculative_tokens=depth),
                    )
                    minimal = 4 * ((4 + depth + 3) // 4)
                    self.assertGreaterEqual(spec.block_size, 4 + depth)
                    self.assertEqual(block_size % spec.block_size, 0)
                    self.assertLessEqual(spec.block_size, 2 * minimal)
                    self.assertEqual(spec.head_size, 128)
                    self.assertEqual(spec.dtype, "bfloat16")
                    if block_size % minimal == 0:
                        self.assertEqual(spec.block_size, minimal)
            with self.assertRaisesRegex(ValueError, "QSA ring"):
                namespace["get_kv_cache_spec"](
                    cache, SimpleNamespace(num_speculative_tokens=13),
                )
        self.assertEqual(namespace["qsa_ring_capacity"](4, 5, 848), 16)
        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Already present", result.stdout)
        self.assertFalse((self.source / path.replace("qsa_indexer", "qsa_cache")).exists())

    def test_unquantized_backend_preserves_rdna_prefill_setting(self):
        upstream, updated = self.apply_upstream_excerpt(
            "10-pr59021.patch", "vllm/config/kernel.py",
            "kernel_rdna_prefill_upstream.py",
        )
        before, after = ast.parse(upstream), ast.parse(updated)
        kernel = next(n for n in after.body if isinstance(n, ast.ClassDef))
        field_index = next(
            i for i, n in enumerate(kernel.body) if isinstance(n, ast.AnnAssign)
            and n.target.id == "unquantized_linear_backend"
        )
        self.assertEqual(kernel.body[field_index].value.value, "auto")
        self.assertIsInstance(kernel.body[field_index + 1], ast.Expr)
        del kernel.body[field_index:field_index + 2]
        normalizer = next(n for n in kernel.body if isinstance(n, ast.FunctionDef)
                          and n.name == "_normalize_linear_backend")
        validator = normalizer.decorator_list[0]
        self.assertEqual([arg.value for arg in validator.args],
                         ["linear_backend", "unquantized_linear_backend"])
        validator.args.pop()
        # Only the new field/docstring and validator registration may differ.
        self.assertEqual(ast.dump(before), ast.dump(after))

        namespace = {
            "Field": lambda **kwargs: kwargs["default"],
            "field_validator": lambda *args, **kwargs: lambda fn: fn,
        }
        exec(compile(updated, "kernel_fixture.py", "exec"), namespace)
        config = namespace["KernelConfig"]
        self.assertEqual(config._normalize_linear_backend("FLASHINFER-CUDNN"),
                         "flashinfer_cudnn")
        self.assertEqual(config.rdna_hybrid_w4a16_prefill_dequant, "disable")
        self.assertEqual(config._normalize_rdna_hybrid_w4a16_prefill_dequant("ENABLE"),
                         "enable")

    def test_flashinfer_refresh_preserves_tuning_policy_and_overrides(self):
        upstream, updated = self.apply_upstream_excerpt(
            "02-pr5943.patch", "flashinfer/autotuner/autotuner.py",
            "flashinfer_tuning_upstream.py",
            hunk_terms=("timer (str | None)", "+    timer: str | None"),
            project="flashinfer",
        )
        before, after = ast.parse(upstream), ast.parse(updated)
        # The upstream replace-based override must keep every old/new field.
        before_tuner = next(n for n in before.body if isinstance(n, ast.ClassDef)
                            and n.name == "AutoTuner")
        after_tuner = next(n for n in after.body if isinstance(n, ast.ClassDef)
                           and n.name == "AutoTuner")
        self.assertEqual(ast.dump(before_tuner), ast.dump(after_tuner))
        namespace = {}
        exec(compile(updated, "tuning_fixture.py", "exec"), namespace)
        config = namespace["TuningConfig"](
            use_cuda_graph=True, use_cold_l2_cache=True, cold_chain_launches=4,
            profile_arena_input_indices=(0,), use_cuda_graph_individual_timing=True,
            timer="cupti", first_tactic_margin=0.02,
        )
        tuner = SimpleNamespace(
            _override_tuning_buckets=None, _override_round_up=False,
            _override_cuda_graph_profile_replays=3, _override_config_cache={},
        )
        result = namespace["AutoTuner"]._apply_tuning_overrides(tuner, config)
        self.assertIsNot(result, config)
        self.assertEqual(result.cuda_graph_profile_replays, 3)
        self.assertEqual(result.timer, "cupti")
        self.assertEqual(result.first_tactic_margin, 0.02)
        self.assertTrue(result.uses_cold_chain)
        self.assertEqual(result.l2_cache_policy, "cold-chain:4")
        self.assertTrue(result.use_cuda_graph_individual_timing)

    def test_flashinfer_cupti_refresh_preserves_collective_fallback(self):
        upstream, updated = self.apply_upstream_excerpt(
            "02-pr5943.patch", "flashinfer/autotuner/autotuner.py",
            "flashinfer_cupti_upstream.py",
            hunk_terms=("requested_timer = tuning_config.timer",
                        "self._get_profiling_repeat(tuning_config)"),
            project="flashinfer",
        )
        before, after = ast.parse(upstream), ast.parse(updated)
        before_class, after_class = before.body[-1], after.body[-1]
        before_profile, before_cupti = before_class.body
        after_profile, after_cupti = after_class.body
        before_try = next(n for n in before_profile.body if isinstance(n, ast.Try))
        after_try = next(n for n in after_profile.body if isinstance(n, ast.Try))

        # Exercise the real timer-selection statements independently of CUDA.
        selection = ast.Module(body=after_try.body[:3], type_ignores=[])
        cases = [
            ("cupti", None, True),
            (None, None, False),
            ("cupti", ("auto", "events", "graph"), True),
            ("cupti", ("events", "events", "graph"), False),
            ("cupti", ("auto", "events", "eager"), False),
            (None, ("cupti", "cupti", "graph"), True),
        ]
        for timer, policy, expected in cases:
            with self.subTest(timer=timer, policy=policy):
                namespace = {
                    "tuning_config": SimpleNamespace(timer=timer),
                    "measure_policy": None if policy is None else SimpleNamespace(
                        _timer=policy[0], timer=policy[1], execution_mode=policy[2]
                    ),
                }
                exec(compile(selection, "timer_selection.py", "exec"), namespace)
                self.assertEqual(namespace["cupti_requested"], expected)

        # Only timer selection and profiling repeat may change. In particular,
        # preserve cross-rank availability/failure agreement, preparation after
        # check_ready, unconditional timing reduction, and exception propagation.
        after_try.body[:3] = before_try.body[:1]
        before_spans = next(n for n in ast.walk(before_cupti)
                            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == "_cupti_measure_spans")
        after_spans = next(n for n in ast.walk(after_cupti)
                           if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id == "_cupti_measure_spans")
        after_spans.args[1] = before_spans.args[1]
        self.assertEqual(ast.dump(before), ast.dump(after))

        sync_status = Mock(return_value=0)
        runner = Mock()
        infra_error = type("CuptiInfraError", (RuntimeError,), {})

        def measure(run_iteration, repeat, *, prologue, check_ready, prepare):
            check_ready(None)
            runner.assert_not_called()
            prepare()
            prologue()
            run_iteration()
            return [3.0, 1.0, 2.0]

        measure_spans = Mock(side_effect=measure)
        namespace = {
            "_sync_cupti_status": sync_status,
            "_CuptiInfraError": infra_error,
            "_profile_measurement_scope": nullcontext,
            "_cupti_measure_spans": measure_spans,
            "statistics": statistics, "logger": Mock(),
        }
        exec(compile(updated, "cupti_fixture.py", "exec"), namespace)
        tuner = namespace["AutoTuner"]()
        tuner.warmup, tuner.repeat = 1, 3
        tuner._get_profiling_repeat = Mock(return_value=7)
        tuner._get_input_sizes = Mock(return_value=[])
        config = SimpleNamespace(use_cold_l2_cache=False, use_cuda_graph=False)
        self.assertEqual(tuner._profile_single_kernel_cupti(runner, [], 0, config), 2.0)
        self.assertEqual(measure_spans.call_args.args[1], 7)
        self.assertEqual(runner.call_count, 2)
        sync_status.assert_called_once_with(0)

        # A peer that cannot initialize CUPTI must stop local kernel execution.
        runner.reset_mock()
        sync_status.return_value = 1
        with self.assertRaisesRegex(infra_error, "another tuning rank"):
            tuner._profile_single_kernel_cupti(runner, [], 0, config)
        runner.assert_not_called()

    def test_flashinfer_refresh_preserves_sm107_and_adds_sm12x_dispatch(self):
        _, updated = self.apply_upstream_excerpt(
            "02-pr5943.patch", "flashinfer/gemm/gemm_base.py",
            "flashinfer_mxfp8_dispatch_upstream.py",
            hunk_terms=("low_m_only``",), project="flashinfer",
        )
        sm12x_runner = object()

        def fake_import(name, *args, **kwargs):
            if name == "kernels.sm12x_mxfp8.runner":
                return SimpleNamespace(get_runner=lambda: sm12x_runner)
            if name == "cutlass":
                return SimpleNamespace()
            if name == "kernels.dense_blockscaled_gemm_sm100":
                return SimpleNamespace(Sm100BlockScaledPersistentDenseGemmKernel=object())
            return __import__(name, *args, **kwargs)

        import builtins
        namespace = {"__builtins__": {**vars(builtins), "__import__": fake_import}}
        exec(compile(updated, "gemm_fixture.py", "exec"), namespace)
        runner = namespace["_cute_dsl_gemm_mxfp8_runner"]
        self.assertIs(runner(12, 1, False, None), sm12x_runner)
        self.assertTrue(runner(10, 7, False, None, low_m_only=True))
        self.assertFalse(runner(10, 0, False, None))

    def test_flashinfer_margin_preserves_tie_confirmation_and_selection_record(self):
        _, updated = self.apply_upstream_excerpt(
            "02-pr5943.patch", "flashinfer/autotuner/autotuner.py",
            "flashinfer_selection_upstream.py",
            ("+                            margin = tuning_config.first_tactic_margin",),
            project="flashinfer",
        )
        namespace = {"functools": functools, "statistics": statistics,
                     "logger": Mock(), "_tune_process_group": None}
        exec("from __future__ import annotations\n" + updated, namespace)
        cases = [
            # Initial times, margin, tie tolerance, confirmation times, winner.
            ([3.0, 2.94], .03, 0, [3.0, 2.94], 0),
            ([3.0, 2.7], .03, 0, [3.0, 2.7], 1),
            ([float("inf"), 2.94], .03, 0, [float("inf"), 2.94], 1),
            ([3.0, 2.94], 0, 0, [3.0, 2.94], 1),
            ([3.0, 2.94], 0, .03, [3.0, 2.94], 0),
            ([3.0, 2.94], 0, .03, [4.0, 2.94], 1),
        ]
        for times, margin, tolerance, retimed, winner in cases:
            with self.subTest(times=times, margin=margin, tolerance=tolerance,
                              retimed=retimed):
                tuner = namespace["AutoTuner"]()
                tuner._tie_tolerance = tolerance
                tuner._TIE_MAX_CONTENDERS, tuner._TIE_CONFIRM_ROUNDS = 3, 2
                tuner._retime_candidate = Mock(side_effect=lambda *args: retimed[args[-1]])
                tuner._get_cache_key = Mock(return_value=SimpleNamespace(file_key="key"))
                tuner._discard_profile_cache = Mock()
                tuner._cache_profile_result = Mock()
                tuner.stats = SimpleNamespace(tuned_op_successful_configs={})
                runners = [Mock()]
                tuner.select(
                    [(t, 0, i) for i, t in enumerate(times)], min(times),
                    SimpleNamespace(first_tactic_margin=margin), runners, [],
                    SimpleNamespace(get_opt_shapes=lambda: ()), "gemm", [],
                )
                args = tuner._cache_profile_result.call_args.args
                self.assertEqual(args[3], winner)
                self.assertEqual(args[4]["time_ms"], times[winner])
                self.assertEqual(args[4]["candidates"], sum(t < float("inf") for t in times))
                if times[1 - winner] < float("inf"):
                    self.assertEqual(args[4]["runner_up_ms"], times[1 - winner])
                if tolerance:
                    self.assertGreater(tuner._retime_candidate.call_count, 0)
                else:
                    tuner._retime_candidate.assert_not_called()
                self.assertEqual(tuner.stats.tuned_op_successful_configs, {"gemm": 1})

    def test_flashinfer_sm12x_requirement_accepts_batch_invariance_argument(self):
        _, updated = self.apply_upstream_excerpt(
            "02-pr5943.patch", "flashinfer/gemm/gemm_base.py",
            "flashinfer_mxfp8_requirement_upstream.py",
            ("+@supported_compute_capability([100, 103, 107, 120, 121])",),
            project="flashinfer",
        )
        import builtins
        check = Mock(return_value=True)

        def fake_import(name, *args, **kwargs):
            if name == "kernels.sm12x_mxfp8.runner":
                return SimpleNamespace(check_requirement=check)
            return __import__(name, *args, **kwargs)

        namespace = {
            "__builtins__": {**vars(builtins), "__import__": fake_import},
            "torch": SimpleNamespace(bfloat16="bf16"),
            "supported_compute_capability": lambda caps: lambda fn: fn,
            "is_sm12x_supported": lambda device: device in (120, 121),
            "_check_cute_dsl_availability": Mock(), "_check_cute_dsl_arch": Mock(),
        }
        exec("from __future__ import annotations\n" + updated, namespace)
        requirement = namespace["_cute_dsl_gemm_mxfp8_requirement"]
        scales = SimpleNamespace(ndim=1)
        for arch in (100, 107, 120, 121):
            for invariant in (False, True):
                with self.subTest(arch=arch, invariant=invariant):
                    check.reset_mock()
                    a, b = SimpleNamespace(device=arch), object()
                    self.assertTrue(requirement(a, b, scales, scales, batch_invariant=invariant))
                    if arch in (120, 121):
                        check.assert_called_once_with(a, b, scales, scales, None, "bf16", True)
                    else:
                        check.assert_not_called()
        check.side_effect = ValueError("unsupported layout")
        a = SimpleNamespace(device=121)
        self.assertFalse(requirement(a, b, scales, scales, backend="auto", batch_invariant=True))
        with self.assertRaisesRegex(ValueError, "unsupported layout"):
            requirement(a, b, scales, scales, backend="cute-dsl", batch_invariant=True)

    def test_heap_trim_preserves_external_weight_budget_update(self):
        _, updated = self.apply_upstream_excerpt(
            "16-flash-next-fp4.patch", "vllm/v1/worker/gpu_worker.py",
            "worker_external_weights_upstream.py",
            ("+        if envs.VLLM_TRIM_HOST_HEAP_AFTER_LOAD:",),
        )
        events = []

        class Tracking:
            def __enter__(self):
                events.append("track")

            def __exit__(self, *args):
                events.append("tracked")

        namespace = {
            "envs": SimpleNamespace(VLLM_TRIM_HOST_HEAP_AFTER_LOAD=True),
            "set_current_vllm_config": lambda config: nullcontext(),
            "trim_host_heap": lambda: events.append("trim"),
            "logger": Mock(), "format_gib": str, "has_ec_transfer": lambda: True,
            "get_ec_transfer": lambda: SimpleNamespace(
                start_worker_services=lambda: events.append("services")),
            "set_torch_threads_for_runtime": lambda: events.append("threads"),
        }
        exec("from __future__ import annotations\n" + updated, namespace)
        for enabled in (False, True):
            for fallback in (0, 1024):
                with self.subTest(trim=enabled, fallback=fallback):
                    events.clear()
                    namespace["envs"].VLLM_TRIM_HOST_HEAP_AFTER_LOAD = enabled
                    worker = namespace["Worker"]()
                    worker.vllm_config = SimpleNamespace(weight_transfer_config=None)
                    worker._maybe_get_memory_pool_context = lambda **kw: nullcontext()
                    worker._scoped_allocator_max_split = lambda **kw: nullcontext()
                    worker._external_weight_charges = SimpleNamespace(
                        track=Tracking, fallback_bytes=fallback,
                    )
                    worker.model_runner = SimpleNamespace(
                        load_model=lambda **kw: events.append("load"),
                    )
                    worker._resolve_requested_memory = Mock(
                        side_effect=lambda: events.append("budget") or 42,
                    )
                    worker.load_model()
                    self.assertEqual(events, ["track", "load", "tracked"]
                                     + (["trim"] if enabled else [])
                                     + (["budget"] if fallback else [])
                                     + ["services", "threads"])
                    if fallback:
                        self.assertEqual(worker.requested_memory, 42)
                    else:
                        worker._resolve_requested_memory.assert_not_called()

    def test_streaming_update_preserves_outputs_and_invalidates_lookahead_hash(self):
        _, updated = self.apply_upstream_excerpt(
            "06-pr50897.patch", "vllm/v1/core/sched/scheduler.py",
            "streaming_session_upstream.py",
            ("+        # A lookahead hash also covers",),
        )
        statuses = SimpleNamespace(WAITING_FOR_STREAMING_REQ="streaming", WAITING="waiting")
        extend = Mock(side_effect=lambda dst, src, start, end: dst.extend(src[start:end]))
        namespace = {"RequestStatus": statuses, "extend_from_slice": extend}
        exec("from __future__ import annotations\n" + updated, namespace)
        for token_count, computed, retained_by_depth in (
            (9, 3, (2, 1, 1, 1, 1, 0)),
            (8, 3, (1, 1, 1, 1, 0, 0)),
            (1, 0, (0, 0, 0, 0, 0, 0)),
        ):
            for lookahead, retained in enumerate(retained_by_depth):
                with self.subTest(tokens=token_count, lookahead=lookahead):
                    runner = namespace["Scheduler"]()
                    runner.hash_block_size = 4
                    runner.num_hash_lookahead_tokens = int(lookahead)
                    runner.num_waiting_for_streaming_input = 1
                    runner.log_stats = False
                    tokens = list(range(token_count))
                    prompt_end = min(2, token_count - 1)
                    session = SimpleNamespace(
                        num_tokens=token_count, num_computed_tokens=computed,
                        num_prompt_tokens=prompt_end, _all_token_ids=tokens.copy(),
                        _output_token_ids=tokens[prompt_end:],
                        prompt_token_ids=tokens[:prompt_end],
                        block_hashes=["block0", "block1"],
                        status=statuses.WAITING_FOR_STREAMING_REQ,
                        update_block_hashes=Mock(),
                        get_prefix_cache_hit_limit=Mock(return_value=7),
                    )
                    update = SimpleNamespace(
                        prompt_token_ids=[20, 21], mm_features=[],
                        arrival_time=123, sampling_params=object(), max_tokens=10,
                    )
                    runner._update_request_as_session(session, update)
                    # Computation may lag behind emitted tokens; only discard
                    # the final sampled token, then append the next input.
                    self.assertEqual(session._all_token_ids, tokens[:-1] + [20, 21])
                    self.assertEqual(session.prompt_token_ids, session._all_token_ids)
                    self.assertEqual(session._output_token_ids, [])
                    self.assertEqual(session.block_hashes, ["block0", "block1"][:retained])
                    session.update_block_hashes.assert_called_once_with()
                    self.assertEqual(runner.num_waiting_for_streaming_input, 0)
                    self.assertEqual(session.status, statuses.WAITING)
                    self.assertEqual(session.prefix_cache_hit_limit, 7)
                    session.get_prefix_cache_hit_limit.assert_called_once_with()
                    extend.assert_called_with(
                        session.prompt_token_ids, session._all_token_ids,
                        prompt_end, token_count - 1,
                    )

    def test_draft_vocab_guard_preserves_parse_mtp_guard(self):
        _, updated = self.apply_upstream_excerpt(
            "03-pr59740-60226.patch", "vllm/config/vllm.py",
            "parse_mtp_guard_upstream.py",
        )
        namespace = {}
        exec(updated, namespace)
        config = namespace["VllmConfig"]()
        config.speculative_config = SimpleNamespace(
            method="mtp", is_xpress=lambda: False,
            enable_adaptive_verification=False, draft_lm_head_quantization=None,
            draft_token_map="vocab.json",
            draft_model_config=SimpleNamespace(architectures=["ParseAuxMTPModel"]),
        )
        self.assertEqual(config.unsupported_features(),
                         ["speculative draft_token_map", "Parse 3.5 MTP"])

    def test_sparse_mtp_imports_preserve_upstream_shared_expert_removal(self):
        for path, fixture in (
            ("vllm/model_executor/models/qwen3_5_mtp.py", "mtp_qwen35_imports_upstream.py"),
            ("vllm/models/qwen4_exp/amd/mtp.py", "mtp_qwen4_amd_imports_upstream.py"),
            ("vllm/models/qwen4_exp/nvidia/mtp.py", "mtp_qwen4_nvidia_imports_upstream.py"),
        ):
            with self.subTest(path=path):
                upstream, updated = self.apply_upstream_excerpt(
                    "03-pr59740-60226.patch", path, fixture,
                    ("+from vllm.model_executor.layers.draft_vocab import DraftVocab",),
                )
                tree = ast.parse(updated)
                added = [n for n in tree.body if isinstance(n, ast.ImportFrom)
                         and n.module == "vllm.model_executor.layers.draft_vocab"]
                self.assertEqual(len(added), 1)
                tree.body.remove(added[0])
                self.assertEqual(ast.dump(tree), ast.dump(ast.parse(upstream)))
                self.assertNotIn("is_model_fused_shared_expert_compatible", updated)

    def test_sparse_vocab_defers_shared_head_quantization(self):
        _, updated = self.apply_upstream_excerpt(
            "03-pr59740-60226.patch",
            "vllm/v1/worker/gpu/spec_decode/eagle/utils.py",
            "eagle_shared_head_upstream.py",
        )
        tree = ast.parse(updated)
        share = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "share_target_lm_head")
        load = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "load_eagle_model")
        share_call = next(n for n in load.body if isinstance(n, ast.Assign)
                          and isinstance(n.value, ast.Call)
                          and getattr(n.value.func, "id", None) == "share_target_lm_head")
        quantized = object()
        quantize = Mock(return_value=quantized)
        can_share = Mock(return_value=True)
        namespace = {"quantized_lm_head_copy": quantize, "_should_share": can_share}
        exec("from __future__ import annotations\n" + ast.unparse(share), namespace)
        for token_map in (None, "vocab.json"):
            for quantization in (None, "fp8", "nvfp4"):
                with self.subTest(token_map=token_map, quantization=quantization):
                    quantize.reset_mock()
                    target_head = object()
                    draft = SimpleNamespace(lm_head=object())
                    namespace.update(
                        eagle_model=draft, target_model=object(),
                        target_language_model=object(),
                        get_target_lm_head=lambda *_: target_head,
                        speculative_config=SimpleNamespace(
                            draft_token_map=token_map,
                            draft_lm_head_quantization=quantization,
                        ),
                    )
                    exec(compile(ast.Module(body=[share_call], type_ignores=[]),
                                 "share_head.py", "exec"), namespace)
                    expected = quantized if quantization and token_map is None else target_head
                    self.assertIs(draft.lm_head, expected)
                    self.assertIs(namespace["draft_head"], expected)
                    if quantization and token_map is None:
                        quantize.assert_called_once_with(target_head, quantization)
                    else:
                        quantize.assert_not_called()
        # Deferral still rejects quantization for a drafter with its own head.
        can_share.return_value = False
        for deferred in (False, True):
            with self.assertRaisesRegex(ValueError, "shares the target lm_head"):
                namespace["share_target_lm_head"](
                    SimpleNamespace(lm_head=object()), object(), "fp8",
                    defer_quantization=deferred,
                )

    def test_dynamic_nvfp4_preserves_upstream_kernel_selection_tests(self):
        upstream, updated = self.apply_upstream_excerpt(
            "07-pr54614.patch",
            "tests/kernels/quantization/test_nvfp4_kernel_selection.py",
            "nvfp4_kernel_selection_upstream.py",
        )
        before, after = ast.parse(upstream), ast.parse(updated)
        after.body = [n for n in after.body if not (
            isinstance(n, ast.FunctionDef)
            and (n.name.startswith("test_dynamic_") or n.name == "_constant_support")
            or isinstance(n, ast.ImportFrom) and n.module == "collections.abc"
        )]
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_dynamic_nvfp4_warmup_preserves_indexer_autotuning(self):
        _, updated = self.apply_upstream_excerpt(
            "07-pr54614.patch",
            "vllm/model_executor/warmup/kernel_warmup.py",
            "nvfp4_indexer_warmup_upstream.py",
        )
        module = ast.parse(updated)
        # Substitute GPU imports while executing the real warmup control flow.
        for function in module.body:
            function.body = [n for n in function.body
                             if not isinstance(n, (ast.Import, ast.ImportFrom))]
        fi_utils = SimpleNamespace(
            autotune=Mock(side_effect=lambda **kwargs: nullcontext()),
            flashinfer_get_hybrid_num_tokens_buckets=lambda n: (n,),
        )
        indexer = Mock()
        randomize = Mock(return_value="randomized hidden states")
        namespace = {
            "fi_utils": fi_utils, "autotune_indexer_wq_b": indexer,
            "torch": SimpleNamespace(rand_like=randomize), "logger": Mock(),
            "_flashinfer_deferred_moe_token_counts": lambda runner: (4,),
        }
        exec(compile(module, "warmup_fixture.py", "exec"), namespace)
        model = SimpleNamespace(compute_logits=Mock())
        config = SimpleNamespace(
            linear_backend="auto", nvfp4_dynamic_max_tokens=2,
            nvfp4_dynamic_silu_max_tokens=3,
        )
        runner = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
            vllm_config=SimpleNamespace(
                kernel_config=config,
                compilation_config=SimpleNamespace(cudagraph_capture_sizes=[0, 1, 2, 4, 16]),
            ),
            speculator=SimpleNamespace(num_query_per_req=3),
            max_num_reqs=4, max_num_tokens=16, get_model=lambda: model,
            _dummy_run=Mock(),
        )
        for dynamic in (False, True):
            for skip_attn in (False, True):
                for sample in (None, SimpleNamespace(shape=(3, 8))):
                    with self.subTest(dynamic=dynamic, skip_attn=skip_attn,
                                      hidden_states=sample is not None):
                        config.linear_backend = "nvfp4_dynamic" if dynamic else "auto"
                        runner._dummy_run.reset_mock()
                        runner._dummy_run.return_value = (None, sample)
                        indexer.reset_mock()
                        model.compute_logits.reset_mock()
                        randomize.reset_mock()
                        fi_utils.autotune.reset_mock()
                        namespace["_run_flashinfer_autotune_dummy_runs"](
                            runner, skip_attn=skip_attn,
                        )
                        counts = [16, 1, 2, 3, 4] if dynamic else [16, 4]
                        self.assertEqual(
                            [c.kwargs["num_tokens"] for c in runner._dummy_run.call_args_list],
                            counts,
                        )
                        for call in runner._dummy_run.call_args_list:
                            self.assertEqual(call.kwargs.get("skip_attn", False), skip_attn)
                        expected_buckets = [max(n, min(n, 4, 16 // 3) * 3) for n in counts]
                        self.assertEqual([c.args for c in indexer.call_args_list],
                                         [(model, n) for n in expected_buckets])
                        self.assertEqual(
                            [c.kwargs for c in fi_utils.autotune.call_args_list
                             if "round_up" in c.kwargs],
                            [{"tuning_buckets": (n,), "round_up": True}
                             for n in expected_buckets],
                        )
                        logits_runs = len(counts) if dynamic and sample is not None else 0
                        self.assertEqual(model.compute_logits.call_count, logits_runs)
                        self.assertEqual(randomize.call_count, logits_runs)
                        for call in model.compute_logits.call_args_list:
                            self.assertEqual(call.args, ("randomized hidden states",))

    def test_dynamic_nvfp4_preserves_fused_activation_for_other_kernels(self):
        _, updated = self.apply_upstream_excerpt(
            "07-pr54614.patch", "vllm/model_executor/models/qwen2_moe.py",
            "qwen2_moe_mlp_upstream.py",
        )
        fuse = Mock(return_value="quantized")
        namespace = {"nn": SimpleNamespace(Module=object), "maybe_fused_act_quant": fuse}
        exec("from __future__ import annotations\n" + updated, namespace)
        mlp = namespace["Qwen2MoeMLP"].__new__(namespace["Qwen2MoeMLP"])
        kernel = SimpleNamespace(apply_silu_or_linear=Mock(return_value="dynamic"))
        mlp.gate_up_proj = Mock(return_value=("gate_up", None))
        mlp.down_proj = Mock(return_value=("fused", None))
        mlp.down_proj.quant_method = SimpleNamespace(kernel=kernel)
        mlp.act_fn = object()
        mlp.expert_gate = None
        mlp._dynamic_nvfp4 = False
        self.assertEqual(mlp.forward("input"), "fused")
        fuse.assert_called_once_with(mlp.act_fn, "gate_up", mlp.down_proj)
        mlp.down_proj.assert_called_once_with("quantized")
        fuse.reset_mock()
        mlp.down_proj.reset_mock()
        mlp._dynamic_nvfp4 = True
        self.assertEqual(mlp.forward("input"), "dynamic")
        kernel.apply_silu_or_linear.assert_called_once_with(mlp.down_proj, "gate_up", True)
        fuse.assert_not_called()
        mlp.down_proj.assert_not_called()

    def test_resident_csf_preserves_online_shared_expert_redirection(self):
        _, updated = self.apply_upstream_excerpt(
            "16-flash-next-fp4.patch",
            "vllm/model_executor/layers/fused_moe/routed_experts.py",
            "routed_experts_loader_upstream.py",
        )
        namespace = {"_index_expert_mapping": lambda mapping: {"0": mapping},
                     "logger": Mock()}
        exec("from __future__ import annotations\n" + updated, namespace)
        layer = namespace["RoutedExperts"]()
        layer.layer_name = "model.experts"
        layer.get_expert_mapping = lambda **kw: [("scale", "0.scale", 0, "w2")]
        redirect = Mock(return_value="csf.fixed")
        layer.shared_expert_online_loader = SimpleNamespace(redirect_param_name=redirect)
        parameter = SimpleNamespace(weight_loader=Mock(return_value=True))
        layer.get_parameter = Mock(return_value=parameter)
        weight = Mock()
        weight.dim.return_value = 2
        weight.unsqueeze.return_value.unbind.return_value = (weight,)
        list(layer.load_weights([("0.scale", weight)]))
        redirect.assert_called_once_with("scale", "w2", 0)
        layer.get_parameter.assert_called_once_with("csf.fixed")
        self.assertIs(parameter.weight_loader.call_args.kwargs["param"], parameter)
        self.assertIs(parameter.weight_loader.call_args.kwargs["loaded_weight"], weight)
        layer.shared_expert_online_loader = None
        layer.scale = parameter
        layer.get_parameter.reset_mock()
        list(layer.load_weights([("0.scale", weight)]))
        layer.get_parameter.assert_not_called()

    def test_cudnn_bf16_refresh_limits_dispatch_to_sm12x_narrow_outputs(self):
        _, updated = self.apply_upstream_excerpt(
            "11-pr59025.patch", "vllm/model_executor/layers/utils.py",
            "bf16_mm_dispatch_upstream.py",
            ("+def _can_use_flashinfer_cudnn_bf16(",),
        )
        node = next(n for n in ast.parse(updated).body
                    if getattr(n, "name", None) == "_can_use_flashinfer_cudnn_bf16")
        shape_check = Mock(return_value=True)
        platform = SimpleNamespace(is_cuda=lambda: True,
                                   is_device_capability_family=lambda family: False)
        namespace = {"current_platform": platform, "_FLASHINFER_BF16_MAX_N": 128,
                     "_bf16_mm_shape_supported": shape_check}
        exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
        dispatch = namespace[node.name]
        x, weight = object(), SimpleNamespace(shape=(128, 5120))
        self.assertFalse(dispatch(x, weight, None))
        shape_check.assert_not_called()
        platform.is_device_capability_family = lambda family: family == 120
        self.assertTrue(dispatch(x, weight, None))
        weight.shape = (129, 5120)
        self.assertFalse(dispatch(x, weight, None))
        weight.shape = (128, 5120)
        shape_check.return_value = False
        self.assertFalse(dispatch(x, weight, None))

    def test_refreshed_sparse_vocab_preserves_logits_and_quantized_padding(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU tensor checks require torch")
        # This file is wholly added by patch 03; execute its production class
        # with CPU linear stand-ins instead of importing CUDA kernel packages.
        path = "vllm/model_executor/layers/draft_vocab.py"
        patch_text = (PROJECT / "docker/patches/qwen3.8/vllm/03-pr59740-60226.patch").read_text()
        section = next(s for s in re.split(r"(?m)(?=^diff --git )", patch_text)
                       if s.startswith(f"diff --git a/{path} "))
        source = "\n".join(line[1:] for line in section.splitlines()
                           if line.startswith("+") and not line.startswith("+++"))
        nodes = [n for n in ast.parse(source).body
                 if getattr(n, "name", None) in {"DraftVocab", "scatter_draft_logits",
                                                 "gather_rows_gemv", "load_draft_token_map"}]
        class Unquantized:
            pass

        class Head(torch.nn.Module):
            def __init__(self, weights, quantized=False):
                super().__init__()
                self.weight = torch.nn.Parameter(weights, requires_grad=False)
                self.org_vocab_size, self.embedding_dim = weights.shape
                self.num_embeddings_padded = self.org_vocab_size
                self.params_dtype = weights.dtype
                self.tp_size = 1
                self.quant_method = (SimpleNamespace(apply=lambda head, x: x @ head.weight.T)
                                     if quantized else Unquantized())
                self.quant_config = SimpleNamespace(get_name=lambda: "test-quant")

        class Rows:
            def __init__(self, weights, quantization):
                self.weights = weights
            def __call__(self, x):
                return x @ self.weights.T

        def listed_rows(head, weights, ids):
            selected = weights["weight"][ids]
            return Head(torch.cat((selected, torch.zeros_like(selected[:1]))), True)

        namespace = {"torch": torch, "nn": torch.nn, "_Rows": Rows,
                     "ParallelLMHead": Head, "VocabParallelEmbedding": Head,
                     "_UNQUANTIZED": (Unquantized,), "_DYNAMIC_RANK": 256, "_CHUNK": 1024,
                     "logger": Mock(), "_listed_rows": listed_rows}
        exec("from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[])), namespace)
        cls = namespace["DraftVocab"]
        gen = torch.Generator().manual_seed(4)
        weights = torch.randn(20, 8, generator=gen)
        hidden = torch.randn(4, 8, generator=gen)
        ids = torch.tensor([0, 3, 7, 13])
        processor = SimpleNamespace(vocab_size=20, head_dtype=None, scale=1.0, soft_cap=None)
        for quantized, dynamic in ((False, 0), (False, 3), (True, 0)):
            with self.subTest(quantized=quantized, dynamic_rows=dynamic):
                head = Head(weights, quantized)
                vocab = cls(processor, 20, keep_quantized_head=quantized)
                if quantized:
                    list(vocab.keep_head_weights([("lm_head.weight", weights)], head))
                vocab.load_token_map(head, ids, dynamic)
                logits = vocab.compute_logits(head, hidden)
                kept = logits.isfinite()
                self.assertTrue(torch.all(kept.sum(-1) == len(ids) + dynamic))
                torch.testing.assert_close(logits[kept], (hidden @ weights.T)[kept])
                torch.testing.assert_close(vocab.get_top_tokens(head, hidden), logits.argmax(-1))
                self.assertEqual(vocab.head_weights, {})
        head = Head(weights)
        head.tp_size = 2
        with self.assertRaisesRegex(ValueError, "tensor parallelism"):
            cls(processor, 20).load_token_map(head, ids)
        head = Head(weights, True)
        vocab = cls(processor, 20, True)
        list(vocab.keep_head_weights([("lm_head.weight", weights)], head))
        with self.assertRaisesRegex(ValueError, "draft_lm_head_quantization"):
            vocab.load_token_map(head, ids, quantization="fp8")
        vocab = cls(processor, 20)
        vocab.load_token_map = Mock()
        namespace["load_draft_token_ids"] = Mock(return_value=ids)
        config = SimpleNamespace(
            speculative_config=SimpleNamespace(draft_token_map="vocab.json",
                                               draft_token_map_dynamic_rows=3,
                                               draft_lm_head_quantization="nvfp4"),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(eos_token_id=0),
                                         try_get_generation_config=lambda: {"eos_token_id": [1, 2]}),
        )
        namespace["load_draft_token_map"](SimpleNamespace(draft_vocab=vocab, lm_head=head), config)
        namespace["load_draft_token_ids"].assert_called_once_with("vocab.json", [0, 1, 2])
        vocab.load_token_map.assert_called_once_with(head, ids, 3, "nvfp4")

    def test_confidence_stop_preserves_watermarked_and_sharded_sampling(self):
        _, updated = self.apply_upstream_excerpt(
            "04-pr60068.patch",
            "vllm/v1/worker/gpu/spec_decode/speculator.py",
            "draft_sampling_upstream.py", ("self.confidence_stop.update",),
        )
        namespace = {}
        exec("from __future__ import annotations\n" + updated, namespace)
        for route in ("watermark", "gumbel", "greedy"):
            for confidence_enabled in (False, True):
                with self.subTest(route=route, confidence=confidence_enabled):
                    events = []

                    class Logits(list):
                        def argmax(self, dim):
                            self.argmax_dim = dim
                            events.append("greedy")
                            return "greedy"

                    logits = Logits([11.0])

                    def sample(values, *args, **kwargs):
                        events.append(route)
                        values[0] = -1.0  # Sampling can change logits in place.
                        return route

                    sampler = Mock(side_effect=sample)
                    namespace["gumbel_sample"] = sampler
                    confidence = Mock(side_effect=lambda values, step: events.append(
                        ("confidence", list(values), step)
                    ))
                    speculator = namespace["DraftModelSpeculator"]()
                    speculator.use_local_argmax_reduction = False
                    speculator.confidence_stop = (
                        SimpleNamespace(update=confidence) if confidence_enabled else None
                    )
                    speculator.draft_watermarker = (
                        SimpleNamespace(sample=sampler) if route == "watermark" else None
                    )
                    speculator.compute_draft_logits = Mock(return_value=logits)
                    speculator.use_fp64_gumbel = True
                    speculator.draft_logits_num_shards = 3
                    speculator.draft_logits_shard_idx = 1
                    speculator._maybe_predict_acceptance = Mock()
                    cache = None if route == "greedy" else object()
                    result = speculator.sample_draft(
                        "hidden", "positions", "indices", "temperature", "seeds",
                        2, cache, spec_step_idx=1, num_steps=2,
                    )
                    self.assertEqual(result, route)
                    self.assertEqual(events,
                                     ([("confidence", [11.0], 2)] if confidence_enabled
                                      else []) + [route])
                    speculator.compute_draft_logits.assert_called_once_with("hidden", 1)
                    speculator._maybe_predict_acceptance.assert_called_once_with(
                        logits, "indices", 2,
                    )
                    if route == "greedy":
                        sampler.assert_not_called()
                        self.assertEqual(logits.argmax_dim, -1)
                    else:
                        kwargs = sampler.call_args.kwargs
                        self.assertIs(kwargs["logits_cache"], cache)
                        self.assertEqual(kwargs["logits_cache_col"], 2)
                        self.assertTrue(kwargs["apply_temperature"])
                        if route == "watermark":
                            self.assertEqual(kwargs["num_steps"], 2)
                        else:
                            self.assertEqual(kwargs["logits_cache_num_shards"], 3)
                            self.assertEqual(kwargs["logits_cache_shard_idx"], 1)

        speculator.use_local_argmax_reduction = True
        speculator.confidence_stop = None
        speculator.get_draft_top_tokens = Mock(return_value="local")
        speculator.compute_draft_logits.reset_mock()
        self.assertEqual(speculator.sample_draft(
            "hidden", None, None, None, None, 0, None, spec_step_idx=2,
        ), "local")
        speculator.get_draft_top_tokens.assert_called_once_with("hidden", 2)
        speculator.compute_draft_logits.assert_not_called()

    def test_confidence_stop_disables_fused_decode_after_metadata_refactor(self):
        _, updated = self.apply_upstream_excerpt(
            "04-pr60068.patch",
            "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py",
            "draft_metadata_upstream.py", ("def _configure_decode_metadata_update",),
        )

        class BaseSpeculator:
            def _configure_decode_metadata_update(self):
                self.supports_decode_metadata_update = self.backend_supports_update

        namespace = {"DraftModelSpeculator": BaseSpeculator}
        exec("from __future__ import annotations\n" + updated, namespace)
        for steps in (1, 3):
            for stop in (None, object()):
                for supported in (False, True):
                    with self.subTest(steps=steps, stop=stop is not None, backend=supported):
                        speculator = namespace["AutoRegressiveSpeculator"]()
                        speculator.num_speculative_steps = steps
                        speculator.confidence_stop = stop
                        speculator.backend_supports_update = supported
                        speculator._configure_decode_metadata_update()
                        self.assertEqual(
                            speculator.supports_decode_metadata_update,
                            steps > 1 and stop is None and supported,
                        )

    def test_confidence_stop_preserves_decode_graph_request_and_token_limits(self):
        _, updated = self.apply_upstream_excerpt(
            "04-pr60068.patch", "vllm/config/vllm.py",
            "decode_graph_limits_upstream.py",
            ("+            and speculative_config.draft_confidence_fallback_depth",),
        )
        namespace = {"bisect": bisect}
        exec("from __future__ import annotations\n" + updated, namespace)
        config = namespace["VllmConfig"]()
        config.scheduler_config = SimpleNamespace(max_num_seqs=8)
        config.num_speculative_tokens = 4
        spec = SimpleNamespace(
            uses_dynamic_speculative_decoding=lambda: False,
            draft_confidence_fallback_depth=2,
        )
        config.speculative_config = spec
        lengths = config.get_decode_cudagraph_query_lens(5)
        self.assertEqual(lengths, {2: 1, 3: 8, 4: 1, 5: 8})
        shapes = namespace["decode_cudagraph_capture_shapes"]
        self.assertEqual(shapes([1, 2, 4, 8], lengths, 12), [
            (1, 2), (1, 3), (2, 3), (4, 3), (1, 4), (1, 5), (2, 5),
        ])
        self.assertEqual(shapes([1, 2, 4, 8], lengths, 3), [(1, 2), (1, 3)])
        # The runner may sample more than one new token per request.
        self.assertEqual(config.get_decode_cudagraph_query_lens(6),
                         {3: 1, 4: 8, 5: 1, 6: 8})
        spec.draft_confidence_fallback_depth = 4
        self.assertEqual(config.get_decode_cudagraph_query_lens(5),
                         {2: 1, 3: 1, 4: 1, 5: 8})
        self.assertEqual(config.get_decode_cudagraph_query_lens(1), {1: 8})
        spec.draft_confidence_fallback_depth = None
        self.assertEqual(config.get_decode_cudagraph_query_lens(5), {5: 8})
        config.speculative_config = None
        config.num_speculative_tokens = 0
        self.assertEqual(config.get_decode_cudagraph_query_lens(1), {1: 8})
        # Upstream dynamic scheduling retains its own per-length request caps.
        config.speculative_config = spec
        config.num_speculative_tokens = 4
        spec.uses_dynamic_speculative_decoding = lambda: True
        spec.num_speculative_tokens_per_batch_size = object()
        lookup = Mock(return_value=[0, 4, 4, 2, 2, 1, 1, 0, 0])
        with patch.dict("sys.modules", {
            "vllm.v1.spec_decode.dynamic.utils": SimpleNamespace(
                build_dynamic_sd_schedule_lookup=lookup,
            ),
        }):
            self.assertEqual(config.get_decode_cudagraph_query_lens(5),
                             {5: 2, 3: 4, 2: 6, 1: 8})
        lookup.assert_called_once_with(
            spec.num_speculative_tokens_per_batch_size,
            vllm_max_batch_size=8, vllm_num_speculative_tokens=4,
        )

    def shared_autoregressive_namespace(self):
        _, updated = self.apply_upstream_excerpt(
            "04-pr60068.patch",
            "vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py",
            "draft_decode_loop_upstream.py",
        )
        # Execute real patched methods without importing CUDA-only dependencies.
        tree = ast.parse(updated)
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        namespace = {
            "DraftModelSpeculator": object, "partial": partial,
            "CUDAGraphMode": SimpleNamespace(FULL="full"),
            "dispatch_cg_and_sync_dp": Mock(),
            "build_slot_mappings_by_layer": Mock(return_value="slots"),
        }
        exec("from __future__ import annotations\n" + ast.unparse(tree), namespace)
        return namespace

    def test_shared_decode_loop_waits_after_metadata_and_stops_before_launch(self):
        namespace = self.shared_autoregressive_namespace()
        for mode in ("full", "eager"):
            for advance in (False, True):
                for skip_attn in (False, True):
                    for decisions in (None, [False], [True, False], [True] * 3):
                        with self.subTest(mode=mode, advance=advance, skip=skip_attn,
                                          decisions=decisions):
                            speculator = namespace["AutoRegressiveSpeculator"]()
                            events = []
                            speculator.input_buffers = SimpleNamespace(
                                positions=[0], query_start_loc=[0, 1],
                            )
                            speculator.idx_mapping = [0]
                            speculator.kv_cache_config = object()
                            speculator.advance_draft_positions = advance
                            speculator.block_tables = Mock()
                            speculator._build_uniform_attn_metadata = Mock(
                                side_effect=lambda **kw: events.append(("metadata", kw["step"])),
                            )
                            speculator.current_draft_step = SimpleNamespace(
                                fill_=lambda step: events.append(("step", step)),
                            )
                            speculator.decode_cudagraph_manager = SimpleNamespace(
                                run_fullgraph=lambda desc: events.append("graph"),
                            )
                            speculator._decode = lambda *args: events.append("eager")
                            stop = None
                            if decisions is not None:
                                pending = iter(decisions)

                                def should_continue():
                                    events.append("wait")
                                    return next(pending)

                                stop = SimpleNamespace(
                                    should_continue=should_continue,
                                    step_launched=lambda: events.append("launched"),
                                    end_round=lambda: events.append("end"),
                                )
                            speculator._multi_step_decode(
                                1, skip_attn,
                                SimpleNamespace(cg_mode=mode, num_tokens=1),
                                None, [4], 4, confidence_stop=stop,
                            )
                            expected = []
                            for step in range(1, 4):
                                if not skip_attn and (advance or step == 1):
                                    expected.append(("metadata", step))
                                if decisions is not None:
                                    expected.append("wait")
                                    if not decisions[step - 1]:
                                        break
                                expected.extend([("step", step),
                                                 "graph" if mode == "full" else "eager"])
                                if decisions is not None:
                                    expected.append("launched")
                            else:
                                if decisions is not None:
                                    expected.append("end")
                            self.assertEqual(events, expected)

    def test_shared_decode_dispatch_preserves_dp_specialization_and_hooks(self):
        namespace = self.shared_autoregressive_namespace()
        dispatch = namespace["dispatch_cg_and_sync_dp"]
        for fused, stop in ((True, None), (False, None), (False, object())):
            for depth in (2, 4):
                with self.subTest(fused=fused, adaptive=stop is not None, depth=depth):
                    speculator = namespace["AutoRegressiveSpeculator"]()
                    events = []
                    speculator.dp_size, speculator.dp_rank = 2, 1
                    speculator.vllm_config = SimpleNamespace(parallel_config=object())
                    sync = SimpleNamespace(num_tokens_across_dp=object())
                    dp_state = SimpleNamespace(num_reqs=1)
                    speculator._build_uniform_batch_dp_sync = Mock(return_value=(sync, 2))
                    descriptor, specialized = object(), object()
                    dispatch.reset_mock()
                    dispatch.return_value = (descriptor, sync)
                    speculator.decode_cudagraph_manager = SimpleNamespace(
                        specialize_spec_tokens=Mock(return_value=specialized),
                    )
                    speculator._prepare_decode_inputs = Mock()
                    speculator.on_multi_step_decode_begin = lambda n: events.append(("begin", n))
                    speculator.on_multi_step_decode_end = lambda n: events.append(("end", n))
                    speculator.supports_decode_metadata_update = fused
                    speculator._run_fused_multi_step_decode = Mock(
                        side_effect=lambda *args: events.append("fused"),
                    )
                    speculator._multi_step_decode = Mock(
                        side_effect=lambda *args, **kw: events.append("separate"),
                    )
                    speculator._run_multi_step_decode(
                        "batch", "rejected", 3, False, dp_state, "bounds", True,
                        depth, confidence_stop=stop,
                    )
                    self.assertEqual(events, [("begin", 1), "fused" if fused else "separate",
                                              ("end", 1)])
                    speculator._build_uniform_batch_dp_sync.assert_called_once_with(dp_state, 1)
                    dispatch.assert_called_once_with(
                        speculator.decode_cudagraph_manager, 1, 2,
                        uniform_token_count=1, dp_size=2, dp_rank=1, need_eager=True,
                        dp_sync_state=sync, parallel_config=speculator.vllm_config.parallel_config,
                    )
                    speculator.decode_cudagraph_manager.specialize_spec_tokens.assert_called_once_with(
                        descriptor, depth,
                    )
                    speculator._prepare_decode_inputs.assert_called_once_with(
                        "batch", "rejected", 1, False, specialized,
                    )
                    selected, unused = (
                        (speculator._run_fused_multi_step_decode, speculator._multi_step_decode)
                        if fused else
                        (speculator._multi_step_decode, speculator._run_fused_multi_step_decode)
                    )
                    selected.assert_called_once_with(
                        1, False, specialized, sync.num_tokens_across_dp, "bounds", depth,
                        **({} if fused else {"confidence_stop": stop}),
                    )
                    unused.assert_not_called()

    def test_target_draft_round_forwards_adaptive_and_fixed_depths_to_shared_loop(self):
        _, updated = self.apply_upstream_excerpt(
            "04-pr60068.patch",
            "vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py",
            "draft_round_upstream.py",
        )
        namespace = {"CUDAGraphMode": SimpleNamespace(FULL="full")}
        exec("from __future__ import annotations\n" + updated, namespace)
        cases = [
            # requests, grammar, dummy, profile, enabled, expected depth/stop
            (1, False, False, False, True, 4, True),
            (2, False, False, False, True, 2, False),
            (1, True, False, False, True, 4, False),
            (2, True, False, False, True, 4, False),
            (1, False, True, False, True, 4, False),
            (2, False, False, True, True, 4, False),
            (2, False, False, False, False, 4, False),
        ]
        for requests, grammar, dummy, profile, enabled, depth, adaptive in cases:
            with self.subTest(requests=requests, grammar=grammar, dummy=dummy,
                              profile=profile, enabled=enabled):
                speculator = namespace["TargetDependentARSpeculator"]()
                stop = Mock(fallback_depth=2)
                speculator.confidence_stop = stop if enabled else None
                speculator.prefill_cudagraph_manager = Mock()
                speculator.on_prefill_end = Mock()
                speculator.pcp_manager = object()
                speculator.block_tables = Mock()
                speculator.draft_tokens = [1, 2]
                speculator._run_multi_step_decode = Mock()
                batch = SimpleNamespace(
                    has_structured_output_reqs=grammar, idx_mapping_np=[7, 8][:requests],
                    idx_mapping=object(), seq_lens_cpu_upper_bound="bounds",
                )
                speculator.propose_tail(
                    batch, "rejected", requests, dummy, True, "dp", profile,
                    4, SimpleNamespace(cg_mode="full"),
                )
                speculator._run_multi_step_decode.assert_called_once_with(
                    batch, "rejected", requests, dummy, "dp", "bounds", profile,
                    depth, confidence_stop=stop if adaptive else None,
                )
                if adaptive:
                    stop.begin_round.assert_called_once_with(7)
                    stop.step_launched.assert_called_once_with()
                    stop.fixed_round.assert_not_called()
                elif enabled and not (dummy or profile):
                    stop.fixed_round.assert_called_once_with(batch.idx_mapping_np, depth)
                    stop.begin_round.assert_not_called()
                    stop.step_launched.assert_not_called()
                else:
                    self.assertEqual(stop.mock_calls, [])
                if dummy:
                    speculator.block_tables.gather_block_tables.assert_not_called()
                else:
                    speculator.block_tables.gather_block_tables.assert_called_once_with(
                        batch.idx_mapping, num_reqs_padded=requests,
                    )

    def test_resident_csf_preserves_modelopt_block_quant_dispatch(self):
        # #59604 adds IQ2/Q8 dispatch at the resident-CSF insertion point.
        _, updated = self.apply_upstream_excerpt(
            "16-flash-next-fp4.patch",
            "vllm/model_executor/layers/quantization/modelopt.py",
            "modelopt_block_quant_upstream.py", ("+            and envs.VLLM_CSF_RESIDENT",),
        )
        linear = type("LinearBase", (), {})
        embedding = type("VocabParallelEmbedding", (), {})
        head = type("ParallelLMHead", (embedding,), {})
        experts = type("RoutedExperts", (), {"moe_config": object()})
        block_linear, block_moe, resident, state, regular = (Mock() for _ in range(5))
        envs = SimpleNamespace(VLLM_CSF_RESIDENT=False)
        namespace = {
            "__name__": "series_quant.modelopt", "__package__": "series_quant",
            "Attention": type("Attention", (), {}), "LinearBase": linear,
            "VocabParallelEmbedding": embedding, "ParallelLMHead": head,
            "RoutedExperts": experts, "envs": envs,
            "B12X_BLOCK_CODECS": {"iq2_xs": (256, 74), "iq2_xxs": (256, 66),
                                  "q8_0": (32, 34)},
            "_BLOCK_FP8_MOE_ALGOS": set(), "ModelOptNvFp4FusedMoE": regular,
        }
        exec("from __future__ import annotations\n" + updated, namespace)
        config = namespace["ModelOptMixedPrecisionConfig"]()
        config.is_layer_excluded = lambda _: False
        config._resolve_weight_scale_encoding = Mock(return_value="csf")
        config._csf_state = None
        config.nvfp4_config, config.w4a16_nvfp4_config = object(), object()
        modules = {
            "series_quant.modelopt_block_quant": SimpleNamespace(
                ModelOptBlockQuantLinearMethod=block_linear,
                ModelOptBlockQuantMoEMethod=block_moe,
            ),
            "series_quant.nvfp4_csf": SimpleNamespace(
                Nvfp4CsfMoEMethod=resident, Nvfp4CsfState=state,
            ),
        }
        with patch.dict("sys.modules", modules):
            for enabled in (False, True):
                envs.VLLM_CSF_RESIDENT = enabled
                for codec in ("IQ2_XS", "IQ2_XXS", "Q8_0"):
                    config._resolve_quant_algo = lambda _: codec
                    for layer_type in (linear, embedding, head, experts):
                        with self.subTest(enabled=enabled, codec=codec, layer=layer_type):
                            layer = layer_type()
                            factory = block_moe if layer_type is experts else block_linear
                            self.assertIs(
                                config.get_quant_method(layer, "model.layer"),
                                factory.return_value,
                            )
                            args = (layer.moe_config,) if layer_type is experts else ()
                            factory.assert_called_once_with(*args, codec=codec.lower())
                            factory.reset_mock()
                    with self.assertRaisesRegex(ValueError, "requires a linear"):
                        config.get_quant_method(object(), "model.layer")
            config._resolve_weight_scale_encoding.assert_not_called()
            resident.assert_not_called()
            state.assert_not_called()

            layer = experts()
            for algo, subconfig in (("NVFP4", config.nvfp4_config),
                                    ("W4A16_NVFP4", config.w4a16_nvfp4_config)):
                config._resolve_quant_algo = lambda _: algo
                for enabled, encoding in ((False, "csf"), (True, None), (True, "csf")):
                    with self.subTest(algo=algo, enabled=enabled, encoding=encoding):
                        envs.VLLM_CSF_RESIDENT = enabled
                        config._resolve_weight_scale_encoding.return_value = encoding
                        result = config.get_quant_method(layer, "model.layer")
                        if enabled and encoding == "csf":
                            self.assertIs(result, resident.return_value)
                            resident.assert_called_once_with(
                                layer.moe_config, state.return_value,
                                use_a16=algo == "W4A16_NVFP4",
                            )
                            regular.assert_not_called()
                        else:
                            self.assertIs(result, regular.return_value)
                            regular.assert_called_once_with(
                                quant_config=subconfig, moe_config=layer.moe_config,
                            )
                            resident.assert_not_called()
                        resident.reset_mock()
                        regular.reset_mock()
            state.assert_called_once_with()

    def test_resident_csf_loader_preserves_default_decoding(self):
        _, updated = self.apply_upstream_excerpt(
            "16-flash-next-fp4.patch",
            "vllm/model_executor/model_loader/default_loader.py",
            "csf_loader_before_resident.py", ("# Apply the prefix.",),
        )
        decoded_scale, packed_weight, fixed, exceptions = (object() for _ in range(4))
        compressed = [
            ("expert.weight", packed_weight),
            ("expert.weight_scale.nvfp4_csf_fixed", fixed),
            ("expert.weight_scale.nvfp4_csf_exceptions", exceptions),
        ]
        decoded = [("expert.weight", packed_weight), ("expert.weight_scale", decoded_scale)]
        decode = Mock(return_value=iter(decoded))
        envs = SimpleNamespace(VLLM_CSF_RESIDENT=False)
        namespace = {
            "envs": envs, "decode_csf_scale_streams": decode,
            "time": SimpleNamespace(perf_counter=lambda: 123.0),
        }
        exec("from __future__ import annotations\n" + updated, namespace)
        loader = namespace["DefaultModelLoader"]()
        loader.counter_before_loading_weights = 0.0
        source = SimpleNamespace(prefix="model.")
        iterator = iter(compressed)
        self.assertEqual(
            list(loader.weights(source, iterator)),
            [("model." + name, tensor) for name, tensor in decoded],
        )
        decode.assert_called_once_with(iterator)
        self.assertEqual(loader.counter_before_loading_weights, 123.0)

        envs.VLLM_CSF_RESIDENT = True
        decode.reset_mock()
        self.assertEqual(
            list(loader.weights(source, iter(compressed))),
            [("model." + name, tensor) for name, tensor in compressed],
        )
        decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
