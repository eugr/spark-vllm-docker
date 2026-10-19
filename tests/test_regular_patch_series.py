#!/usr/bin/env python3
"""Exercise source-series application and lane guards without Docker or GPUs."""

import ast
from abc import ABC, abstractmethod
from contextlib import contextmanager
import os
from pathlib import Path
import re
import subprocess
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

    def apply_upstream_excerpt(self, patch_name, path, fixture, hunk_terms=()):
        target = self.source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        upstream = (PROJECT / "tests/fixtures/qwen38-patch-series" / fixture).read_text()
        target.write_text(upstream)
        self.commit()
        patch = (PROJECT / "docker/patches/qwen3.8/vllm" / patch_name).read_text()
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
        result = self.apply()
        self.assertEqual(result.returncode, 0, result.stderr)
        updated = target.read_text()
        compile(updated, path, "exec")
        return upstream, updated

    def test_draft_head_refresh_preserves_native_mrv2_reload(self):
        upstream, updated = self.apply_upstream_excerpt(
            "02-pr59973.patch", "vllm/v1/worker/gpu/model_runner.py",
            "mrv2_reload_upstream.py",
        )
        before = next(n for n in ast.parse(upstream).body if isinstance(n, ast.ClassDef))
        after = next(n for n in ast.parse(updated).body if isinstance(n, ast.ClassDef))
        # The native reload must survive unchanged, including all cache resets.
        reload_before = next(n for n in before.body if n.name == "reload_weights")
        reload_after = next(n for n in after.body if n.name == "reload_weights")
        self.assertEqual(ast.dump(reload_before), ast.dump(reload_after))
        events = []

        class ReloadBase:
            def reset_lora_state(self):
                events.append("lora")

        @contextmanager
        def config_context(config):
            events.append(("enter", config))
            yield
            events.append("exit")

        reload = Mock(side_effect=lambda *args: events.append(("reload", args)))
        refresh = Mock(side_effect=lambda draft: events.append(("refresh", draft)))
        after.bases = [ast.Name(id="ReloadBase", ctx=ast.Load())]
        after.body = [n for n in after.body if n.name != "get_draft_model"]
        namespace = {
            "ReloadBase": ReloadBase, "reload_weights": reload,
            "set_current_vllm_config": config_context,
            "refresh_quantized_lm_heads": refresh,
        }
        exec("from __future__ import annotations\n" + ast.unparse(after), namespace)
        runner = namespace["GPUModelRunner"]()
        runner.vllm_config = object()
        model, draft, weights = object(), object(), [("weight", object())]
        runner.get_model = lambda: model
        runner.get_draft_model = lambda: draft
        runner.reset_encoder_cache = lambda: events.append("encoder")
        runner.reset_mm_cache = lambda: events.append("mm")
        runner.reload_weights(weights, "new-checkpoint", False)
        self.assertEqual(events, [
            ("reload", (runner.vllm_config, model, weights, "new-checkpoint", False)),
            "lora", ("enter", runner.vllm_config), ("refresh", draft),
            "exit", "encoder", "mm",
        ])
        # Weight transfer reaches the same hook without invoking reload_weights.
        events.clear()
        runner.reset_lora_state()
        self.assertEqual(events, [
            "lora", ("enter", runner.vllm_config), ("refresh", draft), "exit",
        ])
        events.clear()
        runner.get_draft_model = lambda: None
        runner.reload_weights()
        self.assertEqual(events[1:], ["lora", "encoder", "mm"])
        events.clear()
        reload.side_effect = ValueError("reload failed")
        with self.assertRaisesRegex(ValueError, "reload failed"):
            runner.reload_weights()
        self.assertEqual(events, [])

    def test_lookahead_hashing_survives_kv_cache_planning_move(self):
        # #61183 moved the planning functions following the hasher to another
        # module. The hash helpers must still apply without restoring them.
        upstream, updated = self.apply_upstream_excerpt(
            "06-pr50897.patch", "vllm/v1/core/kv_cache_utils.py",
            "kv_cache_planning_move_upstream.py",
            ("+def generate_lookahead_block_hash_extra_keys(",),
        )
        helpers = {
            "generate_lookahead_block_hash_extra_keys",
            "generate_request_block_hash_extra_keys",
        }
        after = ast.parse(updated)
        after.body = [n for n in after.body if getattr(n, "name", None) not in helpers]
        self.assertEqual(ast.dump(ast.parse(upstream)), ast.dump(after))

        extra_keys = Mock(side_effect=[(("block-image",), 4), (("next-image",), 5)])
        namespace = {"generate_block_hash_extra_keys": extra_keys}
        exec("from __future__ import annotations\n" + updated, namespace)
        request = SimpleNamespace(all_token_ids=[10, 11, 12, 13, 14])
        generate = namespace["generate_request_block_hash_extra_keys"]
        self.assertEqual(
            generate(request, 0, 4, 3, True),
            ((("block-image",), 14, ("next-image",)), 4),
        )
        self.assertEqual(
            [call.args for call in extra_keys.call_args_list],
            [(request, 0, 4, 3), (request, 4, 5, 4)],
        )
        extra_keys.reset_mock(side_effect=True)
        extra_keys.return_value = (None, 4)
        self.assertEqual(generate(request, 0, 4, 3, False), (None, 4))
        extra_keys.assert_called_once_with(request, 0, 4, 3)

    def test_core_and_mooncake_retain_matching_replay_boundaries(self):
        _, core = self.apply_upstream_excerpt(
            "06-pr50897.patch", "vllm/v1/core/kv_cache_coordinator.py",
            "replay_boundaries_upstream.py", ("use_eagle=bool(self.eagle_group_ids)",),
        )
        _, mooncake = self.apply_upstream_excerpt(
            "06-pr50897.patch",
            "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py",
            "mooncake_replay_upstream.py",
            ("exclude_mamba: bool = False", "use_eagle=bool(self.eagle_group_ids)",
             "use_eagle = g_idx in self.eagle_group_ids"),
        )
        mask = Mock(return_value=None)
        namespace = {
            "cdiv": lambda n, d: (n + d - 1) // d,
            "_unwrap_spec": lambda spec: spec,
            "MambaSpec": type("MambaSpec", (), {}),
            "KVCacheSpecRegistry": SimpleNamespace(
                get_manager_class=lambda _: SimpleNamespace(reachable_block_mask=mask)
            ),
        }
        exec("from __future__ import annotations\n" + core + mooncake, namespace)
        core_coord = namespace["KVCacheCoordinator"]()
        core_coord.scheduler_block_size = 32
        core_coord.retention_interval = 0
        manager = SimpleNamespace(enable_caching=True, cache_blocks=Mock())
        core_coord.single_type_managers = [manager]
        core_coord.get_num_cacheable_tokens = lambda n, _: n
        store_coord = namespace["MooncakeStoreCoordinator"]()
        store_coord.enable_partial_hash_hits = False
        store_coord.lcm_block_size = 32
        store_coord.kv_cache_groups = [
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=8))
            for _ in range(2)
        ]
        for prompt, lookahead, eagle, expected in [
            (100, False, True, (64,)), (96, False, True, (32, 64)),
            (100, True, True, (99,)), (96, True, True, (95,)),
            (100, False, False, (99,)), (96, False, False, (95,)),
        ]:
            with self.subTest(prompt=prompt, lookahead=lookahead, eagle=eagle):
                groups = {0} if eagle else set()
                core_coord.eagle_group_ids = store_coord.eagle_group_ids = groups
                core_coord.drop_eagle_group_ids = set() if lookahead else groups
                core_coord.cache_blocks(SimpleNamespace(num_prompt_tokens=prompt), 96)
                self.assertEqual(
                    manager.cache_blocks.call_args.kwargs["replay_boundaries"], expected
                )
                mask.reset_mock()
                store_coord._reachable_masks(
                    96, 0, retention_interval=0, num_prompt_tokens=prompt,
                    apply_eagle=not lookahead,
                )
                self.assertEqual(mask.call_count, 2)
                for group, call in enumerate(mask.call_args_list):
                    self.assertEqual(call.kwargs["reachable_boundaries"], expected)
                    self.assertEqual(
                        call.kwargs["use_eagle"], eagle and not lookahead and group == 0
                    )

    def test_file_gather_implements_embedding_interface(self):
        _, updated = self.apply_upstream_excerpt(
            "01-pr59958.patch", "vllm/models/qwen4_exp/nvidia/ngram_embedding.py",
            "ple_file_gather_interface_upstream.py",
            ("class Qwen4ExpPLEFileGatherEmbedding",),
        )
        namespace = {
            "ABC": ABC, "abstractmethod": abstractmethod,
            "PLEVocabParallelEmbedding": type("PLEVocabParallelEmbedding", (), {}),
            "nn": SimpleNamespace(Module=object),
        }
        exec("from __future__ import annotations\n" + updated, namespace)
        embedding_cls = namespace["Qwen4ExpPLEFileGatherEmbedding"]
        # Allocation alone enforces ABC completeness, before tensor allocation
        # or loading weights. This is the startup failure with either loader.
        embedding = embedding_cls.__new__(embedding_cls)
        self.assertFalse(embedding.supports_prefetch)
        self.assertIsNone(embedding.start_prefetch(object()))
        with self.assertRaisesRegex(RuntimeError, "does not support prefetch"):
            embedding.finalize_prefetch(total_tokens=4)

    def test_file_gather_reuses_merged_nvfp4_lookup(self):
        # #56273 is now in main. Apply only the remaining shared-decoder
        # integration to its real device and pinned lookup implementations.
        _, updated = self.apply_upstream_excerpt(
            "01-pr59958.patch", "vllm/models/qwen4_exp/common/ngram_embedding.py",
            "nvfp4_ple_merged_upstream.py",
            ("+    def lookup_rows(",
             "-        _lookup_nvfp4_ple_embedding_kernel[(ids.numel(),)]("),
        )
        launch = Mock()
        grids = []

        class Kernel:
            def __getitem__(self, grid):
                grids.append(grid)
                return launch

        ids = Mock(device="cuda", shape=(4,), is_cuda=True)
        ids.numel.return_value = 4
        ids.reshape.return_value = ids
        output = SimpleNamespace(shape=(4, 160))
        layer = SimpleNamespace(
            weight=SimpleNamespace(shape=(8, 80)), weight_scale=Mock(),
            weight_scale_2=object(), params_dtype="bf16", embedding_dim=160,
            _uva_weight=object(), shard_indices=SimpleNamespace(
                org_vocab_start_index=8, org_vocab_end_index=16,
            ),
        )
        namespace = {
            "torch": SimpleNamespace(empty=Mock(return_value=output), uint8="uint8"),
            "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
            "_lookup_nvfp4_ple_embedding_kernel": Kernel(),
        }
        exec("from __future__ import annotations\n" + updated, namespace)
        method = namespace["Qwen4ExpPLENvFp4EmbeddingMethod"]()
        method._uva_weight_scale = Mock()
        self.assertIs(method.embedding(layer, ids), output)
        launch.assert_called_once_with(
            layer.weight, layer.weight_scale.view.return_value,
            layer.weight_scale_2, ids, output, 160, 0, 8, BLOCK_D=256,
        )
        layer.weight_scale.view.assert_called_once_with("uint8")

        launch.reset_mock()
        method.lookup_from_pinned(layer, ids, output)
        launch.assert_called_once_with(
            layer._uva_weight, method._uva_weight_scale.view.return_value,
            layer.weight_scale_2, ids, output, 160, 8, 16, BLOCK_D=256,
        )
        method._uva_weight_scale.view.assert_called_once_with("uint8")

        launch.reset_mock()
        staged_weight, staged_scale = object(), Mock()
        method.lookup_rows(staged_weight, staged_scale, layer.weight_scale_2,
                           ids, output, 0, 4)
        launch.assert_called_once_with(
            staged_weight, staged_scale.view.return_value,
            layer.weight_scale_2, ids, output, 160, 0, 4, BLOCK_D=256,
        )
        self.assertEqual(grids, [(4,), (4,), (4,)])
        launch.reset_mock()
        ids.numel.return_value = 0
        method.lookup_rows(staged_weight, staged_scale, layer.weight_scale_2,
                           ids, output, 0, 0)
        launch.assert_not_called()

    def test_ple_sequence_parallel_preserves_full_token_rows(self):
        _, updated = self.apply_upstream_excerpt(
            "01-pr59958.patch", "vllm/models/qwen4_exp/nvidia/ngram_embedding.py",
            "ple_sequence_parallel_upstream.py",
            ("self.use_reduce_scatter =", "embedding.finalize_prefetch("),
        )

        class Rows(list):
            dtype = "bf16"

            @property
            def shape(self):
                return (len(self),)

            def flatten(self, _):
                return self

            def view(self, _):
                return self

        class DeviceEmbedding:
            supports_prefetch = False

            def __init__(self, *args, reduce_results, **kwargs):
                self.reduce_results = reduce_results

            def __call__(self, ids):
                return Rows([2 if self.reduce_results else 1] * len(ids))

        class PinnedEmbedding(DeviceEmbedding):
            supports_prefetch = True

            def finalize_prefetch(self, *, total_tokens):
                return super().__call__(range(total_tokens))

        class FileEmbedding(DeviceEmbedding):
            def __call__(self, tokens):
                # File gather already has every rank's vocabulary rows.
                return Rows([2] * len(tokens))

        namespace = {
            "Qwen4ExpPLEDeviceEmbedding": DeviceEmbedding,
            "Qwen4ExpPLEPinnedHostEmbedding": PinnedEmbedding,
            "Qwen4ExpPLEFileGatherEmbedding": FileEmbedding,
            "is_fp8": lambda _: False,
            "get_tp_group": lambda: SimpleNamespace(world_size=2),
            "sp_shard": lambda rows: Rows(rows[:len(rows) // 2]),
            "sp_reduce_scatter": lambda rows: Rows(
                [2 * value for value in rows[:len(rows) // 2]]
            ),
        }
        exec("from __future__ import annotations\n" + updated, namespace)
        for storage in ("device", "pinned", "file"):
            for sequence_parallel in (False, True):
                for etp_size in (1, 2):
                    with self.subTest(storage=storage, sp=sequence_parallel, etp=etp_size):
                        config = SimpleNamespace(
                            host_file_gather=storage == "file",
                            cpu_offload=storage == "pinned",
                        )
                        namespace["get_current_vllm_config"] = lambda: SimpleNamespace(
                            engram_config=config
                        )
                        namespace["get_etp_group"] = lambda: SimpleNamespace(
                            world_size=etp_size
                        )
                        module = namespace["Qwen4ExpNGramEmbedding"]()
                        module.use_sequence_parallel = sequence_parallel
                        module.head_dim = module.ngram_heads = 1
                        module.configure_embedding("bf16", 8, 1, "ple", None, 4, 0)
                        module.compute_ngram_ids = lambda ids, *_: ids
                        hidden = Rows([0] * (2 if sequence_parallel else 4))
                        result = module.forward(hidden, Rows(range(4)), None, None)
                        self.assertEqual(result, [2] * len(hidden))

    def test_confidence_stop_disables_fused_decode_after_metadata_refactor(self):
        _, updated = self.apply_upstream_excerpt(
            "04-pr60068.patch",
            "vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py",
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
                        speculator = namespace["TargetDependentARSpeculator"]()
                        speculator.num_speculative_steps = steps
                        speculator.confidence_stop = stop
                        speculator.backend_supports_update = supported
                        speculator._configure_decode_metadata_update()
                        self.assertEqual(
                            speculator.supports_decode_metadata_update,
                            steps > 1 and stop is None and supported,
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
