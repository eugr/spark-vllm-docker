#!/usr/bin/env python3
"""CPU-only lifetime checks for B12X MoE preparation and serving plans."""

import gc
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import weakref
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCH_PATH = PROJECT_DIR / "docker/patch_vllm_b12x_moe_tuning_memory.py"
SPEC = importlib.util.spec_from_file_location("moe_memory_patcher", PATCH_PATH)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# _PreparedMoECall from local-inference-lab/vllm@298f265c6, with GPU operations
# supplied by lifetime-tracking stand-ins below. No torch/vLLM install needed.
SOURCE = '''
@dataclass(frozen=True)
class _PreparedMoECall:
    state: Any
    tokens: int
    topk: int
    prepared: Any
    output_dtype: torch.dtype

    def make(self, tensors):
        from b12x.preparation import PreparedCall

        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in self.state.scratch.scratch_specs()
        )
        (
            hidden,
            activation_source,
            output,
            route_ids,
            route_weights,
            ids,
            weights,
        ) = tensors

        def reset() -> None:
            output.zero_()
            for buffer in scratch:
                buffer.zero_()

        route_patterns = tuple(route_ids.unbind(0))

        def produce(pattern: int = 0) -> None:
            hidden.copy_(activation_source)
            ids.copy_(route_patterns[pattern])
            weights.copy_(route_weights)

        def restore() -> None:
            reset()
            produce()

        binding = self.state.bind(
            scratch=scratch,
            a=hidden,
            experts=self.prepared,
            topk_weights=weights,
            topk_ids=ids,
            output=output,
            input_scales_static=True,
        )
        return PreparedCall(
            run=binding.run,
            output=output,
            produce=produce,
            reset=reset,
            restore=restore,
            capture_safe=False,
            owners=tensors,
            benchmark_producers=tuple(
                partial(produce, pattern) for pattern in range(len(route_patterns))
            ),
        )
'''


class Tensor:
    def __init__(self, value=0, shape=()):
        self.value, self.shape = value, shape

    def zero_(self):
        self.value = 0

    def copy_(self, other):
        self.value = other.value

    def __getitem__(self, index):
        # A view can retain storage without retaining the original tensor
        # object. The factory's weakref cache needs that original object.
        return Tensor(self.value[index])

    def unbind(self, dim):
        assert dim == 0
        return tuple(self[i] for i in range(self.shape[0]))


class PreparedCall(SimpleNamespace):
    owners = ()


class State:
    def __init__(self):
        self.scratch = SimpleNamespace(scratch_specs=lambda: (
            SimpleNamespace(shape=(16,), dtype="bf16", device="cuda"),
        ))

    def bind(self, **kwargs):
        # The binding owns runtime arguments only while the call exists.
        return SimpleNamespace(run=lambda: kwargs["output"])


class TrialLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.module = ModuleType("b12x.preparation")
        self.module.PreparedCall = PreparedCall
        self.modules = patch.dict(sys.modules, {"b12x.preparation": self.module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def call_type(self, source):
        namespace = {
            "dataclass": dataclass, "Any": Any, "partial": partial,
            "torch": SimpleNamespace(
                dtype=str, empty=lambda shape, **kwargs: Tensor(shape=shape),
            ),
        }
        exec(compile(source, "moe_fixture.py", "exec"), namespace)
        return namespace["_PreparedMoECall"]

    def make_call(self, source, patterns=4):
        tensors = (
            Tensor(), Tensor(42), Tensor(),
            Tensor(tuple(range(patterns)), (patterns, 8, 6)),
            Tensor(0.25), Tensor(), Tensor(),
        )
        refs = tuple(weakref.ref(tensor) for tensor in tensors)
        prepared = self.call_type(source)(State(), 8, 6, object(), "bf16")
        return prepared.make(tensors), refs

    def test_serving_plan_does_not_retain_trial_buffers(self):
        for source, leaked in ((SOURCE, True), (PATCHER.patch_source(SOURCE), False)):
            with self.subTest(leaked=leaked):
                call, refs = self.make_call(source)
                # This is the b12x _publish ownership contract: the prepared
                # serving payload keeps call.owners after call/guard teardown.
                serving_plan = SimpleNamespace(owners=tuple(call.owners))
                call.restore()
                del call
                gc.collect()
                self.assertEqual([ref() is not None for ref in refs], [leaked] * 7)
                self.assertEqual(len(serving_plan.owners), 7 if leaked else 0)

    def test_live_candidates_keep_all_weakref_cache_entries_and_route_patterns(self):
        for patterns in (1, 4):
            with self.subTest(patterns=patterns):
                source = PATCHER.patch_source(SOURCE)
                call, refs = self.make_call(source, patterns)
                gc.collect()
                self.assertTrue(all(ref() is not None for ref in refs))
                # A second candidate can reuse the original seven tensors.
                second = self.call_type(source)(State(), 8, 6, object(), "bf16")
                other_call = second.make(tuple(ref() for ref in refs))
                self.assertIs(other_call.output, call.output)
                self.assertEqual(len(call.benchmark_producers), patterns)
                for index, producer in enumerate(call.benchmark_producers):
                    producer()
                    self.assertEqual(refs[0]().value, 42)
                    self.assertEqual(refs[5]().value, index)
                    self.assertEqual(refs[6]().value, 0.25)
                del producer, call
                gc.collect()
                self.assertTrue(all(ref() is not None for ref in refs))
                other_call.restore()
                self.assertEqual(refs[5]().value, 0)
                del other_call
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))

    def test_removing_owners_alone_would_break_weakref_reuse(self):
        call, refs = self.make_call(SOURCE.replace("            owners=tensors,\n", ""))
        gc.collect()
        self.assertIsNone(refs[3]())
        self.assertIsNotNone(call.output)

    def test_patch_is_scoped_and_idempotent(self):
        unrelated = "\ndef another_call():\n    return PreparedCall(owners=tensors)\n"
        patched = PATCHER.patch_source(SOURCE + unrelated)
        self.assertTrue(patched.endswith(unrelated))
        self.assertEqual(PATCHER.patch_source(patched), patched)

    def test_older_or_absent_backend_is_unchanged(self):
        for source in ("x = 1\n", SOURCE.replace("            owners=tensors,\n", "")):
            self.assertEqual(PATCHER.patch_source(source), source)

    def test_unknown_or_partial_layout_is_rejected(self):
        for source in (
            SOURCE.replace("owners=tensors", "owners=persistent_buffers"),
            SOURCE.replace("route_ids.unbind(0)", "route_ids.split(1)"),
            PATCHER.patch_source(SOURCE).replace(
                "ids.copy_(route_ids[pattern])", "ids.copy_(other[pattern])"
            ),
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                PATCHER.patch_source(source)

    def test_source_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "vllm" / PATCHER.TARGET_REL
            target.parent.mkdir(parents=True)
            for args in ([str(root)], []):
                with self.subTest(args=args):
                    target.write_text(SOURCE)
                    result = subprocess.run(
                        [sys.executable, str(PATCH_PATH), *args],
                        cwd=root,
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(target.read_text(), PATCHER.patch_source(SOURCE))

    def test_dockerfile_patches_only_vllm_source(self):
        dockerfile = (PROJECT_DIR / "Dockerfile").read_text()
        build, runner = dockerfile.split("FROM ${CUDA_IMAGE} AS runner\n", 1)
        self.assertIn(f"RUN python3 /tmp/vllm-patches/{PATCH_PATH.name} .\n", build)
        self.assertNotIn(PATCH_PATH.name, runner)


if __name__ == "__main__":
    unittest.main()
