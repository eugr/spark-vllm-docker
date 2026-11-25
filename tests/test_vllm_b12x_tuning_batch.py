#!/usr/bin/env python3
"""Exercise the source patch against the fork's session factory without GPUs."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "tuning_batch_patch", ROOT / "docker/patch_vllm_b12x_tuning_batch.py"
)
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)
SOURCE = (ROOT / "tests/fixtures/b12x_get_session.py").read_text()


class TuningBatchTests(unittest.TestCase):
    def test_limits_live_candidates_and_preserves_distributed_autotuning(self):
        created = []

        def create(**kwargs):
            session = SimpleNamespace(
                state="OPEN", autotune=kwargs["autotune"],
                configure_tuning_shard=Mock(), kwargs=kwargs,
            )
            created.append(session)
            return session

        modules = {
            "b12x.preparation": SimpleNamespace(PreparationSession=create),
            "vllm.distributed.parallel_state": SimpleNamespace(
                get_tp_group=lambda: SimpleNamespace(ranks=[1, 0])
            ),
        }
        config = SimpleNamespace(
            model_config=SimpleNamespace(model="multimodal-model", dtype="bf16"),
            cache_config=SimpleNamespace(cache_dtype="fp8"),
            parallel_config=SimpleNamespace(tensor_parallel_size=2,
                                            pipeline_parallel_size=1),
            kernel_config=SimpleNamespace(enable_b12x_autotune=True),
            speculative_config=SimpleNamespace(model="draft", method="mtp"),
        )
        worker = SimpleNamespace(device=object(), rank=1, vllm_config=config)
        namespace = {"os": os}
        exec("from __future__ import annotations\n" + patcher.patch_source(SOURCE),
             namespace)
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {"B12X_AUTOTUNE": "1"}):
            session = namespace["get_b12x_session"](worker)
            self.assertEqual(session.kwargs["race_batch"], 1)
            self.assertTrue(session.autotune)
            self.assertIs(session.kwargs["device"], worker.device)
            self.assertEqual(session.kwargs["namespace"]["model"], "multimodal-model")
            self.assertEqual(session.kwargs["namespace"]["draft_model"], "draft")
            self.assertEqual(session.kwargs["namespace"]["tensor_parallel"], 2)
            session.configure_tuning_shard.assert_called_once_with(1, (0, 1))
            self.assertIs(namespace["get_b12x_session"](worker), session)
            self.assertEqual(len(created), 1)
            session.state = "CLOSED"
            self.assertIsNot(namespace["get_b12x_session"](worker), session)

    def test_repeat_application_is_unchanged(self):
        updated = patcher.patch_source(SOURCE)
        self.assertEqual(patcher.patch_source(updated), updated)

    def test_upstream_memory_policy_requires_review(self):
        changed = SOURCE.replace("device=worker.device,", "device=worker.device,\n        race_batch=4,")
        with self.assertRaisesRegex(ValueError, "review its memory policy"):
            patcher.patch_source(changed)

    def test_unknown_factory_does_not_silently_skip(self):
        with self.assertRaisesRegex(ValueError, "Expected one PreparationSession"):
            patcher.patch_source(SOURCE.replace("PreparationSession(", "create_session("))


if __name__ == "__main__":
    unittest.main()
