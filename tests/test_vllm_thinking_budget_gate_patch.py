#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCHER_PATH = PROJECT_DIR / "docker/patch_vllm_thinking_budget_gate.py"
SPEC = importlib.util.spec_from_file_location("thinking_budget_gate_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


CURRENT_GATE = """    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        if np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]):
            return True
        if np.any(self.penalties_state.use_penalty[idx_mapping_np]):
            return True
        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return True

        states = self.sampling_states
        temperatures = states.temperature.np[idx_mapping_np]
        if np.any((temperatures != 0.0) & (temperatures != 1.0)):
            return True
        if np.any(states.min_p.np[idx_mapping_np] != 0.0):
            return True
        if np.any(states.top_k.np[idx_mapping_np] != states.vocab_size):
            return True
        return bool(np.any(states.top_p.np[idx_mapping_np] != 1.0))
"""


class ThinkingBudgetGatePatchTests(unittest.TestCase):
    def test_gate_check_is_added_and_idempotent(self):
        patched = PATCHER.patch_sampler(CURRENT_GATE)

        self.assertIn("self.thinking_budget_state.enabled and np.any(", patched)
        self.assertIn(
            "self.thinking_budget_state.use_thinking_budget[idx_mapping_np]",
            patched,
        )
        # The other checks and their order are preserved.
        self.assertIn("if np.any(self.logit_bias_state.use_logit_bias", patched)
        self.assertIn("states = self.sampling_states", patched)
        self.assertLess(
            patched.index("bad_words_state.num_bad_words"),
            patched.index("thinking_budget_state.enabled"),
        )
        self.assertLess(
            patched.index("thinking_budget_state.enabled"),
            patched.index("states = self.sampling_states"),
        )

        self.assertEqual(PATCHER.patch_sampler(patched), patched)

    def test_unknown_source_shape_fails(self):
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_sampler("class Sampler:\n    pass\n")


if __name__ == "__main__":
    unittest.main()
