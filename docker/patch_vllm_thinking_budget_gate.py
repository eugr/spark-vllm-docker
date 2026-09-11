#!/usr/bin/env python3
"""Fix the V2 GPU sampler's logits-processing gate to include thinking budgets.

``Sampler._requires_logits_processing`` decides whether the per-step
logits-processing pipeline (bias, penalties, bad words, thinking-token-budget,
temperature, min_p, top_k/top_p) runs at all for the current batch. It checks
every other per-request feature (logit bias, penalties, bad words, non-default
temperature/min_p/top_k/top_p) but never checks
``self.thinking_budget_state.use_thinking_budget``.

``ThinkingBudgetState.add_request``/``apply_staged_writes``/``apply`` are all
correctly wired into ``Sampler``, and ``ReasoningConfig`` correctly derives
single-token ``<think>``/``</think>`` markers even when
``--reasoning-config`` leaves ``reasoning_start_str``/``reasoning_end_str``
empty (it falls back to the reasoning parser's own markers). None of that
matters, though: a request that sets only ``thinking_token_budget`` with
otherwise-default sampling params (e.g. ``temperature=0``, no penalties, no
logit bias) makes ``_requires_logits_processing`` return ``False``, so
``apply_sampling_params`` returns the raw logits immediately and the
thinking-budget kernel never runs. The model then spends the full
``max_tokens`` budget on reasoning, silently ignoring the budget and returning
``content: null`` with ``finish_reason: "length"``.

This is a serving-correctness bug, not a B12X-specific one, and it is not
gated behind an opt-in flag: `_requires_logits_processing` is unconditionally
wrong for any request that sets a thinking budget without also tripping one of
the other checks.

The patch is idempotent and fails on an unrecognized source shape instead of
making a best-effort rewrite.
"""

from __future__ import annotations

import sys
from pathlib import Path

TARGET_REL = Path("vllm/v1/worker/gpu/sample/sampler.py")


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(
            f"expected one {description} source anchor, found {count}; "
            "the vLLM source shape has changed"
        )
    return text.replace(old, new, 1)


_GATE_MARKER = "self.thinking_budget_state.enabled and np.any("

_ANCHOR = (
    "        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):\n"
    "            return True\n"
    "\n"
    "        states = self.sampling_states\n"
)

_REPLACEMENT = (
    "        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):\n"
    "            return True\n"
    "        if self.thinking_budget_state.enabled and np.any(\n"
    "            self.thinking_budget_state.use_thinking_budget[idx_mapping_np]\n"
    "        ):\n"
    "            return True\n"
    "\n"
    "        states = self.sampling_states\n"
)


def patch_sampler(text: str) -> str:
    if _GATE_MARKER in text:
        return text
    return replace_once(
        text, _ANCHOR, _REPLACEMENT, "_requires_logits_processing bad-words check"
    )


def main() -> int:
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    target = source_root / TARGET_REL
    if not target.exists():
        print(f"{TARGET_REL} is absent; thinking-budget gate patch is not applicable")
        return 0

    original = target.read_text()
    if "ThinkingBudgetState" not in original:
        print(
            f"{TARGET_REL} does not reference ThinkingBudgetState; "
            "thinking-budget gate patch is not applicable"
        )
        return 0

    updated = patch_sampler(original)
    compile(updated, str(target), "exec")
    if updated != original:
        target.write_text(updated)
        print(f"Applied thinking_token_budget logits-processing gate fix to {TARGET_REL}")
    else:
        print(
            "thinking_token_budget logits-processing gate fix already present "
            f"in {TARGET_REL}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PatchError as exc:
        raise SystemExit(f"thinking-budget gate patch failed: {exc}") from exc
