#!/usr/bin/env python3
"""glm53-port-fork862-prefill-decode-latency: ports the contended-prefill
bound from the fork (PR #862, commit 1fbc36ab7, author yatesdr) onto ab86b7073,
with the new branch OFF by default (env VLLM_GLM53_PREFILL_DECODE_LATENCY=1).

Attribution: repo local-inference-lab/vllm, PR #862
"fix(scheduler): bound contended prefill compute steps",
commit 1fbc36ab798d2777c02c5531a0dbdf6a40e0eef7, Apache-2.0 license
(the vLLM repo license). No knapcio code.

What changes (3 prod files; the PR tests are not ported):
- config/scheduler.py: max_num_prefill_tokens_per_step knob (default 0) +
  validation (requires prefill_compute_share; <= max_num_batched_tokens).
- engine/arg_utils.py: EngineArgs field + CLI flag
  --max-num-prefill-tokens-per-step + pass-through to SchedulerConfig.
- v1/core/sched/scheduler.py: self.prefill_fairness_max_tokens (None unless
  knob>0 AND env=1) + token_budget/input_budget bounding at 3 sites +
  metric. [GLM53-PORT-FORK862-...] log ONCE when actually bounding.

Without env: attr None -> all 3 `is not None` checks are false -> identical
to ab86. Idempotent; --check validates anchors without writing.
"""
from __future__ import annotations
import argparse, ast, sys
from pathlib import Path

MOD = "glm53-port-fork862-prefill-decode-latency"
ENV = "VLLM_GLM53_PREFILL_DECODE_LATENCY"

HELPER = f'''
# --- {MOD} (port PR #862) -----------------------------------------------------
# Gate: the bound only exists when {ENV}=1 (on top of the knob).
import os as _g862_os

_G862_ENV = "{ENV}"
_G862_BOUND_LOGGED = False


def _g862_env_on() -> bool:
    return _g862_os.getenv(_G862_ENV, "0") == "1"
'''

C_FIELD = '''    max_num_prefill_tokens_per_step: int = Field(default=0, ge=0)
    """Maximum local prefill tokens in a contended compute-share step.

    Zero keeps the normal batched-token budget. Decode-only and uncontended
    prefill steps are unchanged.
    """

'''

C_VALID = '''        if (
            self.max_num_prefill_tokens_per_step > 0
            and self.prefill_compute_share is None
        ):
            raise ValueError(
                "max_num_prefill_tokens_per_step requires prefill_compute_share"
            )
        if self.max_num_prefill_tokens_per_step > self.max_num_batched_tokens:
            raise ValueError(
                "max_num_prefill_tokens_per_step cannot exceed "
                "max_num_batched_tokens"
            )
'''

E_FIELD = '''    max_num_prefill_tokens_per_step: int = (
        SchedulerConfig.max_num_prefill_tokens_per_step
    )
'''

E_CLI = '''        scheduler_group.add_argument(
            "--max-num-prefill-tokens-per-step",
            **scheduler_kwargs["max_num_prefill_tokens_per_step"],
        )
'''

E_BUILD = '''            max_num_prefill_tokens_per_step=(
                self.max_num_prefill_tokens_per_step
            ),
'''

S_INIT_NEW = (
    "        self.prefill_fairness_max_tokens = (\n"
    "            self.scheduler_config.max_num_prefill_tokens_per_step or None\n"
    "        ) if _g862_env_on() else None\n"
)

S_B2_NEW = '''        if (
            selected_compute_class == "prefill"
            and compute_contention
            and self.prefill_fairness_max_tokens is not None
        ):
            global _G862_BOUND_LOGGED
            if not _G862_BOUND_LOGGED:
                _G862_BOUND_LOGGED = True
                logger.info(
                    "[GLM53-PORT-FORK862-PREFILL-DECODE-LATENCY] bounding contended "
                    "prefill to %s tokens/step", self.prefill_fairness_max_tokens)
            token_budget = min(token_budget, self.prefill_fairness_max_tokens)
            input_budget = min(
                input_budget,
                self.prefill_fairness_max_tokens + draft_slots,
            )

'''

S_B34_NEW = '''                            if self.prefill_fairness_max_tokens is not None:
                                token_budget = min(
                                    token_budget, self.prefill_fairness_max_tokens
                                )
                                input_budget = min(
                                    input_budget,
                                    self.prefill_fairness_max_tokens + draft_slots,
                                )
'''

S_B4_NEW = '''            if self.prefill_fairness_max_tokens is not None:
                token_budget = min(token_budget, self.prefill_fairness_max_tokens)
                input_budget = min(
                    input_budget,
                    self.prefill_fairness_max_tokens + draft_slots,
                )
'''

S_METRICS_NEW = '''            "max_num_prefill_tokens_per_step": (
                self.scheduler_config.max_num_prefill_tokens_per_step
            ),
'''

EDITS = {
 "config/scheduler.py": [
  ("    max_parallel_prefills: MaxParallelPrefills = 1\n",
   C_FIELD + "    max_parallel_prefills: MaxParallelPrefills = 1\n"),
  ('                "prefill_compute_share cannot be combined with "\n'
   '                "prefill_schedule_interval greater than one"\n'
   "            )\n",
   '                "prefill_compute_share cannot be combined with "\n'
   '                "prefill_schedule_interval greater than one"\n'
   "            )\n" + C_VALID),
 ],
 "engine/arg_utils.py": [
  ("    prefill_compute_half_life: PrefillComputeHalfLife | None = (\n"
   "        SchedulerConfig.prefill_compute_half_life\n"
   "    )\n",
   "    prefill_compute_half_life: PrefillComputeHalfLife | None = (\n"
   "        SchedulerConfig.prefill_compute_half_life\n"
   "    )\n" + E_FIELD),
  ('        scheduler_group.add_argument(\n'
   '            "--prefill-compute-half-life", **prefill_compute_half_life_kwargs\n'
   "        )\n",
   '        scheduler_group.add_argument(\n'
   '            "--prefill-compute-half-life", **prefill_compute_half_life_kwargs\n'
   "        )\n" + E_CLI),
  ("            prefill_compute_half_life=self.prefill_compute_half_life,\n",
   "            prefill_compute_half_life=self.prefill_compute_half_life,\n" + E_BUILD),
 ],
 "v1/core/sched/scheduler.py": [
  ("            if prefill_compute_share is not None\n"
   "            else None\n"
   "        )\n"
   "        self._decode_compute_seconds = 0.0\n",
   "            if prefill_compute_share is not None\n"
   "            else None\n"
   "        )\n" + S_INIT_NEW +
   "        self._decode_compute_seconds = 0.0\n"),
  ("            compute_contention_started = compute_contention and not prior_contention\n",
   "            compute_contention_started = compute_contention and not prior_contention\n"
   + "\n" + S_B2_NEW.rstrip("\n") + "\n"),
  ("                            # The selected decode class proved unrunnable after\n"
   "                            # its resource checks. Fall back immediately.\n"
   "                            adaptive_defer_prefills = False\n"
   "                            defer_prefills = legacy_defer_prefills\n",
   "                            # The selected decode class proved unrunnable after\n"
   "                            # its resource checks. Fall back immediately.\n"
   "                            adaptive_defer_prefills = False\n"
   "                            defer_prefills = legacy_defer_prefills\n"
   + S_B34_NEW),
  ("            adaptive_defer_prefills = False\n"
   "            defer_prefills = False\n"
   '            schedule_running_requests("prefill")\n',
   "            adaptive_defer_prefills = False\n"
   "            defer_prefills = False\n"
   + S_B4_NEW +
   '            schedule_running_requests("prefill")\n'),
  ('            "prefill_compute_half_life": (\n'
   "                self.scheduler_config.prefill_compute_half_life\n"
   "            ),\n",
   '            "prefill_compute_half_life": (\n'
   "                self.scheduler_config.prefill_compute_half_life\n"
   "            ),\n" + S_METRICS_NEW),
 ],
}


class PatchError(RuntimeError):
    pass


def patch_file(rel: str, src: str) -> str:
    for old, new in EDITS[rel]:
        if new in src:
            continue  # hunk already applied (per-hunk idempotent)
        n = src.count(old)
        if n != 1:
            raise PatchError(f"ANCHOR FAILED: {rel} :: {old.splitlines()[0][:60]!r} x{n}")
        src = src.replace(old, new, 1)
    if rel == "v1/core/sched/scheduler.py":
        if "\nlogger = " not in src:
            raise PatchError("ANCHOR FAILED: scheduler.py has no module logger")
        if "def _g862_env_on" not in src:
            src = src + HELPER
    ast.parse(src)
    return src


def _verify_full(root: Path) -> None:
    marks = [
        ("config/scheduler.py", "max_num_prefill_tokens_per_step"),
        ("engine/arg_utils.py", "--max-num-prefill-tokens-per-step"),
        ("v1/core/sched/scheduler.py", "prefill_fairness_max_tokens"),
        ("v1/core/sched/scheduler.py", "def _g862_env_on"),
    ]
    for rel, m in marks:
        if m not in (root / rel).read_text():
            raise PatchError(f"partial patch: missing {m} in {rel}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path(a.vllm_root)
    changed = []
    for rel in EDITS:
        t = root / rel
        src = t.read_text()
        out = patch_file(rel, src)
        if out != src:
            if not a.check:
                t.write_text(out)
            changed.append(rel)
    if a.check:
        print(f"[{MOD}] check: {'already applied' if not changed else 'anchors OK: ' + ','.join(changed)} (no writes)")
        return 0
    if not changed:
        _verify_full(root)
        print(f"[{MOD}] already applied (no-op)")
        return 0
    for rel in changed:
        ast.parse((root / rel).read_text())
    _verify_full(root)
    print(f"[{MOD}] applied: {','.join(changed)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PatchError, SyntaxError) as e:
        print(f"[{MOD}] FAILED: {e}", file=sys.stderr)
        raise SystemExit(1)
