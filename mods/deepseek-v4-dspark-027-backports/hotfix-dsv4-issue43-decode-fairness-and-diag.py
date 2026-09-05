#!/usr/bin/env python3
"""Hotfix (issue #43): bounded decode service during mixed prefill steps +
per-step scheduler diagnostics. Layers on top of the issue #27 hotfix
(``hotfix-dsv4-issue27-partial-prefill-concurrency.py``), which restored the
v1 scheduler's missing ``max_num_partial_prefills`` admission gate.

Why this exists (issue #43 follow-up)
-------------------------------------
The #27 fix caps concurrent in-flight prefills to ``max_num_partial_prefills``
(default 1) and ``--long-prefill-token-threshold`` caps each prefill chunk. It
cured the *per-step* decode starvation (worst ITL 2790 ms -> 67 ms) but the
reporter's six-cell cold retest still shows a wide *whole-request* decode-rate
spread (min/max 0.107-0.238). Issue #43 asks for:

  1. scheduled prefill/decode tokens per step and per request;
  2. zero-token decode skips by request and running-list position;
  3. bounded service for every decode-active request during mixed steps;
  4. per-lane p95 ITL and whole decode-window fairness (separate harness).

This hotfix delivers (1)-(3) inside the scheduler. (4) lives in
``scripts/reproduce-issue43-live.py``.)

What it changes (all in ``Scheduler.schedule``)
-----------------------------------------------
A. Bounded decode service (ask #3) -- the actual *fix*. While iterating the
   RUNNING list, when the current request is a prefill chunk, reserve one
   decode step's worth of token budget for every still-unvis ited,
   decode-active running request behind it, and cap the prefill chunk so that
   reservation is never violated. If the remaining budget can't satisfy both
   the prefill chunk and the decode reservation, the prefill chunk is dropped
   to 0 tokens and skipped (``continue``), letting the decode lanes run.
   This generalizes the #27 cap beyond ``max_num_partial_prefills=1``: it is
   a per-step, per-lane service floor that holds for any
   ``--long-prefill-token-threshold`` tuning, so mis-tuning the chunk cap can
   no longer resurrect #27-style decode skips. It is a no-op under the
   current default knob set (threshold=1024, budget=8192, cap=1), where the
   prefill chunk is already far below the decode reservation.

B. Zero-token decode-skip recording (ask #2). At the ``num_new_tokens == 0``
   skip branch, if the skipped request is decode-active (past its prompt and
   not at its async max-tokens sentinel), record
   ``(request_id, running_list_position, num_computed_tokens)``.

C. Per-request scheduled-token recording (ask #1). For every scheduled
   running request, record its scheduled token count keyed by request_id in
   ``self.issue43_last_step_diag["prefill"|"decode"]``. Decode-active requests
   land in ``"decode"``; prefill chunks land in ``"prefill"``.

D. Step summary log (ask #1). When ``DSPARK_ISSUE43_SCHED_DIAG=1`` is set in
   the container env, emit one compact line per scheduler step:
   ``[issue43-step N] running=K prefill_toks={..} decode_toks={..}
   decode_skips=[(rid,pos,computed)..]``. Off by default (zero overhead: the
   diag dict is cheap to build and only the log line is gated).

Idempotent. Anchors are asserted; re-applying is a no-op once all marks are
present. Safe to apply before or after the issue #27 hotfix (independent
regions).

Patches /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
in-place inside the container (called from the compose entrypoint before
``exec vllm serve``).
"""
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py")
src = P.read_text()

MARK = "# [issue43-hotfix]"
if MARK in src:
    print(f"[issue43-hotfix] already applied to {P}")
    raise SystemExit(0)

# --- 0. import os (module-level diag gate) -----------------------------------
A0_OLD = "import itertools\nimport time\n"
assert A0_OLD in src, "issue43: import anchor not found; refusing to patch"
src = src.replace(
    A0_OLD,
    A0_OLD
    + "# [issue43-hotfix] os import for DSPARK_ISSUE43_SCHED_DIAG gate\n"
    + "import os\n",
    1,
)

# --- 1. module-level diag gate constant --------------------------------------
A1_OLD = "logger = init_logger(__name__)\n"
assert A1_OLD in src, "issue43: logger anchor not found; refusing to patch"
A1_NEW = (
    A1_OLD
    + "# [issue43-hotfix] per-step scheduler diagnostics gate (issue #43).\n"
    + "# Set DSPARK_ISSUE43_SCHED_DIAG=1 in the container env to emit one\n"
    + "# compact scheduled-tokens / decode-skip summary line per step.\n"
    + "_ISSUE43_SCHED_DIAG = os.environ.get(\"DSPARK_ISSUE43_SCHED_DIAG\", \"0\") not in (\"0\", \"\", \"false\", \"False\")\n"
)
src = src.replace(A1_OLD, A1_NEW, 1)

# --- 2. init diag dict at the top of schedule(), after prefill_scheduled -----
A2_OLD = "        prefill_scheduled = False\n"
assert A2_OLD in src, "issue43: prefill_scheduled anchor not found; refusing to patch"
A2_NEW = (
    A2_OLD
    + "        # [issue43-hotfix] per-step scheduler diagnostics (issue #43).\n"
    + "        # Tracks per-request scheduled prefill/decode token counts and\n"
    + "        # zero-token decode skips (by request_id and running-list pos).\n"
    + "        # Always built (cheap); only the step log line (below) is gated.\n"
    + "        issue43_step_diag = {\"prefill\": {}, \"decode\": {}, \"skips\": []}\n"
)
src = src.replace(A2_OLD, A2_NEW, 1)

# --- 3. decode-floor reservation, inserted after the mamba split block and
#        before the num_new_tokens == 0 skip check ----------------------------
A3_OLD = (
    "            if self.need_mamba_block_aligned_split:\n"
    "                num_new_tokens = self._mamba_block_aligned_split(\n"
    "                    request, num_new_tokens\n"
    "                )\n"
    "\n"
    "            if num_new_tokens == 0:\n"
)
assert A3_OLD in src, "issue43: mamba-split/zero-check anchor not found; refusing to patch"
A3_NEW = (
    "            if self.need_mamba_block_aligned_split:\n"
    "                num_new_tokens = self._mamba_block_aligned_split(\n"
    "                    request, num_new_tokens\n"
    "                )\n"
    "\n"
    "            # [issue43-hotfix] bounded decode service during mixed prefill\n"
    "            # steps (issue #43 ask #3). Generalizes the #27\n"
    "            # max_num_partial_prefills cap: regardless of the configured\n"
    "            # --long-prefill-token-threshold, never let a prefill chunk\n"
    "            # consume so much remaining token budget that a decode-active\n"
    "            # request later in self.running is forced to num_new_tokens==0\n"
    "            # and skipped. Reserve >=1 decode step of tokens for every\n"
    "            # not-yet-visited decode-active lane; if the reservation can't\n"
    "            # be met alongside the prefill chunk, drop the chunk to 0 so the\n"
    "            # zero-check below skips it (continue) and the decodes run.\n"
    "            if getattr(request, \"is_prefill_chunk\", False):\n"
    "                _dec_floor = 0\n"
    "                for _ri in range(req_index + 1, len(self.running)):\n"
    "                    _r = self.running[_ri]\n"
    "                    if (_r.num_output_placeholders > 0 and\n"
    "                            _r.num_computed_tokens + 2\n"
    "                            - _r.num_output_placeholders\n"
    "                            >= _r.num_prompt_tokens + _r.max_tokens):\n"
    "                        continue\n"
    "                    if self.current_step < _r.next_decode_eligible_step:\n"
    "                        continue\n"
    "                    if defer_prefills and getattr(_r, \"is_prefill_chunk\", False):\n"
    "                        continue\n"
    "                    if _r.num_computed_tokens >= _r.num_prompt_tokens:\n"
    "                        _dec_floor += self.num_sampled_tokens_per_step\n"
    "                if _dec_floor > 0:\n"
    "                    num_new_tokens = min(\n"
    "                        num_new_tokens,\n"
    "                        max(0, token_budget - _dec_floor))\n"
    "\n"
    "            if num_new_tokens == 0:\n"
)
src = src.replace(A3_OLD, A3_NEW, 1)

# --- 4. record zero-token decode skips inside the skip branch ----------------
A4_OLD = (
    "                # NOTE(woosuk): Here, by doing `continue` instead of `break`,\n"
    "                # we do not strictly follow the FCFS scheduling policy and\n"
    "                # allow the lower-priority requests to be scheduled.\n"
    "                req_index += 1\n"
    "                continue\n"
)
assert A4_OLD in src, "issue43: woosuk-skip anchor not found; refusing to patch"
A4_NEW = (
    "                # NOTE(woosuk): Here, by doing `continue` instead of `break`,\n"
    "                # we do not strictly follow the FCFS scheduling policy and\n"
    "                # allow the lower-priority requests to be scheduled.\n"
    "                # [issue43-hotfix] record zero-token decode skips (issue #43\n"
    "                # ask #2): a request past its prompt with no pending async\n"
    "                # max-tokens sentinel is decode-active and got skipped here.\n"
    "                if (request.num_computed_tokens >= request.num_prompt_tokens\n"
    "                        and request.num_output_placeholders == 0):\n"
    "                    issue43_step_diag[\"skips\"].append(\n"
    "                        (request.request_id, req_index,\n"
    "                         request.num_computed_tokens))\n"
    "                req_index += 1\n"
    "                continue\n"
)
src = src.replace(A4_OLD, A4_NEW, 1)

# --- 5. record per-request scheduled tokens in the running branch -----------
A5_OLD = (
    "            scheduled_running_reqs.append(request)\n"
    "            prefill_scheduled |= request.is_prefill_chunk\n"
    "            request_id = request.request_id\n"
    "            req_to_new_blocks[request_id] = new_blocks\n"
    "            num_scheduled_tokens[request_id] = num_new_tokens\n"
    "            token_budget -= num_new_tokens\n"
    "            req_index += 1\n"
)
assert A5_OLD in src, "issue43: running-schedule anchor not found; refusing to patch"
A5_NEW = (
    "            scheduled_running_reqs.append(request)\n"
    "            prefill_scheduled |= request.is_prefill_chunk\n"
    "            request_id = request.request_id\n"
    "            req_to_new_blocks[request_id] = new_blocks\n"
    "            num_scheduled_tokens[request_id] = num_new_tokens\n"
    "            token_budget -= num_new_tokens\n"
    "            # [issue43-hotfix] per-request scheduled-tokens record (issue\n"
    "            # #43 ask #1). Decode-active => \"decode\", else prefill chunk.\n"
    "            _is_dec = (request.num_computed_tokens >= request.num_prompt_tokens\n"
    "                       and not request.is_prefill_chunk)\n"
    "            issue43_step_diag[\n"
    "                \"decode\" if _is_dec else \"prefill\"][request_id] = num_new_tokens\n"
    "            req_index += 1\n"
)
src = src.replace(A5_OLD, A5_NEW, 1)

# --- 6. step summary log + stash diag on self, at end of running loop -------
A6_OLD = "        # Record the LoRAs in scheduled_running_reqs\n"
assert A6_OLD in src, "issue43: end-of-running-loop anchor not found; refusing to patch"
A6_NEW = (
    "        # [issue43-hotfix] step summary (issue #43 asks #1/#2). Stash the\n"
    "        # per-step diag for the live reproducer; emit a compact log line\n"
    "        # only when DSPARK_ISSUE43_SCHED_DIAG=1 to keep default overhead 0.\n"
    "        self.issue43_last_step_diag = issue43_step_diag\n"
    "        if _ISSUE43_SCHED_DIAG and (issue43_step_diag[\"prefill\"]\n"
    "                                    or issue43_step_diag[\"decode\"]\n"
    "                                    or issue43_step_diag[\"skips\"]):\n"
    "            logger.info(\n"
    "                \"[issue43-step %d] run=%d prefill_toks=%s decode_toks=%s \"\n"
    "                \"decode_skips=%s\",\n"
    "                self.current_step, len(scheduled_running_reqs),\n"
    "                issue43_step_diag[\"prefill\"], issue43_step_diag[\"decode\"],\n"
    "                issue43_step_diag[\"skips\"])\n"
    "\n"
    "        # Record the LoRAs in scheduled_running_reqs\n"
)
src = src.replace(A6_OLD, A6_NEW, 1)

P.write_text(src)
print(f"[issue43-hotfix] patched {P}")