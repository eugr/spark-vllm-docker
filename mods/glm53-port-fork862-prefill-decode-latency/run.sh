#!/usr/bin/env bash
# glm53-port-fork862-prefill-decode-latency — port of the contended-prefill
# bound (fork local-inference-lab/vllm PR #862, commit 1fbc36ab7, yatesdr).
#
# STATUS: EXPERIMENTAL, lab only. Inert without
#   VLLM_GLM53_PREFILL_DECODE_LATENCY=1 (also requires the knob > 0).
#
# WHY: with prefill compute-sharing, one long prefill (author measured a 286 ms
#   decode gap during a 229K-token prefill) hogs whole steps and decodes only
#   get leftovers. The --max-num-prefill-tokens-per-step knob caps prefill
#   tokens per CONTENDED step; decode-only and uncontended prefill steps are
#   unchanged.
# WRITTEN PREDICTION: steady c1 decode ~0 ms (no contention without a
#   concurrent prefill). On mixed prefill+decode steps: bounded step spikes
#   instead of 200+ ms; total prefill slightly slower (more steps). Starting
#   point: VLLM_GLM53_PREFILL_DECODE_LATENCY=1 plus
#   --max-num-prefill-tokens-per-step 512 (needs prefill_compute_share already
#   active; must be <= max_num_batched_tokens, e.g. 2048). Risk: too low slows
#   prefill without helping decode; too high bounds nothing. Sweep 256/512/1024.
# EXECUTION PROOF: with env=1 and knob>0, ONCE when actually bounding:
#   "[GLM53-PORT-FORK862-PREFILL-DECODE-LATENCY] bounding contended prefill to N tokens/step".
#   Without env or with knob 0: line absent, scheduler identical to ab86b7073.
# ROLLBACK: unset the env (inert without it) or drop it from the mods list.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VLLM=/usr/local/lib/python3.12/dist-packages/vllm
python3 "$HERE/patch_fork862_prefill_decode_latency.py" --vllm-root "$VLLM"
python3 -m py_compile "$VLLM/config/scheduler.py"
python3 -m py_compile "$VLLM/engine/arg_utils.py"
python3 -m py_compile "$VLLM/v1/core/sched/scheduler.py"
echo "[glm53-port-fork862-prefill-decode-latency] OK"
