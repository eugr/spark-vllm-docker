#!/usr/bin/env bash
# fix-spec-accepted-clamp-upstream — prevent an accepted-token assertion from stopping EngineCore.
#
# STATUS: Defensive fix. The normal path produces the same result.
# WHY: Guided decoding can invalidate scheduled draft tokens after sampling,
# causing the fork's accepted-token assertion to compare different counts.
# PREDICTION: 0 ms/step measurable change; decode throughput and acceptance
# length should be unchanged. The fix matters only when the assertion would
# otherwise stop EngineCore.
# RUNTIME EVIDENCE: [SPEC-ACCEPTED-CLAMP-UPSTREAM] SOFT or HARD in the log.
# The prefix is emitted only when the corresponding branch executes.
#
# VALIDATED AGAINST: vLLM 0.1.dev21460+gaf9e4dca1 in a 2x DGX Spark TP2 setup.
#
# The affected assertions and histogram update in that build:
#   v1/core/sched/scheduler.py:2796  assert num_accepted <= num_draft_tokens
#   v1/spec_decode/metrics.py:43     assert num_accepted_tokens <= num_draft_tokens <= k
#   v1/metrics/stats.py              self.histogram[num_accepted] += 1
#
# ROLLBACK: remove this mod from the recipe and recreate the container.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VLLM=/usr/local/lib/python3.12/dist-packages/vllm

python3 "$HERE/patch_spec_accepted_clamp.py" --vllm-root "$VLLM"
python3 -m py_compile "$VLLM/v1/core/sched/scheduler.py"

echo "[fix-spec-accepted-clamp-upstream] OK"
