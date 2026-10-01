#!/usr/bin/env bash
# deepseek-v4-dspark-027-backports — apply the DSpark 2x-DGX-Spark live hotfix
# chain (vLLM 0.27.0 backports + issue fixes) onto the Anemll
# dspark-vllm-gx10:0.1.1 image, in the same order the upstream compose
# entrypoint applies them (docker-compose.dspark.yml). Ported verbatim from
#   DeepSeek-v4-Flash-DSpark-2x-DGX-Spark/patches/.
# Runs inside the container against /usr/local/lib/python3.12/dist-packages/vllm.
# Every patch is idempotent, auto-detects its target, and never restarts vllm.
#
# Control knobs (env; defaults reproduce Mia's on-by-default behavior):
#   DSPARK_SKIP_ISSUE22_HOTFIX=1        skip the nvfp4_ds_mla long-ctx decode fix
#                                       (no-op on fp8 KV; don't skip if KV=nvfp4_ds_mla)
#   DSPARK_SKIP_HOTFIX=1                skip the six v0.27 perf backports (#22 still applies)
#   DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX=1 do not apply the stop-in-<think> guard
# The #21/#31/#55 and #27/#43/#26 python hotfixes ALWAYS run (not gated).
set -euo pipefail
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Always-on encoder + serving-layer python hotfixes.
python3 "$MOD_DIR/hotfix-encoding-dsv4-issue21.py"
python3 "$MOD_DIR/hotfix-dsv4-issue31-v2-thinking-budget-gpu.py"
python3 "$MOD_DIR/hotfix-dsv4-issue55-tool-truncation.py"

# #22 nvfp4_ds_mla long-context decode fix (on by default; before the .sh chain).
if [ "${DSPARK_SKIP_ISSUE22_HOTFIX:-0}" != "1" ] && [ -f "$MOD_DIR/hotfix-nvfp4-ds-mla-issue22.sh" ]; then
  bash "$MOD_DIR/hotfix-nvfp4-ds-mla-issue22.sh" || true
fi

# Six vLLM 0.27.0 perf backports + grammar-advance (skip when DSPARK_SKIP_HOTFIX=1).
if [ "${DSPARK_SKIP_HOTFIX:-0}" != "1" ]; then
  for _hf in \
    hotfix-dsv4-mtp-buffer-50312.sh \
    hotfix-dsv4-adaptive-topk-50004.sh \
    hotfix-dsv4-skip-topk-49486.sh \
    hotfix-dsv4-dense-prefill-indexer-48407.sh \
    hotfix-dsv4-skip-empty-c128-48957.sh \
    hotfix-dsv4-flashmla-workspace-50298.sh \
    hotfix-dsv4-grammar-advance.sh; do
    [ -f "$MOD_DIR/$_hf" ] && bash "$MOD_DIR/$_hf" || true
  done
fi

# Always-on scheduler / cache python hotfixes (#27, #43, #26).
python3 "$MOD_DIR/hotfix-dsv4-issue27-partial-prefill-concurrency.py"
python3 "$MOD_DIR/hotfix-dsv4-issue43-decode-fairness-and-diag.py"
python3 "$MOD_DIR/hotfix-dsv4-issue26-hybrid-swa-min.py"

# Keep client stop strings dormant until </think> (default on).
if [ "${DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX:-0}" != "1" ]; then
  python3 "$MOD_DIR/hotfix-dsv4-suppress-stops-in-reasoning.py"
fi
