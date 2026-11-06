#!/bin/bash
set -e

# Prefix caching on hybrid (GDN/mamba + attention) models collapses to ZERO
# hits whenever an eagle-family drafter is configured (DFlash/DFlash2, and
# partially degraded under MTP). Five subsystems must agree on the one
# boundary a hit can land on — the eagle drop's target, one hash unit below
# the prompt tail — and upstream they do not: the scheduler's prefill split,
# the mamba tail-state registration, the full-attention partial-tail
# registration, sliding-window retention, and last_cache_position each use a
# different flooring. The (n-1)-basing here also repairs a latent upstream
# off-by-one for hash-aligned prompt lengths.
#
# Measured on Qwen3.8-27B-NVFP4 + DFlash2 (GB10, solo and 2-node TP=2):
# before = 0 hits at nearly every prompt length; after = 100% hits at every
# tested length, multi-turn 99%, TTFT 14-24x on cache hits, ~60% hit rate
# across a full tool-eval-bench run, speculative acceptance unregressed.
echo "Patching hybrid prefix caching for eagle-family drafters"
patch -p0 -d /usr/local/lib/python3.12/dist-packages \
  < prefix_cache_eagle.diff \
  || echo "Patch not applicable (already fixed upstream?), skipping"

python3 - <<'PY' || echo "WARNING: prefix-cache fix self-check failed"
import inspect
import vllm.v1.core.sched.scheduler as s
assert "mamba_tail_eagle_shift" in inspect.getsource(s)
print("hybrid prefix-cache fix active")
PY
