# Decision Log

## 2026-09-23 — MiMo looping: fp8 KV mod capped full-attention layers at 128 tokens (uncommitted, major)

- **Files**: `mods/mimo-diffkv-fp8-kv/run.sh`, `tests/test_mimo_diffkv_fp8_kv_mod.sh`,
  `mods/mimo-diffkv-fp8-kv/README.md`, `recipes/3x-spark-cluster/mimo-v2.6-flash-pp3.yaml`
- **Symptom**: MiMo-V2.6-Flash-RL (base and dealignai) looped in long generations: coherent for a few
  hundred tokens, then one phrase or token repeated to max_tokens=6000. Seen at TP=2 (4/4 and 5/6 probe
  prompts) and PP=3 (1 reproduced), thinking on and off, default sampling.
- **A/B runs** (same six-prompt probe): fp8 KV + DeepGEMM on: loops. bf16 KV (mod off): clean (PP=3 6/6,
  TP=2 1/1). fp8 KV + `VLLM_USE_DEEP_GEMM=0`: still loops (5/6), so DeepGEMM is not the cause.
- **Cause**: the mod passes `cache_config` into `Attention()` to make fp8 KV apply, but `Attention()` falls
  back to `cache_config.sliding_window` (128) for layers without a per-layer window, so the 9
  full-attention layers became 128-token sliding-window layers whenever the mod was on. Found by
  comparing with tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe patch 01, which clears the window.
- **Fix**: full-attention layers get a copy of `cache_config` with `sliding_window = None`. Verified: the
  unmodified TP=2 recipe (fp8 KV) passes 6/6 (all finish=stop, no repetition). The reported KV pool drops
  from 29.5M to 2.14M tokens at 1M max-model-len: the old figure counted full-attention layers as
  128-token windows.
- **Reverted**: the earlier same-day entry that dropped fp8 KV from the PP=3 recipe; fp8 KV and the mod are
  back. The GitHub-reported fp8 QKV loader and omni `SupportsEagle3` gaps are already fixed in the image
  (vllm#57508 loader, marker present); `VLLM_USE_DEEP_GEMM=0` is kept in the homelab env files anyway
  (DeepGEMM#417).

## 2026-09-21 — MiMo-V2.6 TP=3 request → PP=3 recipe (uncommitted, minor)

- **Files**: `recipes/3x-spark-cluster/mimo-v2.6-flash-pp3.yaml`
- **Choices**: literal TP=3 (requested) vs. PP=3 on 3 nodes vs. stay TP=2 and
  skip the 3rd node. **Decision**: PP=3 (TP=1 x PP=3). Why: vLLM rejects TP=3
  for this model (64 Q heads, 4 KV heads, 8 SWA KV heads, 256 experts — none
  divisible by 3); 48 layers divide into 16 per stage and ~58 GB/stage fits
  the reduced-memory (108 GB) 3rd node, which TP=2 (86.5 GB weights + KV)
  would not.
- **Choices**: carry DFlash spec decode and/or UCM onto the 3-node recipe vs.
  baseline first. **Decision**: baseline only. Why: spec decode is restricted
  and KV connectors are partial under PP>1 upstream; validate parallelism and
  memory first, carry features back after verification (`todo.md`).
- **Choices**: `--distributed-executor-backend ray` (present in existing nested
  recipes) vs. omit. **Decision**: omit, per `docs/AGENT_DEVELOPMENT.md`
  (runner/launch-cluster supply backends; no-ray is default).
- **Choices**: keep `FULL_DECODE_ONLY` cudagraphs vs. default. **Decision**:
  default PIECEWISE (no `--compilation-config`). Why: FULL_DECODE_ONLY was a
  TP=2 memory-headroom choice; no PP>1 precedent exists in this repo.
- The 3rd node's address is intentionally not embedded in tracked files
  (`docs/AGENT_DEVELOPMENT.md`: no machine-specific IPs); pass it via `-n` or
  `.env` `CLUSTER_NODES` at launch time.

## 2026-09-22 — Review of official vLLM MiMo-V2.6 recipe (uncommitted, minor)

- **Reference**: vllm-project/recipes `@09dea3c390ca123b6d3f1aaca7774c8fffa51654`
  `models/XiaomiMiMo/MiMo-V2.6-Flash-RL.yaml`.
- **Choices**: adopt official args vs. keep local set vs. comment-only.
  **Decision**: comment-only changes to `recipes/mimo-v2.6-flash.yaml` (draft
  TP pin note, thinking-toggle/sampling note, official FULL_DECODE_ONLY+DFlash
  precedent link). Why: official agrees with local on the load-bearing flags
  (concrete local dflash path, 7 draft tokens, FULL_DECODE_ONLY with DFlash,
  #57784 requirement, mimo parsers, 1M context); the remaining diffs are
  platform-specific to Sparks (`marlin`, `fp8` KV, 0.87 util) or neutral
  (`--max-model-len auto` == pinned 1048576).
- **Not adopted**: `--gpu-memory-utilization 0.95` (H200 dedicated VRAM),
  dropping `--moe-backend marlin` / `--kv-cache-dtype fp8` (sm_121 needs),
  image `vllm/vllm-openai:mimo-v26` (repo builds its own from main).
- **Follow-up**: official points `--speculative-config` directly at
  `<snapshot>/dflash` with no staged copy — re-verify the staging mod's
  sanitization premise once the local snapshot has `dflash/` (`todo.md`).
  (2026-09-22 update: tonyd2wild's repo independently fixes the same
  trailing-comma `dflash/config.json`, so staging stays for now.)

## 2026-09-22 — Review of tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark (uncommitted, minor)

- **Reference**: https://github.com/tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark
  (measured GB10/2x-Spark deployment of the same checkpoint, TP2).
- **Files**: new `mods/mimo-diffkv-fp8-kv/` + `tests/test_mimo_diffkv_fp8_kv_mod.sh`;
  all three MiMo recipes gain the mod; `mimo-v2.6-flash` and
  `mimo-v2.6-flash-ucm` gain `--no-async-scheduling`.
- **Choices**: fp8 KV broken upstream — fix via mod vs. drop the flag vs.
  copy Tony's 300K/bf16 config. **Decision**: mod adapting his patches 01+03
  (verified against vllm main 2026-09-22: `mimo_v2.py` never passes
  `cache_config` — dead param, layer dtype stays "auto"; DiffKV backend
  rejects quantized KV). Why: the 1M-context memory math in all three
  recipes depends on fp8 KV, and he validated the identical fix on GB10.
  Deviation: the mod keeps rejecting non-e4m3 quantized caches (his patch
  removed the guard; the fp8 view would corrupt e5m2/nvfp4).
- **Choices**: async scheduling — add `--no-async-scheduling` vs. trust the
  default. **Decision**: flag on the two DFlash recipes. Why: vllm#46669 is
  open (MTP/DFlash + async -> garbage at concurrency>1 on MiMo; async off is
  the only mitigation) and upstream whitelists dflash as async-compatible, so
  the default resolves to ON. pp3 exempt: no spec decode.
- **Not adopted**: `VLLM_USE_DEEP_GEMM=0` (our Dockerfile pins DeepGEMM
  `a6b593d`, the pre-regression side of DeepGEMM#417; his image pinned the
  broken `8b1392b`); GMU 0.90 / 300K context (throughput profile, conflicts
  with the documented 1M goal); `SupportsEagle3` patch (upstream now marks
  both mimo classes); TP4 `--linear-backend triton` note (not TP4 here);
  NFS weight sharing (repo has hf-download distribution). Thinking-off and
  rep-penalty server defaults: question left unanswered, not adopted —
  revisit (`todo.md`).
## 2026-09-21 — UCM KV cache offload for MiMo-V2.6 (uncommitted, minor)

- **Files**: `mods/ucm-kv-offload/`, `recipes/mimo-v2.6-flash-ucm.yaml`,
  `tests/test_ucm_kv_offload_mod.sh`
- **Choices**: wire UCM into the existing `mimo-v2.6-flash` recipe vs. mod-only
  opt-in vs. separate recipe. **Decision**: separate recipe reusing a shared
  mod; original recipe untouched. Why: user request; keeps the proven recipe
  unchanged and UCM opt-in per launch.
- **Choices**: store backend = compressed Cache|Compress|Posix vs. DRAM vs.
  uncompressed Cache|Posix. **Decision**: uncompressed `Cache|Posix` on local
  storage now; remote compressed NFS store deferred (`todo.md`). Why: user
  pick; fastest I/O, no compression overhead.
- **Choices**: store dir `/ucm-kv` (needs manual `-v`) vs. default inside the
  host-mounted vLLM cache. **Decision**: default
  `/root/.cache/vllm/ucm-kv`, overridable via `UCM_KV_DIR`. Why: works
  out of the box, persists across fresh containers; launch-cluster mounts
  `~/.cache/vllm` by default.
- **Choices**: install UCM at mod runtime vs. bake into Dockerfile vs. vendor
  wheel. **Decision**: pinned `uv pip install uc-manager[cu130]==0.7.0` at
  container launch. Why: same pattern as `mods/exp-b12x`; no image rebuild, no
  binary in git; image env already sets `UV_SYSTEM_PYTHON=1`.