# Qwen3.8-27B-NVFP4 — Decode Performance Findings

> **Status: superseded historical diagnosis.** This note records the initial
> stale-image investigation and its performance hypotheses. The predicted
> 40–50 tok/s target was not confirmed after refreshing the image. See
> [Qwen3.8-27B-NVFP4 MTP Configuration Benchmark](qwen3.8-27b-nvfp4-mtp-benchmark.md)
> for the controlled post-refresh measurements and current recommendations.

**Date:** 2026-08-16
**Recipe:** `recipes/qwen3.8-27b-nvfp4.yaml` (solo, single DGX Spark / GB10)
**Model:** `unsloth/Qwen3.8-27B-NVFP4`
**Symptom:** ~20 t/s average token generation (user-reported)

> Working note, not part of the permanent documentation.

## TL;DR

Decode is **8.2 t/s (ITL ≈ 121 ms)** because the container runs a **stale vLLM
build (2026-08-06) that predates the Qwen GDN MTP speculative-decoding fixes
merged 2026-08-14**, which the recipe explicitly requires. MTP itself is
healthy (77% draft acceptance); the pre-fix GDN+MTP path costs ~370 ms of GPU
time per verification step (94–95% GPU utilization) instead of ~55 ms.
Refreshing the runner image and restarting is the fix. Secondary: the host is
under memory pressure (3 GB free, 5 GB swap in use, GUI running).

## Environment

- Single DGX Spark (GB10, ~121 GB usable unified memory), container `vllm_node`
  (image `vllm-node`, built **2026-08-06T11:37:21-07:00**)
- vLLM in container: `0.26.1rc1.dev439+g7b9f2dad8.d20260806`
- Host memory at time of diagnosis (`free -h`):

  ```
  Mem:  total 119Gi  used 112Gi  free 3.0Gi  buff/cache 5.6Gi  available 7.4Gi
  Swap: total 15Gi    used 5.0Gi
  ```

- Other notable consumers: GUI (Xorg/gnome-shell), Firefox, ChatGPT desktop,
  VS Code, open-webui, prometheus, grafana. vLLM `EngineCore` holds
  ~98 GB of the unified pool (per `nvidia-smi`).
- Recipe settings in effect (as launched):

  ```
  --tensor-parallel-size 1 --gpu-memory-utilization 0.5 --max-model-len 262144
  --max-num-seqs 4 --max-num-batched-tokens 8192 --kv-cache-dtype fp8
  --attention-backend flashinfer --load-format fastsafetensors
  --enable-chunked-prefill --enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  ```

## Measurements

### 1. Pure decode probe (streaming completions, short prompt, 160–512 tokens)

| Run | Tokens | Span | ITL | Rate |
|-----|--------|------|-----|------|
| probe 1 | 224 | 27.51 s | 123.4 ms | 8.1 t/s |
| probe 2 (256-token request) | ~110–256 | ~12–14 s | — | ~8–9 t/s |
| run 1 | 54 | 6.44 s | 121.5 ms | 8.2 t/s |
| run 2 | 54 | 6.49 s | 122.4 ms | 8.2 t/s |
| run 3 | 55 | 6.50 s | 120.4 ms | 8.3 t/s |

Consistent across runs → not intermittent; ~121 ms/token sustained.

### 2. MTP speculative-decoding metrics (cumulative, `/metrics`)

- `spec_decode_num_drafts_total` = 13,792; `num_draft_tokens_total` = 41,376 (3/draft ✓)
- `num_accepted_tokens_total` = 24,952 → cumulative mean accepted length ≈ 2.8
  (user's Grafana 1-min window showed **3.09**; range 1.0–4.0 for k=3)
- Per-position acceptance: pos0 **76.5%**, pos1 **58.5%**, pos2 **46.0%**

**Interpretation:** MTP is working well. At ~3.09 tokens/step, decode should be
~2.5–3× the no-spec bandwidth floor.

### 3. Bandwidth-floor check

- 27B dense model, NVFP4 ≈ 14–15 GB weights; GB10 ≈ 273 GB/s → a single decode
  step (one weight read) ≈ **50–60 ms** → ~17–19 t/s without speculation.
- With MTP, one step ≈ 1 fused verify (4 tokens) + ~1/48 model MTP draft →
  expected **~45–55 t/s**.
- Measured: ~370 ms per verify step at 8.2 t/s → **~6–7× the expected step cost.**

### 4. Where the time goes

- `nvidia-smi` during decode: **GPU utilization 94–95%**, 68 °C, ~37 W, P0
  → the step is **GPU-bound**, not CPU launch overhead.
- Swap counters (`/proc/vmstat` pswpin/pswpout) **flat during generation**
  → swap thrashing is **not** the decode bottleneck (though the host is
  clearly memory-pressured overall: 5 GB swap in use, direct-reclaim history
  `pgscan_direct` ≈ 6 M).

## Root cause

The container's vLLM build is from **2026-08-06**
(`g7b9f2dad8.d20260806`). The recipe and README changelog (2026-08-16) state:

> Native MTP requires vLLM with the Qwen GDN speculative-decoding fixes from
> vllm-project/vllm#51812 and #51674 (**merged 2026-08-14 or later**).
> Refresh or rebuild older runner images before launching this recipe.

The pre-fix GDN+MTP path produces correct output (acceptance metrics are
healthy) but spends ~370 ms of GPU time per speculative verification step —
consistent with redundant/non-fused verify work (e.g., sequential per-token
targets forwards and/or redundant GDN state updates) that the fixes remove.

No server-side launch parameter can compensate; the fix ships in the vLLM
build.

## Secondary findings

1. **Host memory pressure.** vLLM holds ~98 GB of the pool; GUI + apps leave
   3 GB free and 5 GB on swap. README explicitly warns the graphical session
   costs RAM and a little performance. Not the decode bottleneck (no swap I/O
   during generation), but it limits KV-cache headroom and OOM safety.
2. **`--async-scheduling` missing** from this recipe (the sibling
   `qwen3.6-35b-a3b-nvfp4` recipe has it). Minor decode-latency win.
3. **MTP k=3 is the right setting** for this workload: pos-2 acceptance is
   46%, so `num_speculative_tokens: 4` would buy little (each extra MTP draft
   costs ~1/48 of the model).

## Recommended actions

1. **Refresh the runner image and restart the container** (the real fix;
   expect ~40–50 t/s decode, i.e. ~5× current):

   ```bash
   ./run-recipe.sh qwen3.8-27b-nvfp4 --solo --setup --force-build
   ```

   Then verify:

   ```bash
   docker exec vllm_node vllm --version   # must be dated >= 2026-08-14
   ```

   and re-run the decode probe below; target ITL < ~40 ms.

2. **Relieve host memory:** log out of the GUI or
   `sudo systemctl isolate multi-user.target` (or close Firefox/ChatGPT/VS
   Code); watch `free -h` (target: no swap growth, >10 GB available).

3. **Optional:** add `--async-scheduling`:

   ```bash
   ./run-recipe.sh qwen3.8-27b-nvfp4 --solo -- --async-scheduling
   ```

## Re-test probe (used for this diagnosis)

```bash
curl -sN http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "unsloth/Qwen3.8-27B-NVFP4",
  "prompt": "Probe run '"$(date +%s)"': Explain the water cycle briefly.",
  "max_tokens": 160, "stream": true, "temperature": 0.7}' \
| while IFS= read -r line; do case "$line" in data:\[DONE\]*) ;; data:*) date +%s.%N;; esac; done \
| awk 'NR==1{first=$1} {last=$1; n++} END{printf "chunks=%d  span=%.2fs  ITL=%.1fms  %.1f t/s\n", n, last-first, (last-first)*1000/(n-1), (n-1)/(last-first)}'
```

Acceptance metrics: `curl -s localhost:8000/metrics | grep spec_decode` or the
**vLLM Speculative Decoding** Grafana dashboard
(`grafana/dashboards/vllm-spec-decode.json`).

## Baseline vs. target

| | Before (2026-08-16, stale build) | Target (post image refresh) |
|---|---|---|
| Decode rate | 8.2 t/s (ITL 121 ms) | ~40–50 t/s (ITL ~25–35 ms) |
| GPU time per MTP step | ~370 ms at 95% util | ~55–60 ms (bandwidth floor) |
| MTP mean accepted length | 3.09 (keep) | ≥ 2.8 |
| Host free / swap | 3 GB / 5 GB used | no swap growth, >10 GB free |
