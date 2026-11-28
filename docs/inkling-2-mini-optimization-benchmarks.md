# Inkling 2 Mini Optimization Benchmarks on DGX Spark

- **Date:** 2026-11-28
- **Hardware:** Single NVIDIA DGX Spark (GB10, SM 12.1, 48 SMs, 119 GiB unified memory, ~207.7 GB/s measured copy bandwidth, NVIDIA driver 580.178.04)
- **Pinned image:** `vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e` (`vLLM 0.1.dev25855+g7cb5dae22.cu130`, PyTorch 2.14.0, CUDA 13.0)
- **Target checkpoint:** `thinkingmachines/Inkling-2-Mini` (`c9d028302cd0f035fdf8e7e91f089a4ce94d2195`)
- **Draft checkpoint:** `incoai/Inkling-2-Mini-DFlash2` (`4f9088ea66c21f2b99ed657d1941c306b03e84d2`, 7 probabilistic draft tokens)
- **Shared settings:** `--tensor-parallel-size 1`, `--max-model-len 262144`, `--gpu-memory-utilization 0.8`, `--moe-backend humming`, `--reasoning-parser inkling`, `--tool-call-parser inkling`, `--enable-auto-tool-choice`, plus `mods/inkling2-dflash2`
- **Companion HTML report:** [docs/inkling-2-mini-spark-findings.html](inkling-2-mini-spark-findings.html)

---

## 1. End-to-End Controlled Benchmark Sweep

All 8 configurations were benchmarked back-to-back on the same host with the same frozen workload:

- **Coding (`Code`):** 4 tasks (`python_ttl_lru`, `typescript_async_map`, `rust_log_aggregator`, `postgres_event_analytics`) × 2 measured runs after per-task warmup, 512 output tokens, `temperature=0`, `seed=42`, `ignore_eos=true`, `reasoning_effort="none"`. **Step** = `mean_accepted_len / median_decode_tps`.
- **Sampled (`Samp`):** 3 technical prompts, 512 output tokens, `temperature=0.6`, `ignore_eos=true`, `reasoning_effort="high"`.
- **Concurrency (`4-req` / `8-req`):** 4 and 8 concurrent requests, 256 output tokens each, `temperature=0.6`, `reasoning_effort="none"`.
- **Prefill (`PF 32K`):** Warmed prefill on fresh unique-nonce ~32.5K-token prompts (2 measured repeats after 1 warmup).
- **Long-context decode (`Dec 41K`):** ~41K-token cold prompt followed by 256 output tokens at `temperature=0`.

| Configuration | Coding (tok/s) | Step (ms) | Acc % | TTFT (ms) | Sampled (tok/s) | 4-req (tok/s) | 8-req (tok/s) | PF 32K (tok/s) | Dec 41K (tok/s) | 41K Step | KV Capacity (tok) | Greedy Hash Match vs. Baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_humming_dflash2` | 55.77 | 61.0 | 34.3% | 217.5 | 50.91 | 107.91 | 178.35 | 2,850.5 | 47.03 | 64.3 ms | 1,450,299 | 4/4 (ref) |
| `exp1_fp8_kv` | 60.26 | 59.4 | 36.8% | 231.2 | 57.50 | 102.93 | 166.18 | 2,666.0 | 46.24 | 61.8 ms | **2,282,491** | 0/4 (det) |
| **`exp2_dflash2_quant` (Lossless recipe)** | **59.04** | **57.9** | 34.6% | **204.0** | **54.22** | **114.70** | 178.16 | 2,869.0 | **51.36** | **61.3 ms** | 1,458,487 | **4/4 exact** |
| `exp3_moe_w4a16_sk` | 60.14 | 62.2 | 39.1% | 231.3 | 51.04 | 112.33 | 165.45 | 2,933.1 | 42.35 | 69.5 ms | 1,464,628 | 0/4 (non-det) |
| `exp4_lossless_stack` | 55.52 | 57.6 | 31.4% | 205.4 | 55.60 | 112.55 | 173.83 | 3,154.2 | 48.69 | 62.8 ms | 1,468,722 | 0/4 (det) |
| `exp5_target_dense_mxfp8` | 70.88 | 51.0 | 37.3% | 192.4 | 60.42 | 121.09 | 177.20 | 3,091.9 | 49.08 | 56.1 ms | 1,468,722 | 0/4 (det) |
| **`exp6_mxfp8_all` (MXFP8 recipe)** | **73.40** | **48.8** | 36.9% | **192.1** | **68.41** | **126.10** | **191.10** | 3,084.7 | **60.11** | **53.2 ms** | 1,452,005 | 0/4 (det) |
| `exp7_mxfp8_all_b4096` | 73.00 | **48.8** | 36.6% | 193.9 | 68.03 | **126.78** | 181.96 | **3,335.5** | 53.81 | 54.3 ms | 1,106,066 | 0/4 (det, 4/4 vs. `exp6`) |

### Per-Task Coding Breakdown (Median tok/s)

| Configuration | Python TTL LRU | TypeScript async map | Rust log aggregator | PostgreSQL analytics | Overall Median | Mean Accepted Len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_humming_dflash2` | 66.83 | 51.65 | 53.49 | 58.58 | 55.77 | 3.40 |
| `exp1_fp8_kv` | 76.98 | 54.39 | 66.13 | 48.58 | 60.26 | 3.58 |
| **`exp2_dflash2_quant`** | **70.52** | **55.57** | **56.73** | **61.17** | **59.04** | **3.42** |
| `exp3_moe_w4a16_sk` | 64.38 | 58.90 | 58.20 | 60.01 | 60.14 | 3.74 |
| `exp4_lossless_stack` | 72.73 | 53.75 | 56.86 | 49.50 | 55.52 | 3.20 |
| `exp5_target_dense_mxfp8` | 80.35 | 71.04 | 64.85 | 70.76 | 70.88 | 3.61 |
| **`exp6_mxfp8_all`** | **80.98** | **65.57** | **72.95** | **73.65** | **73.40** | **3.58** |
| `exp7_mxfp8_all_b4096` | 80.74 | 65.83 | 73.09 | 72.83 | 73.00 | 3.56 |

---

## 2. Experiment Descriptions

| Label | Flags / `mods/inkling2-perf-opt` Env Vars | Summary of Findings |
| --- | --- | --- |
| `baseline_humming_dflash2` | `recipes/inkling-2-mini-dflash2.yaml` defaults | Reference baseline (`55.77` coding, `50.91` sampled, `107.9`/`178.4` 4/8-req, `2,851` 32K prefill, `47.03` @ 41K). |
| `exp1_fp8_kv` | `--kv-cache-dtype fp8` | Expands KV capacity **+57%** (`1.45M → 2.28M` tok, `8.71×` at 262K) and trims 1-req step latency 2–4%, but slows 32K prefill (−6.5%) and 4/8-req concurrency (−5–7%) due to FP8→BF16 dequant in Triton `_load_fp8_kv`. |
| `exp2_dflash2_quant` | `INKLING2_OPT_DFLASH_LMHEAD=nvfp4`<br>`INKLING2_OPT_DFLASH_BACKBONE=mxfp8` | **Best strictly target-lossless config.** Target weights/activations/`lm_head` untouched; **4/4 bitwise-identical greedy hashes** with baseline; +5.9% coding, +6.5% sampled, +6.3% 4-req, +9.2% decode @ 41K, −130 MiB model memory (~1.0 s load-time quant). |
| `exp3_moe_w4a16_sk` | `INKLING2_OPT_MOE=w4a16_sk` | Runs all 24 MoE layers via `HummingExperts` W{4,8}A16 with Stream-K on NVFP4. BF16 activations raised coding acceptance `34.3% → 39.1%`, but Stream-K BF16 atomics broke greedy determinism and W4A16 regressed 8-req throughput (`165.5` vs. `178.4 tok/s`). |
| `exp4_lossless_stack` | `INKLING2_OPT_MOE=mxfp8_only`<br>`INKLING2_OPT_DFLASH_LMHEAD=nvfp4`<br>`INKLING2_OPT_DFLASH_BACKBONE=mxfp8`<br>+ 15-shape sparse prefill CUDA graphs | Keeps deterministic W4A4 on NVFP4 layers (with cached `moe_align_block_size` between `w13` and `w2`), switches the 4 MXFP8 coda layers from Triton to Humming MXFP8 W8A16, and trims graph memory −21% (`2.17 → 1.71 GiB`). Lifts 32K prefill **+10.6%** (`2,851 → 3,154 tok/s`) and sampled decode **+9.2%** (`50.91 → 55.60 tok/s`). Greedy hashes differ from baseline because the 4 coda layers switch from W8A8 to W8A16. |
| `exp5_target_dense_mxfp8` | `exp4` + `INKLING2_OPT_TARGET_DENSE=mxfp8` | Adds online block-32 MXFP8 W8A16 on target `qkvr` + `wo_ud` (cutting 2.73 GB → 1.37 GB/step) while keeping target `lm_head` in BF16. Jumps coding to **70.88 tok/s** (+27.1%), sampled to **60.42 tok/s** (+18.7%), 4-req to **121.09 tok/s** (+12.2%), raw step latency `61.0 → 51.0 ms` (−16.4%). |
| `exp6_mxfp8_all` | `exp4` + `INKLING2_OPT_TARGET_DENSE=mxfp8_all` | **Fastest overall config.** Also quantizes target `lm_head` (0.93 GB → 0.46 GB) to block-32 MXFP8 W8A16. Beats baseline on every metric: **73.40 tok/s** coding (+31.6%, Python `80.98 tok/s`, step `48.8 ms` / −20.0%), **68.41 tok/s** sampled (+34.4%), **126.10 / 191.10 tok/s** 4/8-req (+16.9% / +7.1%), **3,085 tok/s** 32K prefill (+8.2%), **60.11 tok/s** @ 41K (+27.8%). |
| `exp7_mxfp8_all_b4096` | `exp6` + `--max-num-batched-tokens 4096` | Lifts 32K prefill to **3,335.5 tok/s** (+17.0% vs. baseline) and 41K prefill to **3,199.3 tok/s**, matching `exp6` on 4/4 greedy hashes and 1-req decode (`73.00 tok/s`), while reducing KV capacity (`1.45M → 1.11M` tok) and 8-req throughput (`191.1 → 182.0 tok/s`). |

---

## 3. Standalone GB10 Kernel Microbenchmarks

Measured directly on GB10 inside `vllm/vllm-openai:inkling-2-mini`:

### MoE Layer (128 experts, top-8, H=2304, I=1152, single layer)

| Tokens (`M`) | NVFP4 Triton | NVFP4 Humming W4A4 | NVFP4 Humming W4A16 (SK) | MXFP8 Triton | MXFP8 Humming W8A16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.055 ms | 0.989 ms | **0.899 ms** | 1.837 ms | **1.710 ms** |
| 32 | 2.315 ms | 2.161 ms | **2.100 ms** | 4.204 ms | **4.003 ms** |
| 64 | 2.807 ms | 2.623 ms | **2.580 ms** | 4.615 ms | **4.510 ms** |
| 512 | 3.759 ms | 3.266 ms | **3.097 ms** | 5.670 ms | **5.194 ms** |
| 2048 | 8.537 ms | **4.783 ms** | 5.217 ms | 14.621 ms | **7.531 ms** |

### Output Head (`N=200064`, `K=2304`) & 5-Layer DFlash2 Backbone

| Op / Batch (`M`) | cuBLAS BF16 | `sm12x_bf16_gemv` | Humming MXFP8 W8A16 | Humming NVFP4 W4A16 |
| --- | ---: | ---: | ---: | ---: |
| `lm_head` `M=7` (1-req draft) | 4.063 ms | 3.916 ms | 2.109 ms | **1.174 ms** |
| `lm_head` `M=8` (1-req target) | 3.993 ms | 3.904 ms | 2.064 ms | **1.162 ms** |
| `lm_head` `M=28` (4-req draft) | 4.542 ms | — | 2.094 ms | **1.220 ms** |
| `lm_head` `M=56` (8-req draft) | 4.674 ms | — | 2.210 ms | **1.377 ms** |
| 5-layer DFlash2 `M=8` | 2.913 ms | 3.089 ms | **1.398 ms** | 1.805 ms *(suboptimal `M=8` tile)* |
| 5-layer DFlash2 `M=16` | 2.780 ms | 3.213 ms | 1.480 ms | **0.795 ms** |
| 5-layer DFlash2 `M=32` | 3.204 ms | — | 1.228 ms | **0.474 ms** |

---

## 4. Quality & Functional Verification (`exp6`/`exp7` `mxfp8_all`)

- **DFlash2 acceptance alignment:** Coding acceptance `36.9%` (`3.58` mean len) and sampled reasoning acceptance `32.7%` (`3.29` mean len) remained at or above baseline (`34.3%` / `28.5%`), indicating tight logit agreement with the BF16 target.
- **Math + reasoning (`17 * 23 + 19`):** PASS (`410` with reasoning trace).
- **Tool calling (`get_weather`):** PASS (`{"city":"Tokyo","unit":"celsius"}`).
- **Long-context 3-needle retrieval:** PASS (`3/3` at 30,086 tokens in 13.3 s; `3/3` located at 93,852 tokens in 42.4 s).
- **Caveat:** Greedy outputs differ from unquantized BF16 dense projections, and an un-fenced 3,072-token Python generation check hit a syntax error in a unit test. For strict target-model fidelity, use `inkling-2-mini-dflash2-lossless` (`exp2`, 4/4 exact greedy hashes) or override `-e INKLING2_OPT_TARGET_DENSE=mxfp8` (`exp5`, BF16 target `lm_head`, `70.88 tok/s`).

---

## 5. Packaged Recipes

Both winning configurations are packaged in `recipes/` using [mods/inkling2-perf-opt](../mods/inkling2-perf-opt/README.md):

```bash
# Best strictly target-lossless (exp2_dflash2_quant)
./run-recipe.sh inkling-2-mini-dflash2-lossless --solo --setup

# Fastest overall (exp6_mxfp8_all)
./run-recipe.sh inkling-2-mini-dflash2-mxfp8 --solo --setup
```

---

## 6. Candidates for Future Optimization

1. **Direct flat-layout Humming W4A4 MoE (lossless):** Skip tile-padding (`64 → 512` rows) and R8c4 scale swizzling in `route()`/`swiglu_quantize()` when `--moe-backend humming` is active to eliminate 88 `_prepare_rows` Triton launches/step and shrink `moe_align_block_size` 8×.
2. **GB10-specific Humming autotuning (lossless):** Replace generic SM120 heuristics for the `(2304, 2304)` / `(2304, 1152)` MoE shapes and fix the `M=8` NVFP4 tile for the 5-layer DFlash2 backbone (`1.81 → 0.80 ms`).
3. **Sparse draft vocabulary for DFlash2 `lm_head` (lossless):** Limit draft candidate projection to the top 32K–64K tokens (`1.17 → ~0.3 ms/step`).
4. **SM121 Triton attention & native FP8 KV kernel tuning:** Autotune `attention_sm90.py` for 48 SMs and remove software FP8→FP16→BF16 conversion overhead in `_load_fp8_kv` so `--kv-cache-dtype fp8` helps across all lengths.
5. **Deterministic W4A16 MoE (`INKLING2_OPT_MOE=w4a16`, no Stream-K) + `mxfp8_all`:** Re-evaluate single-request interactive coding throughput with higher BF16-activation acceptance (`~39%`).
6. **Startup & memory cleanup:** Avoid creating unused fused/split `qkvr` Humming copies (~0.6 GiB saved), add `--limit-mm-per-prompt '{"image":0}'` for text-only serving, and cache repacked Humming MoE weights on disk to cut ~100 s load time.
