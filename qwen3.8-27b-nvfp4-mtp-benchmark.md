# Qwen3.8-27B-NVFP4 MTP Configuration Benchmark

- **Date:** 2026-08-17
- **Model:** `unsloth/Qwen3.8-27B-NVFP4`
- **System:** Single DGX Spark / GB10
- **General benchmark:** `benchmark-model-configs.py`, one warm-up followed by
  five measured requests per configuration, with the same frozen prompt and
  1,024 generated tokens per request.

## General-purpose results

| Configuration | Runs | Decode rate | Relative to MTP-4 | TTFT | Total time | MTP acceptance | Mean accepted length | Same output as MTP-4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MTP-4 | 5 | 19.16 tok/s | 1.000x | 0.298 s | 53.69 s | 39.0% | 2.56 | Yes |
| MTP-3 | 5 | **19.78 tok/s** | **1.032x** | 0.277 s | **52.11 s** | 48.9% | 2.47 | No |
| No MTP | 5 | 10.77 tok/s | 0.562x | **0.147 s** | 95.13 s | — | — | No |

### General-purpose findings

- MTP-3 delivered the best measured decode rate at **19.78 tok/s**. It was
  **3.2% faster than MTP-4** and **83.7% faster than no MTP**.
- MTP-4 reached **19.16 tok/s**, a **77.9% improvement over no MTP**, but its
  fourth speculative position did not provide enough accepted output to
  recover its additional draft and verification cost.
- Increasing from three to four speculative tokens raised mean accepted length
  by only **0.09 tokens per verification step**, from 2.47 to 2.56. This is a
  3.6% increase in emitted tokens per step while proposing 33% more draft
  tokens per step.
- For a 1,024-token response, MTP-3 reduced total request time from 95.13 to
  52.11 seconds, saving **43.02 seconds** or approximately **45.2%**.
- Speculative decoding increased TTFT by 0.130 seconds for MTP-3 and 0.151
  seconds for MTP-4. This small startup cost is outweighed quickly by the
  higher decode rate for normal-length responses.
- Acceptance is content-dependent. The isolated workload produced lower MTP
  acceptance than some earlier Grafana snapshots, demonstrating why throughput
  must be measured using the same controlled workload rather than inferred
  from acceptance alone.

## Code-generation results

The code workload used `benchmark-code-generation.py` with thinking disabled,
one warm-up, four frozen tasks, three measured requests per task, and 512
generated tokens per request. The tasks covered Python, TypeScript, Rust, and
PostgreSQL.

### Aggregate code performance

| Configuration | Samples | Decode rate | Relative to MTP-3 | TTFT | Request time | MTP acceptance | Mean accepted length | Same output as MTP-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MTP-3 | 12 | 27.75 tok/s | 1.000x | 0.313 s | 18.73 s | 83.5% | 3.51 | Yes |
| MTP-4 | 12 | 29.03 tok/s | 1.046x | **0.283 s** | 17.92 s | 71.6% | 3.86 | No |
| MTP-5 | 12 | **29.25 tok/s** | **1.054x** | 0.314 s | **17.80 s** | 63.6% | **4.18** | No |

MTP-4 improved aggregate code throughput by **4.6%** over MTP-3. MTP-5
increased that improvement to **5.4%**, but was only **0.8% faster than MTP-4**.
That final difference is small enough to be measurement noise, while MTP-4 had
the lowest aggregate TTFT.

### Per-language results

Each cell below shows median decode rate followed by mean accepted length.

| Language/task | MTP-3 | MTP-4 | MTP-5 | Measured winner |
| --- | ---: | ---: | ---: | --- |
| Python TTL LRU | 28.85 tok/s / 3.66 | **31.18 tok/s / 4.16** | 30.10 tok/s / 4.34 | MTP-4 |
| TypeScript async map | 28.06 tok/s / 3.54 | 31.80 tok/s / 4.27 | **32.82 tok/s / 4.73** | MTP-5 |
| Rust log aggregator | 27.47 tok/s / 3.47 | 27.85 tok/s / 3.75 | **28.41 tok/s / 4.10** | MTP-5, marginal |
| PostgreSQL analytics | **26.47 tok/s / 3.35** | 25.40 tok/s / 3.40 | 25.49 tok/s / 3.69 | MTP-3 |

### Code-generation findings

- **Python:** MTP-4 was 8.1% faster than MTP-3 and 3.6% faster than MTP-5.
  Mean accepted length continued to rise with MTP-5, but the extra accepted
  output did not recover the fifth draft's cost.
- **TypeScript:** MTP-5 was 17.0% faster than MTP-3 and 3.2% faster than MTP-4.
  Its mean accepted length of 4.73 shows that TypeScript's highly predictable
  syntax and repeated structure benefit from deeper speculation.
- **Rust:** MTP-5 led MTP-4 by only 2.0%. This is too small to justify a
  separate profile without additional samples.
- **SQL:** MTP-3 was 4.2% faster than MTP-4 and 3.8% faster than MTP-5. The
  rising mean accepted length did not compensate for deeper verification.
- Acceptance percentages naturally declined as the draft depth increased
  because later positions are harder to predict. Mean accepted length and
  measured decode throughput are more useful than acceptance percentage alone
  when selecting the depth.
- The results demonstrate that the best MTP depth is workload-dependent. A
  higher mean accepted length is useful only when its additional emitted tokens
  exceed the cost of generating and verifying a deeper draft.

## Output-equivalence caveat

Runs within each configuration produced stable response hashes, but hashes
differed between MTP depths in both the general and code benchmarks. These
benchmarks therefore compare the same inputs, sampling parameters, and fixed
output-token counts, but not byte-identical generated text across all
configurations.

Small floating-point differences between single-token decoding and multi-token
verification can change a close greedy token choice and lead to a different
continuation. Fixed-length outputs still make these useful performance
comparisons, but they are not strict output-equivalence results. Code should be
compiled and tested separately if quality or exact reproducibility matters.

## Conclusion

No single MTP depth is optimal for every workload. Use these profiles based on
the current measurements:

- **General-purpose and prose:** MTP-3. It reached 19.78 tok/s and was 83.7%
  faster than disabling MTP in the controlled general benchmark.
- **Unknown or mixed code:** MTP-4. It provides nearly all of MTP-5's aggregate
  code throughput, lower aggregate TTFT, and a better compute/performance
  balance.
- **Python:** MTP-4.
- **TypeScript:** MTP-5, if maintaining a language-specific profile is useful.
- **Rust:** MTP-4 as the practical default; the measured MTP-5 advantage was
  only 2.0%.
- **SQL:** MTP-3.

The general MTP-3 versus MTP-4 difference was 3.2%, and several per-language
differences were similarly small. Before maintaining multiple production
profiles, confirm them with 10–20 measured requests per task and alternate the
configuration order to reduce warm-state or thermal bias.
