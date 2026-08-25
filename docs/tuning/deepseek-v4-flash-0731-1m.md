# DeepSeek-V4-Flash-0731 @ 1M — measurement ledger

Evidence, tuning history, and traps for `recipes/deepseek-v4-flash-0731-1m.yaml`.
The recipe header carries only the deviation list and one-line rationale; every
number behind it lives here. Append new campaigns to this file — dated, with the
config they ran against — rather than growing the recipe header.

## Why the recipe deviates from official

The official DGX Spark profile (recipes.vllm.ai, hardware=dgx_spark_gb10,
nodes=2, pinned to eugr/spark-vllm-b12x:latest — the image the recipe runs)
omits `--max-model-len` entirely, so vLLM takes the checkpoint's own
max_position_embeddings of 1,048,576. It cannot boot that way on 2x Spark:

    ValueError: To serve at least one request with the model's max seq len
    (1048576), 11.04 GiB KV cache is needed, which is larger than the available
    KV cache memory (10.33 GiB).

Measured 2026-08-22, two boots at official's util 0.85 / max_num_seqs 8:
10.33 and 10.61 GiB available against 11.04 needed — short by 4–6%. The
arithmetic on a 121.69 GiB node: 0.85 x 121.69 = 103.4 GiB budget, minus
81.34 GiB of weights, minus ~11.5 GiB of activations, non-torch, CUDA graphs
and indexer/draft scratch, leaves ~10.6 GiB. vLLM's own log notes that CUDA
graph profiling makes 0.85 "equivalent to 0.84", which is most of the gap.

So the recipe raises gpu_memory_utilization 0.85 -> 0.90 and pins
max_model_len to the 1,048,576 official intends implicitly.

| Config | KV available | Outcome |
| --- | --- | --- |
| util 0.85, seqs 8 | 10.33 / 10.61 GiB | fails at init |
| util 0.90, seqs 8 | 19.22 GiB | 1.72x at 1,048,576 — adopted |
| util 0.90, seqs 4 | 17.07 / 19.37 GiB | 1.70x / 1.89x |
| util 0.95, seqs 8 | — | refused before allocating anything |

0.95 is not available on this hardware. vLLM gates on free memory at startup
(free >= util x total) and sees ~111 GiB free while the loader is staging
weights, against 115.61 GiB wanted. Dropping the page cache does not help:
with mods/drop-caches running on both nodes and Cached down to 1.4 GiB, the
gate still read 111.04 GiB. An idle box reports 117 GiB free, but that is not
the moment vLLM measures. The real ceiling is ~0.912, so 0.90 sits near the
top of the usable band with ~1.5 GiB of slack.

### VLLM_MOE_SKIP_PADDING

The base recipe sets it to "0"; this recipe does not set it at all. Measured
A/B 2026-08-22, both sides verified by reading /proc/<serving-pid>/environ
and resolving vllm.envs rather than trusting the export: off gave 11/11
graded and 57.11 tok/s mean over 15 runs, on (the library default, which
official uses) gave 11/11 and 58.11. The 1.8% sits inside the per-family
spread — chat moved 42–62 -> 39–50 while repetitive moved 54–69 -> 71–78,
opposite directions on the same change — and the only digest differences were
'Yes' vs 'yes' and '53,59,61' vs '53, 59, 61' on equally graded probes. The
flag is neutral here and the default is what official ships. eugr's reason
for setting it on the base recipe is undocumented and may still hold there;
this only speaks for this recipe.

### mods/instanttensor-hybrid-draft-loader

Inherited from the base recipe. It patches the model loader so selected
drafts load via lazy safetensors while target loads stay on InstantTensor.
Official runs --load-format instanttensor without it, so it is a candidate
for removal, but it is a source patch affecting DSpark draft loading and has
not been tested off.

## Why pin the window at all

The base recipe uses max_model_len: auto, which vLLM turns into -1 and
resolves with _auto_fit_max_model_len() — a binary search for the longest
sequence that fits the pool. That is a documented vLLM tuning option, not a
mistake, and it guarantees the advertised window is servable. The cost is
that the window becomes whatever memory was free at boot; six boots gave
995328, 987136, 940288, 926720, 921856 and 833792. This recipe takes the
other trade: fix the window at the checkpoint's real ceiling (YaRN factor 16
over original_max_position_embeddings 65536 = 1,048,576) and buy the memory
to honour it, so a shortfall is a boot failure instead of a shorter context.

## Measured behaviour (2026-08-22, TP=2, DSpark k=5, fp8_ds_mla KV)

Boots are reproducible; the pool still varies with host state, but the window
no longer does: 1,779,018 / 1,977,584 / 1,805,091 tokens across three boots,
every one advertising 1,048,576.

Decode, single stream, 1024 tokens, 3 runs per family (mean of 15: 57.5 tok/s):

| Family | tok/s | accept_len |
| --- | --- | --- |
| prose | 38.3 – 40.9 | 2.56 – 2.72 |
| chat | 36.7 – 68.9 | 2.50 – 4.97 |
| code | 48.1 – 72.6 | 3.27 – 5.17 |
| repetitive | 68.8 – 77.3 | 4.80 – 5.61 |
| peak | 64.1 – 66.9 | 4.49 – 4.78 |

Spread inside a family is acceptance, not noise: identical code prompts ran
48 and 73 tok/s. Quote a range, never a point.

Decode does NOT decay with depth, which is the headline result here:

| Depth | tok/s | TTFT | accept_len |
| --- | --- | --- | --- |
| 870 tok | 65.97 | 0.9 s | 4.68 |
| 56,190 | 36.70 | 24.1 s | 2.49 |
| 224,706 | 43.09 | 139.0 s | 2.98 |
| 928,582 | 48.43 | 980.2 s | 3.12 |

48 tok/s at 928k with acceptance above 3 means the DSpark drafter still earns
its passes 14x past the native 65,536-token YaRN window. That is the opposite
of the Qwen3.8 MTP cliff (qwen3.8-27b-nvfp4-1m.yaml, where acceptance goes to
exactly zero past native), so do not carry the "drop speculation for long
context" conclusion across model families.

The context is usable, not merely allocatable: a needle planted 2% into a
~1.04M-token prompt came back verbatim, unaided, and the same at 449k. What
1M costs is TTFT — 974–980 s to first token. Prefill peaks mid-range and
decays: 411 tok/s at 6.9k, 888 at 27k, 2,147 at 86k, 1,929 at 225k, ~818 near
1M. This is a batch context, not an interactive one, and the pool holds one
full-length request plus part of a second, whatever max_num_seqs says.

Concurrency (512 tokens): c1 30.3 tok/s, c2 28.4 aggregate / 20.0 per stream,
c4 59.4 / 23.1. tools/quality-probe.py scores 10/11 graded with static YaRN
always on. (See the 2026-08-25 fleet campaign below before reading the c2
number as "aggregate is flat": under a real multi-agent load with proper
scheduling, batch amortization is measurable.)

## Benchmark on /v1/chat/completions, not /v1/completions

The 2026-08-22 harness used /v1/completions, which gives this thinking model
no chat template, and it produced three spurious "empty completion" results
that cost an afternoon. With no template the model believes it is mid-<think>,
and its first token is a near-tie: at 449k the top-5 was EOS -0.663, ' Do'
-1.288, '</think>' -2.163, so greedy picked EOS and returned zero tokens with
200 OK. At ~1.04M the same prompt put '</think>' first and answered
correctly. Forcing past that one token (min_tokens=32 or ignore_eos)
retrieved the needle every time. Nothing to do with depth, sparse attention,
or the recipe — treat any future empty completion on /v1/completions as a
harness artifact until reproduced on the chat endpoint.

## NVFP4 KV is not an option — do not retry it

Tested 2026-08-22: --kv-cache-dtype nvfp4_ds_mla dies in ~2 s at config
validation, before weights load:

    ValueError: nvfp4 KV cache is not supported with MLA (Multi-head Latent
    Attention) backends.

The guard is VllmConfig.validate_nvfp4_kv_cache_with_mla —
cache_dtype.startswith("nvfp4") and model_config.use_mla — so it rejects
nvfp4, nvfp4_4over6 and nvfp4_ds_mla for every MLA model. It exists because
vllm-project/vllm#43562 had NVFP4 KV boot clean on SM120 and then kill the
engine on the first request. The same error is reported for GLM-5.2-NVFP4
(HF discussions/1), so the 432 B/token nvfp4_ds_mla layout in
B12xMLASparseBackend is unreachable for any MLA model here. fp8_ds_mla at
584 B/token is the floor, and util plus max_num_seqs are the only levers.

Third-party recipes advertising "1M NVFP4 KV" for this checkpoint delete that
validator and then pin the page back to fp8's 584-byte envelope, so their KV
is byte-identical to this recipe's under a different dtype string. Their own
notes record the real 416-byte layout failing past ~411 tokens.

## Tuning already tried — do not re-run these

Measured 2026-08-22/23, decode suite at --repeats 10 (50 runs per config,
five prompt families x ten) with a matched same-session baseline, because at
--repeats 5 four of five families sat inside the noise floor. The noisy
families carry sd ~13 tok/s (~30% of mean, acceptance-driven), so t is
quoted; |t| < 2 is unresolved, not a result. Baseline itself drifted 58.7 ->
60.3 between campaigns, so only within-campaign comparisons count.

- **num_speculative_tokens 7** (the model card's value, vs 5 here and in the
  official recipe): REJECTED. prose 44.0 -> 34.5 tok/s (-22%), the other four
  families inside noise. Mechanism measured, not inferred: draft positions 5
  and 6 accept at 0.012 / 0.005 on prose against 0.81 / 0.65 on code, so on
  unpredictable text the two extra drafter passes are pure cost. Code's
  ceiling does rise (74.7 -> 85.0, accept_len reaching the full 7.00) but its
  mean barely moves, so the peak is not what a workload gets.

- **--enable-chunked-prefill**: REDUNDANT. Already the default in this build
  (SchedulerConfig.enable_chunked_prefill = True); every boot logs
  enable_chunked_prefill=True with or without the flag, so the "baseline vs
  chunked" runs compared two identical configs. Everything they appeared to
  show was variance, and that makes them the most useful measurement here:
  between IDENTICAL configs, KV available ranged 16.9–19.6 GiB, prefill at
  32k ranged 1898–2328 tok/s, and peak-family decode differed by 0.8%
  (t=+0.44, n=20 each). Any effect smaller than that is unmeasurable on this
  box, whatever the flag.

- **--async-scheduling**: NO MEASURABLE EFFECT (t=0.09 over 50 runs per side,
  and exactly 0.0% on the peak family), and most likely redundant — but the
  evidence is weaker than for chunked prefill. The engine config dump prints
  enable_chunked_prefill but NOT async_scheduling, so its resolved value was
  never observed. What is established: the field defaults to None
  (config/scheduler.py), vllm/config/vllm.py resolves None through five
  disable branches to `else: async_scheduling = True`, dspark is explicitly
  on the compatible list, the mp executor returns
  supports_async_scheduling()=True, and every disable branch emits a warning
  that no boot logged. That chain says on-by-default; it is inference from
  source plus absent warnings, not a read value. To settle it, boot with
  --no-async-scheduling and diff against the default.

- **max_num_batched_tokens 16384**: WILL NOT BOOT at this window. The KV
  requirement rises 11.04 -> 17.85 GiB while availability falls 19.22 ->
  12.09, because larger chunks inflate both the per-request reservation and
  peak activation. Not a throughput question.

- **max_num_batched_tokens 4096**: TRADEOFF — ADOPTED 2026-08-25 (evening),
  and the largest lever found. KV pool 1.70M -> 2.89M tokens, 1.62x -> 2.76x
  at a full window. Costs decode: -7.7% overall (t=-1.60) and chat -23.1%
  (t=-2.56, the only per-family regression here besides k=7's prose).
  Quality 11/11. Adoption rationale and trigger data in the fleet campaign
  below.

NOTHING from the 2026-08-22/23 campaign was adopted. Two of the four levers
turned out to be defaults already in force, and the two that genuinely
changed behaviour both regressed. Check a flag's default before benchmarking
it — both redundant tests produced plausible-looking numbers that were pure
variance, and the sign even flipped between two runs of the same config.

## Fleet scheduling campaign (2026-08-25) — ADOPTED: seqs 12, capture 128

A coding-agent orchestrator fanning out ~12 concurrent sessions at 44–90k
tokens each, every session resubmitting its grown prompt each turn, collapsed
the seqs-8 scheduling. Measured live on this recipe, /metrics counters, 12
in-flight agents:

- num_requests_running pinned at 8, num_requests_waiting at 4, for hours.
- Queue time per request: p50 0.18 s, mean 15.8 s, p90 77.8 s (n=1426).
- One agent went 75 minutes between completed turns while others cycled.
- Prefix cache hit rate: 94% lifetime, 12% over the final 80-minute window.
- In that window the cluster prefilled 16.5M tokens (~3,500 tok/s,
  continuously) while generating 75k output tokens (~16 tok/s aggregate).
- kv_cache_usage 41% -> 63% across the run; num_preemptions stayed 0.

The mechanism is an eviction cycle, not preemption. Running requests' KV is
live; the 4 queued agents' cached prefixes are idle blocks — LRU fodder. As
running contexts grow, the queued prefixes are evicted; when a starved agent
finally gets a slot, its full 60–90k re-prefill floods the pool and evicts
the next idle agent's prefix. Repeat: the box spends ~99% of its compute
recomputing prefills it already did. Queued-at-server is the poisonous state,
so the fix is to stop queueing the fleet:

- **max_num_seqs 8 -> 12.** Every concurrent session admitted means every
  working set stays pinned (12 x ~85k = ~1.0M tokens, inside the 1.70M pool).
  Batch decode also amortizes the per-step weight read — measured elsewhere
  on this checkpoint, seqs 2 -> 8 nearly doubled aggregate decode — so this
  buys throughput as well as fairness. Sizing rule: healthy seqs ~= pool
  tokens / average session context with ~30% headroom; at 1.70M that is 12
  sessions of ~100k. Past that, adopt max_num_batched_tokens 4096 (the pool
  lever above) before raising seqs further — admitted sequences beyond what
  the pool can pin just recreate the eviction cycle with extra steps.

- **max_cudagraph_capture_size 64 -> 128.** Not independent: the dspark k=5
  drafter makes a decode step carry ~6 tokens per running sequence, so
  8 x 6 = 48 fit official's 64 but 12 x 6 = 72 do not, and decode would fall
  off the captured CUDA graphs exactly when the new slots are in use. 128
  covers ~21 running sequences.

Boot caveat: this pair has not been A/B-booted at the 1M window — capture
memory grows with graph count and the profiler may reprice the KV pool. If
init fails on KV arithmetic, drop capture back to 64 first (keeps the thrash
fix, costs graph coverage above 10 seqs) and record the numbers here.
Client-facing behaviour is unchanged: same model, same window, same served
name — scheduling does not change the math.

### Post-change measurement (2026-08-25 afternoon)

Server config as adopted (seqs 12, capture 128, batched 8192 unchanged; boot
pool 1,548,133 tokens) plus a client-side spawn cap (OMP
`task.maxConcurrency: 4`; agent contexts ~130k by this point, so 4-5
concurrent sessions ~= 650k working set, well inside the pool).

20-minute window, 60 s samples, fleet of 4 reviewers + parent (+1 unrelated
single-request session):

- generation: mean 77.0 tok/s, peak 105.1 (best of any regime measured);
  minutes 14-18 at width 4-5 sustained 89-105 tok/s
- prefix cache hit rate: mean 95.5%; replicate window mean 99.1%, min 91.2%
- computed prefill: 283 tok/s average, bursty (the reviewers' legitimate
  cold starts, absorbed without a throughput dent)
- waiting: 0 in every sample; preemptions: 0; KV usage peaked at 41%

Same day, same server config, fleet of 12-15 sessions (working set 1.5-1.9M
tokens > pool): hit rate 0-7%, generation 3-9 tok/s, 1,600-3,900 tok/s of
continuous re-prefill, agents 26-75 minutes between completed turns.
Preemptions stayed 0 throughout - the mechanism is LRU eviction of idle
sessions' cached prefixes, not overflow preemption, and it is binary: no
middle regime was ever observed.

Conclusions recorded:

- seqs 12 did its job: at width 12 the engine reached 12 running / 0 waiting
  (the old config pinned at 8/4) and briefly hit 47 tok/s aggregate before
  cache churn caught up. Scheduling is no longer the constraint.
- Throughput is governed entirely by cache fit: `sum(session contexts) <=
  pool` is the operating invariant. Size client fan-out to
  `pool / avg context` with ~30% headroom (4-6 sessions at 130k on this
  pool). Aggregate throughput saturates by width ~4-5 anyway (~20 tok/s per
  active stream), so wider fleets buy little even when they fit.
- OMP caveat: `task.maxConcurrency` gates fresh spawns only; process
  restarts cold-revive parked subagents around the semaphore and can
  stampede the pool (observed: 11 revived at once under cap 4).
- max_num_batched_tokens 4096 (the pool lever above) remains NOT adopted -
  owner decision; it is the knob to revisit only if fleets wider than the
  sizing rule allows become a requirement.

### max_num_batched_tokens 4096 adoption (2026-08-25 evening)

Adopted after the post-change measurement, on measured demand rather than
speculation. Context percentiles across 514 subagent requests that afternoon:
p50 104,517 / p95 193,419 / p99 203,539 / max 214,675 tokens - long-lived
agents drift to ~200k, so seats must be sized by the tail, not the median.
The operator's normal pattern includes two concurrent orchestrators (2 x
(parent + 4 subagents) = 10 sessions) plus ad-hoc side conversations; at
tail-sized contexts that is ~1.7M+ tokens of working set against the 1.55M
pool - the eviction cycle's exact trigger condition, observed live twice
that day. The 2.89M pool holds two orchestrators plus side sessions with
~30% headroom (thrash threshold moves from ~9 to ~15 concurrent sessions).
The ~8% decode cost was accepted knowingly.

Scope note: this adoption is for TP=2. When the cluster moves to TP=4
(planned), re-run the A/B between 8192 and 4096 there - expectation is 8192
wins at TP=4, whose ~6x pool makes seat count a non-issue, restoring the
decode edge as the deciding factor.

Client-side pairing (OMP): task.maxConcurrency 4 for two-orchestrator days,
6 for single-orchestrator days; revive-bypass caveat above still applies.

## Known risk

vllm-project/vllm#40969 (open) reports DeepSeek-V4-Flash hanging after ~6
requests with cudagraph_mode=FULL_AND_PIECEWISE plus chunked prefill on
SM 12.x, which is what this recipe runs. Not reproduced here across ~100
requests and five boots including four 200k+ prefills, but it is the
likeliest explanation for third-party reports of this stack dying after 1–2
hours.
