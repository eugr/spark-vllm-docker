# Long-context recipe — vLLM + NVFP4 + MTP

**For chat, reasoning, and anything that reads long documents.**

This is the even performer. It runs at roughly the same speed whatever you ask it, reads long
documents about five times faster than the other recipe, and does not slow down as the context
fills up. It is *not* the one to use if a coding agent is rewriting whole files — see
[../llamacpp-edit](../llamacpp-edit/) for that.

## Measured

| | |
|---|---|
| Free-form prose | **32.2 tok/s** |
| Rewriting a file with one change | 39.1 tok/s |
| Fixing a bug in a file | 35.0 tok/s |
| Adding a function | 33.6 tok/s |
| Spread across those four | **1.2×** (the edit recipe swings 3.2×) |
| Time to first token, short prompt | **~0.3 s** |
| Prefill | **~2,200–2,460 tok/s**, flat to 195k tokens (independently ~2,030–2,230, see below) |
| Vision | **works** — 0.967 on the atlas image eval, and 2.6× faster than the GGUF recipe, which also scores 0.967 |
| Decode at 1k / 32k / 128k context | 31.7 / 33.5 / 31.7 — **no falloff** |
| Concurrent requests | **16** served well; 64 possible for batch work |
| Aggregate decode, 16 concurrent | **96–109 tok/s**, TTFT under 2.7 s |

Full method and raw figures: [../../docs/measurements.md](../../docs/measurements.md).

## Independently measured

The figures above come from this repository's own `bench/portable_bench.py`. The same
configuration has since been run through the pinned workloads of
[inference-atlas](https://github.com/0xBakeer/inference-atlas), which is a different harness with
a different prompt set, and the numbers are close but not identical.

**Prefill**, same definition on both sides — input tokens divided by time to first token:

| context | this repo | inference-atlas |
|---|---:|---:|
| ~8k | — | 2,031 |
| ~20k / 32k | 2,463 | 2,230 |
| ~128k | 2,297 | 2,057 |

Flat in both, which is the claim that matters and the thing that distinguishes this recipe from
the editing one. The atlas run is 5–10% lower throughout. The harnesses differ in prompt content,
in warmup handling and in `max-num-seqs`, so this is a second opinion rather than a correction —
but the honest range across both is **~2,030–2,460**, and the lower half of that is what a
different harness saw.

**Accuracy**, from the atlas eval suite on this exact configuration, every workload at 100%
request completion:

| eval | accuracy |
|---|---:|
| format, knowledge, math | **1.000** |
| reasoning | 0.992 |
| tools | 0.988 |
| json | 0.982 |
| instruction | 0.971 |
| vision | 0.967 |
| multilingual | 0.950 |
| code | 0.829 |
| hallucination | 0.387 |

Two of those are worth pulling out. **`tools` at 0.988** exercises the
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` path this recipe ships, so the tool
configuration is not merely present but correct. **`hallucination` at 0.387** is a hard
benchmark that measures whether a model declines to answer what it does not know; a low score
there is normal and is not a fault of this configuration.

## Vision works here, and it is faster — but it is no longer exclusive

The NVFP4 checkpoint carries the full vision tower — **333 tensors, 448,931,056 parameters,
unquantized** — verified by tensor name in the shards actually served. It scores **0.967** on the
atlas image eval, 58 of 60.

This section used to say the other recipe could not do images at all. That was wrong. The GGUF
shards do contain none of the vision tower, but llama.cpp ships multimodal as a separate
projector and Unsloth publishes one in the same repository as the quants — so
[../llamacpp-edit](../llamacpp-edit/) fetches it in `setup` and scores **0.967 as well**, with
identical per-category splits.

What remains true is speed: the same 60 images take **233 s here against 598 s there**, and this
recipe generated 13% fewer tokens and used 2.5× less energy doing it. Some of that gap is
queueing — the eval runs at concurrency 4, which this recipe absorbs and two llama.cpp slots do
not. Full comparison: [../../docs/vision.md](../../docs/vision.md).

## Why it behaves this way

The model ships a **trained MTP head** — a small predictor that guesses the next few tokens.
vLLM runs it; llama.cpp cannot, because the GGUF converter drops the head (`supports_mtp_export
= False`). We confirmed this directly: our GGUF contains **zero** MTP tensors across all four
shards, the NVFP4 checkpoint contains **all 31**.

Because the head is trained rather than copying from your prompt, it works the same on any text.
Hence the flatness — and hence why this recipe wins on prose and loses on file rewriting.

The gain is real but modest: across engines, prose goes 27.8 → 32.2 (**~1.16×**). We still
have not run our own MTP-off A/B — the upstream recipe's in-engine one measured 17 → 27 tok/s
at MTP=2, so part of our cross-engine delta is the engine and quant, not the head. An
in-engine A/B on this hardware class now exists, but it is a third-party result: Jürgen
Schmied ran both arms on their own DGX Spark — same checkpoint, same harness, same day — in
[#6](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6). Their checkpoint is **not**
this recipe's — an NVFP4-FP8 variant with a locally requantized `lm_head`, not a public
artifact — and their vLLM build differs, so nothing in this table can be reproduced from this
repository. It relays their report; read the shape, not the numbers:

| *as reported in [#6](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6)* | 1 caller, decode tok/s | 16 concurrent, aggregate |
|---|---:|---:|
| MTP off | 26.4 *(single run)* | 96.6 |
| MTP k=2 | 35.7 *(mean of 6)* | 99.1 |
| | **+35%** | **inside the ~7% noise floor — not measurable** |

They report acceptance at k=2 of 56.6% on that workload (position 0: 67.3%, position 1:
46.0%). The right-hand column is their finding: **under load, speculation does not hurt — it
stops paying.** Their explanation — once the batch saturates the machine, accepted draft
tokens have no idle capacity to convert into throughput — matches the expert-union argument
below, and if it holds, a single "MTP is worth Nx" number is the wrong shape of claim: the
multiplier depends on how busy the box is. (Their TTFT at 16 concurrent also moved
9.71 s → 6.79 s, with variance not characterized the way decode's was, so treat that as
indicative only.)

Either way the gain is far from the 3–5× dense models see: on a top-10-of-512
mixture-of-experts, verifying *k* draft tokens touches the *union* of experts across those
positions, so speculation buys far less here than it does on a dense model. Raising `MTP`
above 3 makes it worse, not better.

## Install

```bash
./setup.sh      # builds the container, downloads ~126 GB
./serve.sh      # starts on http://localhost:8000/v1
```

`setup.sh` clones and builds [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
(Apache-2.0), whose patch serves the n-gram table from disk. That patch is the reason this fits at
all, and it stays in their repository rather than being copied into this one. See
[../../CREDITS.md](../../CREDITS.md).

First start reads ~83 GiB off disk and takes **12–15 minutes**.

## Settings

| variable | default | what it does |
|---|---|---|
| `PORT` | `8000` | host port |
| `BIND` | `127.0.0.1` | loopback only. `0.0.0.0` exposes it to your whole network |
| `CTX` | `262144` | context length. The full native window |
| `SEQS` | `16` | concurrent sequences. See *How many at once* below |
| `GPU_MEM` | `0.85` | share of the 128 GB pool for weights + KV |
| `MTP` | `3` | speculative tokens. `0` disables, for an A/B |
| `PREWARM` | `1` | stream the table at boot so the first request is not cold |

The upstream default is `GPU_MEM=0.78`, which leaves only 10.82 GiB of KV — **227,651 tokens,
less than the model's own context window**, so a single full-length request will not fit. We
raise it to 0.85, which measured 18.13 GiB of KV = **641,601 tokens**, comfortably 2.4× the
context, while still leaving ~19 GiB of headroom.

## How many at once

`SEQS` was `2`, the upstream container's default. This model and this recipe are both days
old, so there was no published figure to set it against. Two slots is a scheduler cap rather
than a memory limit: at `GPU_MEM=0.85` the server reports a KV pool of **654,635 tokens**,
while 64 concurrent ~1.3k-token requests need about **83,000**. It left roughly 87% of the
pool unused.

Measured with the pinned parallel sweeps from
[inference-atlas](https://github.com/0xBakeer/inference-atlas), 32 and 256 requests
respectively, every request completed:

| concurrent | tok/s each | aggregate | TTFT p50 | | tok/s each | aggregate | TTFT p50 |
|---:|---:|---:|---:|---|---:|---:|---:|
| | *512-token prompts* | | | | *1k-token prompts* | | |
| 1 | 27.70 | 27.7 | 0.68 s | | 25.01 | 25.0 | 1.23 s |
| 2 | 20.23 | 40.5 | 1.02 s | | 19.63 | 39.3 | 1.39 s |
| 4 | 15.06 | 60.2 | 1.10 s | | 14.04 | 56.2 | 1.48 s |
| 8 | 10.68 | 85.4 | 1.27 s | | 9.01 | 72.1 | 1.73 s |
| **16** | 6.80 | 108.8 | **2.15 s** | | 6.00 | 96.0 | **2.64 s** |
| 32 | 4.21 | 134.8 | 13.80 s | | 3.46 | 110.8 | 16.32 s |
| 64 | — | — | — | | 2.02 | 129.4 | 70.42 s |

**Aggregate throughput never plateaus** — it keeps climbing to 64, just with sharply
diminishing returns: 1→8 buys 2.9x, 8→16 buys 33%, 16→32 buys 15%, 32→64 buys 17%.

**Time to first token is where the wall is, and it is a cliff rather than a slope.** It sits
under 2.7 s all the way to 16 concurrent, then goes to 16 s at 32 and **70 s at 64**. Both
sweeps agree on where it breaks. The cause is `--max-num-batched-tokens 8192`: past a certain
number of simultaneous prefills, each one is chunked across many scheduler steps before its
first token appears.

So **16 is the last concurrency this configuration serves well**, at 96–109 tok/s aggregate,
which is already about 75% of everything the box will ever do. Going to 64 buys ~35% more
aggregate throughput and costs 27x on first-token latency.

The sweep varies offered concurrency against a 64-slot server, so it shows where the box
stops serving well — not what the `SEQS` cap itself costs. That direct A/B now exists too,
published in [inference-atlas](https://github.com/0xBakeer/inference-atlas): the same pinned
workloads, same box, one arm at `SEQS=2` and one at `SEQS=64`. Only `max-num-seqs` changes
between the columns:

| workload | concurrent | `SEQS=2` tok/s | `SEQS=64` tok/s | TTFT p50, 2 → 64 |
|---|---:|---:|---:|---:|
| serve-single | 1 | 30.95 | 30.89 | 0.5 s → 0.5 s |
| prefill-8k | 1 | 3.07 | 2.89 | 4.5 s → 4.5 s |
| prefill-32k | 1 | 0.84 | 0.86 | 18.1 s → 17.5 s |
| prefill-128k | 1 | 0.21 | 0.20 | 76.7 s → 77.8 s |
| serve-long-c4 | 4 | 28.09 | 34.15 (1.22×) | 42.9 s → 7.9 s |
| serve-chat-c8 | 8 | 43.36 | 81.41 (1.88×) | 36.3 s → **1.4 s** |
| serve-code-c8 | 8 | 41.32 | 78.57 (1.90×) | 150.8 s → **2.4 s** |
| serve-prefix-c16 | 16 | 27.72 | 46.50 (1.68×) | 127.7 s → 5.9 s |
| serve-short-c16 | 16 | 53.93 | 146.73 (**2.72×**) | 33.5 s → **1.2 s** |
| serve-chat-c32 | 32 | 42.98 | 105.00 (2.44×) | 178.8 s → 18.4 s |
| serve-chat-c64 | 64 | 42.90 | 105.48 (2.46×) | 294.9 s → 95.9 s |

(Cells are the atlas `output_tok_s` value. The prefill workloads emit almost no output tokens,
hence the small numbers; their prefill throughputs — 2,170 vs 2,031 tok/s at 8k, 2,176 vs 2,230
at 32k — also sit inside the single-run noise floor.)

**All four one-caller rows are identical.** The cap costs nothing when the server is quiet,
which is the whole reason raising it is safe. Every loaded row is 1.2–2.7× better with the cap
lifted, and the first-token column is where it really shows: eight callers sending 2k-token
prompts wait **150.8 seconds** for a first token at two slots against 2.4 s at sixty-four, a
63× difference.

**The 64-caller row understates the gap, and should be read with its footnote.** At `SEQS=2`
that cell completed only **513 of 640 requests** — the median caller waited 294.9 s for a first
token, which is the workload's own 300 s budget, so the tail simply ran out of time. It is the
only cell in either arm that lost requests. A configuration that fails a fifth of its callers is
not really delivering 42.9 tok/s, so treat 2.46× as a floor on the difference rather than a
measurement of it.

This is stronger evidence for raising the default than the sweep above, because it isolates the
cap: same workloads, same box, one variable.

We ship `SEQS=16` for that reason. It is a cap, not a batch size — with one caller it behaves
exactly like `SEQS=2`, so nothing is lost when the server is quiet, and a burst of up to 16 is
served without queueing. Beyond 16, callers queue rather than all being served badly: 64
requests through a 16-slot server finish in about 171 s against 127 s if batched 64-wide, but
the ones being served see 2.6 s to first token instead of 70 s.

**Set `SEQS=64` if you are running batch work** where nothing is waiting on a first token and
the extra ~35% aggregate throughput is what matters. That is also the configuration the atlas
cells for this recipe were measured at, deliberately, so that no cell is capped by the
scheduler.

## Prefix caching

**On by default since 2026-08-30.** It was off, and the reason this repository gave — "a GB10 GDN
kernel bug" — had no source next to it. [@faparicior](https://github.com/faparicior) reported it
working over a five-hour coding session at a 96% hit rate
([#9](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/9)), which is what prompted
measuring it instead of repeating the claim.

Correctness first. Three identical requests at temperature 0 over an 8.6k-token prompt returned
byte-identical, correct answers, with a real cache hit behind calls 2 and 3, and `eval-format-v1`
scored **30/30** with caching on, matching the cache-free cell.

**That probe was the wrong observable, and this section said more than it should have.** It also
reported the same result with the image's `block_size` patch reverted, and concluded "nothing
here depends on that patch". [@blazux](https://github.com/blazux), who diagnosed the underlying
bug, explained why that does not follow —
[qwen3.8-Flash-DGX#5](https://github.com/blazux/qwen3.8-Flash-DGX/issues/5):

- The image has a **third** KV group beyond the attention and Mamba ones. The QSA raw-key ring
  is a `CircularBufferSpec` (`vllm/models/qwen3_8_flash_next/common/qsa_cache.py`) whose block is
  its ring capacity, `compress_ratio * cdiv(compress_ratio + num_speculative_tokens,
  compress_ratio)` — with this model's `indexer_compress_ratio = 4` that is **8 at `MTP=3`**, and
  4 with speculation off. So `EngineCore`'s `min()` over the groups returns 8, not the 1,600 the
  page alignment produced, and the mismatch the diagnosis rests on **is** present here. Confirmed
  by reading the class out of the image this recipe ships.
- Identical outputs do not clear it. With the split bug a cold request almost never ends a chunk
  on a 1,600 boundary, so it publishes no Mamba block at all; the repeat then takes an
  *attention-only* hit and recomputes the recurrent state from scratch — correct output, no
  guard hits. Our 49.3% hit rate was exactly that. Reaching the zero-state restore needs a Mamba
  block to have been published first, which depends on scheduling.

**Upstream `main` is not affected — but that is a statement about today, not about vLLM.**
`CircularBufferSpec` is not on `main`, so its `min()` is 1,600 and the same lines are harmless
there. It arrives with
[vllm-project/vllm#53896](https://github.com/vllm-project/vllm/pull/53896) ("[Model] Support
Qwen3.8-Flash-Next", open against `main`), which the official Flash-Next image is built from —
and that PR adds `class CircularBufferSpec` to `vllm/v1/kv_cache_interface.py` while also
touching both consumers, `vllm/v1/core/sched/scheduler.py` and
`vllm/v1/worker/gpu/model_states/mamba_hybrid.py`. So the defect is upstream's, on a release
branch rather than on `main`, and it lands when that branch does.

That is [@blazux](https://github.com/blazux)'s correction, verified here against the PR's own
file list. It is also the third time this investigation has had to narrow a claim: first "a GB10
GDN kernel bug", then "the mismatch does not arise here", then "upstream is not affected". The
report belongs on that PR, and its author is reporting it there.

For running it, the practical rule is simple: the fix is in the container from `8347e7c`
(2026-08-29) onward, `serve.sh` prints the upstream sha it was built from, and caching is on by
default because the image this recipe builds today carries the patch. **Do not enable it on an
image built before that commit.**

Then the benefit, on `serve-prefix-c16-v1` — 192 requests, 16 concurrent, grouped by shared
prefix — same box, same `SEQS=64`, only the flag differing:

| | caching off | caching on |
|---|---:|---:|
| Aggregate decode | 46.50 tok/s | **81.79** (1.76×) |
| Prefill | 16,453 tok/s | **30,728** (1.87×) |
| Time to first token, p50 | 5.86 s | **2.55 s** |
| Time to first token, p90 | 12.16 s | **5.28 s** |
| Wall clock | 1,020.9 s | **573.7 s** |
| Requests completed | 192/192 | 192/192 |

Server-reported hit rate over that run: **66.5%**.

Two caveats worth carrying:

- **Every prefill number published in this repository was measured cache-free**, including the
  ~2,030–2,460 tok/s range quoted above. They are not comparable with a cache-assisted run, and
  the 30,728 tok/s figure in that table is a *cache-assisted* prefill on a workload built out of
  shared prefixes — it is not a prefill speed. `PREFIX_CACHE=0` restores the measured
  configuration.
- **The benefit is entirely about repetition.** Multi-turn chat, an agent loop re-sending its
  scratchpad, several requests behind one system prompt. A workload of unrelated prompts gains
  nothing, and pays a little bookkeeping.

## Turn thinking off

Thinking is on by default and **86% of generated tokens were reasoning** in our measurements. The
same prompt answered in **15.0 s with thinking off against 55.1 s with it on** — not because
tokens got faster, but because there were a quarter as many.

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Known issues

- **Prefix caching is on** (`PREFIX_CACHE=1`), and it used to be off here for a reason we could
  not substantiate — see [Prefix caching](#prefix-caching). Set `PREFIX_CACHE=0` to
  reproduce any prefill figure published in this repository — all of them were measured
  cache-free.
- **Full `torch.compile` is off** — an Inductor int64-indexing assert on sm_121.
- **The n-gram gather must stay outside CUDA graphs.** `serve.sh` declares it a splitting op and
  captures `PIECEWISE`. Do not switch to a `FULL` capture mode.
- **`VLLM_TORCH_PROFILER_DIR` is inert in this build.** Profiling moved to a CLI flag; the env
  var registers no routes and prints no warning, so `/start_profile` just 404s. Working form:
  `--profiler-config '{"profiler":"torch","torch_profiler_dir":"/abs/path"}'` — the profile
  router only attaches when `profiler_config.profiler` is non-null
  (`entrypoints/serve/profile/api_router.py: attach_router`). If you launch from a systemd
  unit, note that `Environment=` strips the JSON's quotes; build the value inside the script
  instead of passing it through the unit file. (Reported in
  [#6](https://github.com/0xBakeer/qwen38-flash-next-spark/issues/6); verified against this
  build — the env var is absent from `envs.py`, and `--profiler-config` is accepted where
  `--torch-profiler-dir` is not.)
- **`VLLM_PLE_CPU_OFFLOAD=1` hangs.** The official pinned-host-RAM path registers a
  `PleOffloadLayer` and then spins a core with no disk I/O, indefinitely — it expects an offload
  worker this image does not launch. Only `VLLM_PLE_MMAP` works here.
- **Sixteen sequences, not two hundred.** A chat UI, a coding agent and a background job can
  all point at this endpoint without queueing. This is still not a multi-user server: past 16
  concurrent requests time to first token collapses from under 2.7 s to 16 s at 32 and 70 s
  at 64.

## Verifying it works

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next",
       "messages":[{"role":"user","content":"Reply with exactly: ok"}],
       "max_tokens":50,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

Then measure it yourself:

```bash
../../bench/portable_bench.py --api http://127.0.0.1:8000 --label mine
```
