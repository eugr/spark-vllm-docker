# Evaluating a local LLM deployment

Notes from bringing up GLM-5.2-NVFP4 on an 8x DGX Spark cluster. The scripts
in this directory are the actual tests used; they assume an OpenAI-compatible
endpoint at `localhost:8000` serving a model named `GLM-5.2-NVFP4` — edit the
constants for your setup.

## The layers

Run cheap tests often and expensive tests rarely. Each layer catches what the
one below it can't.

| Layer | Cost | Script | Catches |
|---|---|---|---|
| 1. Canaries | seconds | part of `accuracy_probe.py` | garbage output, dead server |
| 2. Needles | minutes | `accuracy_probe.py`, `deep_context_test.py` | broken long-context attention |
| 3. Stability | 20-60 min | `burnin_test.py`, `soak_test.py` | hangs, leaks, concurrency bugs |
| 4. Benchmark | hours | lm-eval GSM8K | subtle reasoning damage |
| 5. Your real tasks | ongoing | — | everything the above misses |

After a config change: run layers 1-2 always, layer 3 if the change touches
kernels/scheduling/parallelism, layer 4 if it could plausibly affect what
tokens get generated (quantization, attention, sampling — not, say, a port
number).

## What each script does

- `ladder_probe.py` — prompts of increasing size (50 to 1500 tokens). Exists
  because our cluster used to hang on any prompt over ~64 tokens; the ladder
  finds size-dependent failures and their boundary.
- `accuracy_probe.py` — three needles at 10/50/90% depth of a 12.5K-token
  document, plus greedy math/string canaries (127*43, Fibonacci, reverse a
  word). Wrong answers here mean the stack is corrupting output; exits
  nonzero so it can gate automation.
- `speed_probe.py` — 3x 400-token generations, reports tok/s. Run before and
  after any perf change; keep the numbers.
- `burnin_test.py` — 10 short requests, a 20K-token needle, 2 post-checks.
  The 10-request count is deliberate: one bug we chased only appeared after
  the 5th-6th request.
- `soak_test.py` — 60 mixed requests including 4-way concurrent bursts and
  long contexts. Concurrency exercises mixed prefill/decode batches, which
  is where scheduler and cudagraph bugs live.
- `deep_context_test.py` — plants 5 related facts across a ~320K-token
  document, asks for a briefing that uses all of them (synthesis, not just
  retrieval), then a cached-prefix follow-up to measure decode speed at
  depth. Read the output yourself; string checks only catch the worst.

## Things we learned the hard way

**Measure on your own stack.** The model's published GSM8K was ~95.5% — on
FP8, datacenter GPUs, a different harness. Our NVFP4 deployment measures
92.7%. Neither number is wrong; they aren't comparable. The only baseline
that matters is the one measured on your hardware with your harness.

**Only compare same-harness numbers.** Chat template vs completion format,
few-shot count, answer-extraction regex, thinking-token budget
(`max_gen_toks` — set it high for reasoning models or they get cut off
mid-thought and score near zero), temperature: each moves the score by a
point or more.

**Respect the error bars.** 1319 samples gives roughly +/-0.7%. Our
before/after MTP scores (93.1 vs 92.7) differ by less than the noise: that's
"no change", not "regression". A 250-problem subset has +/-1.1% or worse —
fine for smoke, useless for fine comparisons.

**Save per-sample logs** (`lm_eval --log_samples`). A percentage tells you
something changed; a per-problem diff tells you what. Also read a few
failures: "model misread the question" vs "harness couldn't parse a correct
answer" are very different problems with the same score impact.

**Greedy (temperature 0) for regression testing.** Deterministic-ish output
makes before/after comparisons meaningful. It also makes speculative
decoding exactly losseless, which is worth verifying rather than assuming —
we did, twice.

**Benchmarks are regression detectors, not leaderboards.** GSM8K is
saturated; frontier models cluster at 95-97 and good small models hit 85.
Its value for a deployment is sensitivity: a broken speculative-decoding
setup scores 1%, a broken quant path scores 60%. 92.7% says "nothing is
broken", full stop.

**Design tests to fail loudly.** Every script here exits nonzero on failure
and prints what it expected. Silent truncation or a "pass" that didn't
actually check anything is worse than no test.

**Budget real time.** One full GSM8K pass at 4-way concurrency on this
cluster: about 6 hours, ~3M generated tokens. That's one number for one
config. This is why you smoke-test before you benchmark, and why you don't
re-run the expensive layer for changes that can't affect it.
