# EAGLE Fine Prefix Fix

**Last updated:** `2026-07-23T18:37:36-07:00`

This runtime-only mod restores fine-grained hybrid prefix-cache hits when MTP
or EAGLE is enabled on full-attention + Mamba models such as Qwen 3.5/3.6.
It targets current vLLM after
[#46384](https://github.com/vllm-project/vllm/pull/46384), based on
`41ea2dd44a3a20c46ebeb985de0022c7673fb953`.

## Why it helps

PR #46384 registers only the final prompt-tail hash at `prefix_match_unit`
granularity. A context-load probe such as llama-benchy's `"."` can occupy that
last hash unit, so a follow-up user turn misses the entry even though the long
system context is identical. Under EAGLE, full attention then rewinds the next
available match, while Mamba falls back to the previous physical state block
(2,144 tokens for this Qwen model).

This mod also retains the predecessor FullAttention hash. Since EAGLE rewinds
that match by one more hash unit, the scheduler materializes and caches the
Mamba state at the resulting replay boundary. The extra metadata/state cost is
constant per prompt rather than changing physical allocation geometry.

## Exact scope

This is a configurable bounded-tail heuristic, not a general arbitrary-fork
checkpointing policy. For a request containing `N` prompt tokens and match
unit `H`, the mod retains the final hash boundary and one predecessor:

```text
tail = floor(N / H) * H
retained FullAttention boundaries = tail, tail - H
retained Mamba replay boundary     = tail - 2H
```

Suppose request A and request B share `S` tokens and then diverge. Request A's
predecessor entry can match request B only when:

```text
tail(A) - H <= S
```

Equivalently, if A has `U = N - S` unique tail tokens:

```text
U < 2H - (S mod H)
```

With `--prefix-match-unit 16`, A's complete tokenized tail after the fork must
therefore be at most 16 to 31 tokens, depending on alignment. Chat-template
role markers, generation headers, and control tokens count toward this tail.
EAGLE then rewinds the matching predecessor by one further match unit to the
Mamba replay state.

In general, unit `H` guarantees coverage when A's unique tail is at most `H`
tokens. Favorable alignment extends coverage as far as `2H - 1` tokens.

## Choosing `prefix_match_unit`

Every KV cache group's physical `block_size` must be divisible by
`prefix_match_unit`. For the tested Qwen 3.6 FP8-cache configuration, the
physical block size is 2,144 tokens:

```text
2144 = 32 * 67
```

This makes 134, 268, and 536 valid larger units. The nearby powers of two 128,
256, and 512 are invalid for this cache geometry. Recalculate valid divisors
when using another model, KV cache dtype, or parallel configuration.

A larger unit increases the producer-tail length that can differ without an
explicit fork checkpoint, but also moves the reusable EAGLE replay state
farther behind the shared prefix:

| Match unit | Alignment-dependent tail coverage | Maximum replay behind fork |
| ---: | ---: | ---: |
| 16 | 16-31 tokens | 47 tokens |
| 134 | 134-267 tokens | 401 tokens |
| 268 | 268-535 tokens | 803 tokens |
| 536 | 536-1,071 tokens | 1,607 tokens |

The following three-run results used regular `vllm-node`, the Qwen 3.6 NVFP4
recipe, `pp=2048`, `tg=256`, and context depths 8K, 16K, and 32K. Each value is
cached follow-up prefill throughput in tokens per second:

| Configuration | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| No mod, unit 16 | 1,867 | 1,713 | 1,495 |
| Mod, unit 16 | **4,098** | **3,505** | **2,795** |
| Mod, unit 134 | 3,751 | 3,190 | 2,524 |
| Mod, unit 268 | 3,252 | 2,853 | 2,325 |
| Mod, unit 536 | 2,926 | 2,409 | 2,258 |

llama-benchy's one-token context probe favors unit 16. Relative to unit 16,
unit 134 was 9-10% slower, unit 268 was 17-21% slower, and unit 536 was 19-31%
slower on cached follow-up prefill. All larger units still outperformed the
no-mod result:

- 134: 69-101% faster;
- 268: 56-74% faster;
- 536: 41-57% faster.

Depth-zero prefill was 5,494 tokens/s without the mod and 4,747-4,884
tokens/s across the tested mod settings, an 11-14% regression. Uncached context
loading changed by 0-4%, generation throughput showed no consistent trend, and
all benchmark coherence checks passed.

Recommended starting points:

- Use **16** for tiny changed tails, explicit prefix warming, or
  llama-benchy-style context probes.
- Use **134** as the conservative mixed-workload starting point. It guarantees
  rendered tails through 134 tokens and can cover up to 267 with favorable
  alignment, with a modest cost on tiny-tail hits.
- Use **268** when rendered user prompts or tool suffixes commonly exceed the
  134-unit coverage window. It guarantees 268 tokens and can cover up to 535.
- Use **536** when producer tails frequently exceed 268 tokens. It guarantees
  536 tokens and can cover up to 1,071, but may replay as many as 1,607 shared
  tokens.

Measure rendered tokenized tail lengths rather than visible text length. The
larger unit does not create more checkpoints or solve arbitrary forks; it moves
the same retained checkpoint farther back and makes the heuristic cover a
broader suffix window.

## Workloads that benefit

The mod helps whenever the earlier producer
request's rendered unique tail fits the configured H-dependent coverage
window:

```text
request A: [long shared prefix][qualifying unique tail A]
request B: [long shared prefix][different tail B]
```

Only producer A's unique tail determines whether its predecessor checkpoint
lands before the fork. B's tail can be longer than the coverage window because
B consumes the checkpoint rather than creating it for this hit. Request order
therefore matters: a qualifying A can accelerate B, while a long A that falls
outside the window cannot seed the same checkpoint for B.

### Large system or developer prompts

Different user requests can reuse a large policy, tool schema, catalog, or
other system context when A's rendered user turn and generation suffix fit the
selected window:

```text
request A: [large shared system context][qualifying user prompt A]
request B: [large shared system context][different user prompt B]
```

Unit 134 covers many concise user turns; 268 and 536 progressively include
longer prompts. Tokenized chat-template markers after the shared system context
are part of A's unique tail.

### Conversation forks and edited turns

A common history can seed a different continuation when the first branch fits
the configured window:

```text
branch A: [common conversation][qualifying continuation A]
branch B: [common conversation][different continuation B]
```

This includes editing the latest user turn, best-of-N orchestration, retrying
with a different final instruction, or evaluating alternative branches. A
larger match unit allows a correspondingly longer first branch without an
explicit checkpoint.

### Agentic and tool workflows

An agent path can seed another path when its post-fork tool result, status, or
next-action suffix fits:

```text
path A: [agent policy][shared working state][qualifying tool result A]
path B: [agent policy][shared working state][different tool result B]
```

Unit 268 or 536 can cover moderate tool outputs and planning suffixes that unit
16 cannot. Longer traces still need an explicit boundary.

### RAG, few-shot, and batch evaluation

A fixed corpus or example set can be reused across questions when the producer
question and its rendered suffix fit:

```text
query A: [instructions][shared documents or examples][qualifying question A]
query B: [instructions][shared documents or examples][different question B]
```

### Context probes and prefix warming

Tiny probes remain the strongest case and work with the smallest unit:

```text
context load: [long shared context]["."]
follow-up:    [long shared context][benchmark prompt]
```

The `"."` producer tail is short enough for every tested unit. Explicit prefix
warming remains useful when the desired fork is known but the first real
producer tail would exceed the selected coverage window.

## Workloads that need an explicit fork checkpoint

An explicit or reactively discovered checkpoint is still needed when the
earlier producer's unique tail exceeds the configured coverage window:

```text
request A: [large shared system context][oversized user prompt A]
request B: [large shared system context][arbitrary user prompt B]

branch A: [long common conversation][oversized continuation A]
branch B: [long common conversation][arbitrary continuation B]

path A: [agent policy][shared working state][oversized tool result A]
path B: [agent policy][shared working state][long tool result B]
```

In that case both retained partial hashes include A-specific tokens and cannot
match B at the actual fork. Increasing `prefix_match_unit` may bring the
workload inside the window, at the cost of replaying more shared tokens. If the
tail remains outside the largest practical unit, use one of the following:

- an explicit frontend-provided checkpoint at the end of the stable shared
  input, using the checkpoint-boundary mechanism explored by
  [#49574](https://github.com/vllm-project/vllm/pull/49574);
- a deliberate prefix-warming request ending at the known fork point;
- reactive shared-prefix detection, which can materialize the branch after it
  has been observed but cannot avoid the first branch miss;
- denser or periodic Mamba-state retention, with a corresponding memory cost.

A prefix-warming request is a useful application-level optimization when the
shared boundary is known in advance, and a valid way to isolate cache behavior
in a benchmark. It is not a general fix for discovering arbitrary future
conversation forks.

## When it will not help

- The earlier producer's unique rendered tail fails
  `U < 2H - (S mod H)`, and no checkpoint was created at the desired fork.
- A long-tail request arrives first and therefore cannot seed the fork for the
  next request. A later qualifying request may still seed it for future work.
- The requests do not share an exact token prefix, or the shared cache entry
  has already been evicted.
- Requests are exact repeats whose final prompt-tail hash already matches; the
  merged partial-prefix implementation can handle those without this fix.
- Prefix caching, MTP/EAGLE, or the affected full-attention + Mamba hybrid cache
  layout is not enabled.

## Related upstream work

The following status is a point-in-time snapshot from the last-updated
timestamp above. No searched upstream PR currently implements this mod's exact
two-part mechanism: retaining the predecessor `prefix_match_unit`
FullAttention hash and materializing the corresponding Mamba replay snapshot
after the EAGLE rewind.

- [#49574](https://github.com/vllm-project/vllm/pull/49574), an open
  experimental draft, is the closest conceptual match. It evaluates explicit
  input-end and decode-end recurrent checkpoints to avoid first-continuation
  misses when generation-only prompt suffixes make the literal prompt-end
  checkpoint unreachable. It requires structured frontend checkpoint metadata
  rather than automatically retaining the predecessor prompt hash.
- [#48815](https://github.com/vllm-project/vllm/pull/48815) is the closest
  compact performance fix. It conditionally avoids EAGLE's full physical-block
  backoff for an MTP prompt with an uncached tail and reports a substantial
  hot-cache TTFT improvement. It does not add the predecessor fine-grained
  FullAttention hash or partial Mamba replay snapshot. The PR had merge
  conflicts at the snapshot timestamp.
- Merged [#46384](https://github.com/vllm-project/vllm/pull/46384) provides the
  partial-prefix infrastructure on which this mod builds, but only registers
  the final prompt-tail hash.
- Open [#45614](https://github.com/vllm-project/vllm/pull/45614) and
  [#46281](https://github.com/vllm-project/vllm/pull/46281) address unsafe or
  mismatched EAGLE/Mamba cache hits. They are correctness fixes and do not
  retain the extra fine-grained changed-suffix checkpoint.
- [#47861](https://github.com/vllm-project/vllm/pull/47861) was a broader
  hybrid MTP prefix-cache correctness attempt and closed without merging.

## Usage

Pass the match unit after the recipe runner's `--` separator:

```bash
./run-recipe.sh \
  --apply-mod mods/fix-eagle-fine-prefix \
  --solo recipes/qwen3.6-35b-a3b-nvfp4.yaml \
  -- --prefix-match-unit 134
```

Do not combine this with `mods/pr-46251-hybrid-large-blocks`. The older mod
changes physical allocation geometry; this one keeps current vLLM's physical
blocks and fixes the missing fine-grained Mamba snapshot.

At startup, verify the `cache_config_info` metric reports the configured
`prefix_match_unit`. For performance tests, compare the delta in
`vllm:prefix_cache_hits_total` as well as TTFT. A successful warm request should
gain hits near the shared-prefix length instead of a multiple of 2,144.

## Validation

The change was tested against current vLLM's partial-prefix-cache suite:

```text
14 passed
```

Two added regression cases fail on unpatched vLLM and pass with the mod:

- the scheduler materializes the Mamba state at the replay point behind the
  predecessor prompt hash;
- a follow-up whose short user suffix differs from the context-load request
  retains a fine-grained hybrid hit instead of falling back 2,144 tokens.
