# Disk-Backed KV Offload Tier Fix

**Last updated:** `2026-09-28`

Makes vLLM's `OffloadingConnector` + `TieringOffloadingSpec` with a filesystem
secondary tier actually usable. Two bugs, shipped as one mod because **neither
fix is useful without the other**:

1. **`01-eagle-store-filter`** — with an EAGLE/MTP draft group, the tier returns
   **zero hits, ever**. Nothing is ever promoted, so nothing else matters.
2. **`02-multinode-promoted-row-resync`** — once hits do happen, **tensor
   parallelism across more than one node silently returns wrong KV**.

Apply only #1 and a two-node cluster gets a working cache that corrupts. Apply
only #2 and there is nothing to correct, because the cache never hits.

Verified against vLLM `e2666d9a65f41fc376607531453cbd57c4c71016` on
DeepSeek-V4-Flash-0731 across two DGX Sparks (TP=2, one GPU per node) with an
`fs` secondary tier.

---

## 1. EAGLE/MTP groups can never certify a hit

The store path drops sliding-window chunks that no lookup could reach, keeping
only the trailing `tail` chunks of each full-attention alignment segment:

```python
if pos_in_segment < alignment_chunk_count - tail:
    continue
```

But `_sliding_window_lookup` finds `tail` chunks and an **unverified EAGLE group
then pops one** (`num_hit_chunks -= 1`). It therefore needs `tail + 1`
*consecutive* chunks to certify anything, and the store filter has already
guaranteed it can never see them.

Because `_lookup()` **ANDs the per-group results**, the eagle group's permanent
zero vetoes every other group as well. The whole tier reports a 0% hit rate
while happily writing hundreds of GB. Observed here as 502 GB stored and
0 bytes ever read back.

The fix keeps the segment-head chunk for eagle groups, so the retained set is
`{0} ∪ {acc-tail .. acc-1}` — a run of exactly `tail + 1` consecutive chunks
across the segment boundary, which is what the reader demands. The pop lands on
a chunk at `0 (mod acc)`, so the resulting length stays a multiple of the
full-attention chunk size, i.e. inside the admissible set.

Cost depends on `blocks_per_chunk`: at 8 it is a swap (2-in-4 kept either way);
at 1 it is a ~50% increase in that group's stores. Correctness is
`blocks_per_chunk`-independent.

## 2. Multi-node TP silently serves wrong KV

`SharedOffloadRegion` is a **per-node `/dev/shm` mmap**. On one node the
scheduler-side region (`rank=None`) and every worker region (`rank=r`) are the
*same file* — `rank` only selects a slot *within* a row — so a promotion the
tier manager performs is visible to every worker for free. That assumption is
undocumented, and false as soon as TP spans nodes: each node has its own mmap,
and only the rank co-located with the manager runs a secondary tier at all.

- **GPU→CPU stores stay symmetric** — every rank writes its own region.
- **disk→CPU promotions land only on the manager's rank.**
- the following **CPU→GPU load reads each rank's *own* region**.

Every other rank feeds its GPU whatever stale bytes occupied that row. Nothing
raises, nothing warns, no checksum fails.

Hashing the same row index in both nodes' mmaps, before the fix:

| rows | identical across ranks |
|---|---|
| written by GPU→CPU stores | **112 / 112** |
| written by disk→CPU promotion | **0 / 112** |

Over a longer run the correspondence was exact: every divergent row was a
promoted row, and every promoted row was divergent. After the fix, **112 / 112**.

The symptom depends only on what was in the stale row, which is why it presents
as several unrelated bugs:

| the other rank's row held | output |
|---|---|
| zeros (fresh mmap after restart) | coherent, first checkpoint correct, later ones confabulated |
| another request's KV (recycled row) | multilingual token soup, immediate EOS, **other sessions' content bleeding into the response** |

Hits served entirely from the CPU tier are always correct, because no promotion
is involved and the regions already agree — which is what makes this so hard to
catch. It only misbehaves once a prefix has been evicted to disk and comes back.

### The fix

The tier manager records which rows a **completed** promotion filled;
`build_connector_meta()` drains them into
`OffloadingConnectorMetadata.promoted_rows`; every rank re-syncs those rows from
the manager's rank over the existing TP group **before any load is submitted**.

`build_connector_meta()` runs `on_schedule_end()` → completed-job processing
first, so a promotion that lands in a step ships its row ids in that same step's
metadata. Every rank receives an identical list, so all ranks issue the same
collectives in the same order — which is what keeps the broadcast from
deadlocking. Rows are sorted and coalesced into contiguous runs under a 64 MiB
cap, so a large promotion costs a handful of collectives rather than hundreds.

**Read once, transfer over the link.** Letting every rank read the tier itself
would be strictly worse: the tier would have to be shared, so a second reader
pulls the same bytes over the same link *anyway* and hits the backing disk
twice. On a Spark pair the disk is the slow part (0.65–1.4 GB/s, worse under
concurrency) and the ConnectX link is not (3.45 GB/s single-stream).

**Consequence worth having: the secondary tier no longer needs to be shared.**
Only the manager's rank reads it, so it can be node-local — no NFS, no shared
filesystem, no mount guard on the worker node.

### Safety gate

Re-syncing one rank's rows onto another is only correct if the KV cache is
**replicated** across TP ranks. MLA stores a single compressed latent per token
and is replicated by construction. A head-sharded cache (GQA/MHA) or per-rank
recurrent state genuinely differs per rank, and copying over it would corrupt it
exactly as thoroughly as the bug being fixed.

So the mod does nothing unless every KV group is known-replicated, and it says
which way it decided:

```
KV offload: kv_replicated_across_tp=True (all KV groups are MLA:
  MLAAttentionSpec, SlidingWindowMLASpec); promoted rows will be re-synced
KV offload: re-synced 112 promoted rows in 17 collectives (965214208 bytes)
KV offload: promoted-row re-sync verified on 8 rows across 2 ranks
```

The last line is a one-shot check run immediately after a broadcast, on the rows
just sent, where equality holds by construction — it catches a wrong row stride
or a rank writing into the wrong region, neither of which has any symptom other
than bad output.

**On single-node deployments patch #2 is a no-op**, so it is safe to leave
applied.

---

## Usage

```bash
./launch-cluster.sh --apply-mod mods/fix-kv-offload-disk-tier exec vllm serve <model> \
  --tensor-parallel-size 2 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{
      "spec_name":"TieringOffloadingSpec",
      "cpu_bytes_to_use":4294967296,
      "blocks_per_chunk":8,
      "eviction_policy":"lru",
      "secondary_tiers":[{"type":"fs","root_dir":"/root/.cache/vllm-kv-offload",
                          "n_read_threads":16,"n_write_threads":4}]}}'
```

Mount the tier following the repo's usual cache convention, on the **head node
only**:

```
-v $HOME/.cache/vllm-kv-offload:/root/.cache/vllm-kv-offload
```

**Set `PYTHONHASHSEED` to the same fixed value everywhere.** Without it
`NONE_HASH` is seeded from `os.urandom(32)` per process, so identical tokens
hash differently after every restart and nothing on disk is ever found again.
vLLM already warns about this; the tier makes it expensive. Verify it actually
reaches the engine process rather than just the container — and note that
`/proc/<pid>/environ` is unreliable for `VLLM::EngineCore`, which calls
`setproctitle` and clobbers that region.

`blocks_per_chunk` is the disk-size lever: ~3.6 GB per 131k-token prompt at 8,
~13.3 GB at 1. Both ranks must carry the same value.

### Environment variables

| var | default | meaning |
|---|---|---|
| `VLLM_OFFLOAD_KV_REPLICATED` | unset | force the replication gate `0`/`1`, for cache types the check does not recognise |
| `VLLM_OFFLOAD_MIRROR_STRICT` | `1` | `0` downgrades the post-broadcast check from raise to warn |
| `VLLM_OFFLOAD_MIRROR_MAX_BYTES` | `67108864` | bytes per collective |
| `VLLM_OFFLOAD_MIRROR_LOG_EVERY` | `100` | log the first re-sync then every Nth; `0` disables the periodic line |

## Verifying on your own cluster

Output alone is a weak signal — a corrupt load can still produce fluent text.
Check the mechanism: hash the same row index in both nodes' mmaps after a load
served from disk.

```bash
docker exec <container> python3 -c "
import hashlib, os
p = [f for f in os.listdir('/dev/shm') if f.startswith('vllm_offload_')][0]
p = '/dev/shm/' + p
n = 498                                   # your num_blocks
stride = os.path.getsize(p) // n
f = open(p, 'rb')
for r in (0, 1, 2, 50, 100):
    f.seek(r * stride)
    print(r, hashlib.sha256(f.read(stride)).hexdigest()[:16])
"
```

Digests must match across nodes for any row a promotion filled. Before patch #2
they never do; after it, they always do.

Make sure the load really came from disk — a CPU-tier hit proves nothing, and
`reset_prefix_cache` does **not** drain the CPU tier at `blocks_per_chunk > 1`.
Force eviction with unrelated filler prompts first.

## Upstream

Bug #2 is a concrete cause for the failure class in vLLM RFC #54363 — "content
that is the right length and the wrong bytes … consumed as attention KV,
producing wrong logits with no error signal anywhere". If re-syncing is
considered out of scope upstream, the minimal alternative is for
`TieringOffloadingSpec` to **refuse to start** when the tier manager's region is
not shared by every rank, rather than silently serving wrong KV.


---

## Patch 03 — matching decoupled from staging (2026-09-08)

**Patches 01 and 02 make the disk tier *correct*. They do not make it *useful*
on a prefix larger than your primary tier. This one does.**

If you applied this mod before 03 existed and saw the tier do nothing — no
error, no warning, just persistent zero hits — this is why.

### The bug

`TieringOffloadingManager.lookup` ends:

```python
return LookupResult.MISS if not promoted else LookupResult.RETRY
```

The matching walk promoted **one primary-tier row per queried key**, purely to
confirm the key was there. Once the tier filled, every subsequent key — *including
keys sitting on disk* — returned `MISS`, indistinguishable from "never stored".
The cross-group AND (`if num_hit_chunks == 0: return 0`) then discarded the whole
external hit.

Self-reinforcing: the walk consumed the rows `prepare_store` needed, so the tier
stopped being **written** too, and never recovered on its own.

**This will hit most users of this image.** The tier is sized from host RAM and
these are 128 GB boxes; with a large model resident, `cpu_bytes_to_use` of a few
GiB buys only a few hundred rows. A long agent conversation queries far more keys
than that in a single match. On our 4 GiB / 498-row tier, a 242k-token prefix
queried **12,434** keys.

### The 30-second diagnosis

Sum the per-group RETRY counts on a vetoed lookup. **If the sum pins at exactly
your tier's row count, you are hitting this and your data is on disk.** Confirm
the row count independently from metric granularity: `kv_offload_cpu_cache_usage_perc`
only ever takes values `k/rows` (ours reported `0.08032128514056225` = `40/498`).

Do not read a high miss count as disk absence without checking this first. That
misreading cost us two weeks.

### The fix

Match with `promote=False`, then stage the confirmed hit in **waves** so a hit
larger than the tier can still be served. Three new stdlib-only modules
(`wave_slicer.py`, `parking_gate.py`, `parking_sm.py`), each with host-runnable
tests, plus the driver in `scheduler.py` and the `promote` flag in `manager.py`.

### Measured on a live 2-node TP=2 DeepSeek-V4-Flash-0731 cluster

| | before | after |
|---|---|---|
| summed RETRY per probe | 498 | **0** |
| `exit=ZERO` | 14 | **0** |
| `cannot store chunks` | 2 | **0** |
| cold 294,186-token prompt | `ext=0` | **`ext=290,816`** (98.9%) |

Also verified since:

- **Multi-wave really runs.** 23 loads at `num_waves=2`/`3`. Slicing exact:
  `ext=305152 wave_sizes=[64, 64, 26]` is 64+64+21 = 149 g0 chunks =
  305152/2048, sliding-window groups riding the last wave. Blocks close end to
  end: 512+512+190 = 1214 = 149x8 + 22.
- **Both ranks.** Same job ids and `src_blocks` on each; rank 1 emits the
  matching `cpu_to_gpu` transfers. The `src_offset`/`dst_offset` asserts survive
  wave boundaries.
- **Streaming beats tier size.** A **1 GiB (124-row)** tier served a cold
  **348,000-token** prompt at `ext=346112` — **99.5%**. That prefix needs ~170
  rows to stage, so it is monolithically impossible; only waves can do it.
- **Output is not degraded.** Next-token distribution from tier-served KV is
  *within the engine's own noise floor*, and equal to vLLM's own GPU prefix
  cache (2.95 vs 2.93, floor 3.20). Tooling: `tools/kv-quality-ab.py` in our
  repo.
- **Parking works** (16 slot events, all released, no hang) but is **OFF by
  default** — `VLLM_OFFLOAD_PARK=1` to enable. Treat it as the least-exercised
  part of this patch.

### Tuning

| var | default | meaning |
|---|---|---|
| `VLLM_OFFLOAD_STREAM_WAVE_CHUNKS` | 64 | chunks per wave. **0 makes this patch fully inert** — the fastest revert, verified by a dark baseline. |
| `VLLM_OFFLOAD_PARK` | 0 | admission gate. Only reachable when a request's demand exceeds half the tier. |

A wave holds `WAVE_CHUNKS x tokens_per_chunk` tokens. At 256 and a 2048-token
chunk that is 524,288 — larger than most workloads, so it never splits. If you
want waves to actually engage, size it against your prefixes.

### Honest scope

- Ships ~70 lines of **env-gated diagnostics** (`KVPROBE`/`KVCOV`/`KVPROV` and a
  shadow load-solver), all inert unless their env var is set. They are the
  instruments that found this; removing them by hand would ship code we have not
  run.
- **`cannot store chunks` is a separate, pre-existing failure, and you should
  expect to meet it.** `prepare_store` cannot allocate rows, and that path
  neither advances the cursor nor backs off nor throttles its log — so the
  request retries the identical oversized batch on *every* scheduler step.
  Measured on our workload (~250-350k-token prompts):

  | `cpu_bytes_to_use` | rows | `cannot store chunks` |
  |---|---|---|
  | 1 GiB | 124 | **8,870** lines from 40 requests, worst one 1,847x, ~218/min |
  | 2 GiB | 249 | **0** |

  So: **raise `cpu_bytes_to_use` until it stops.** The right value scales with
  your prompt length, not with this table — a store batch is
  `num_offloadable_tokens / tokens_per_chunk` summed over groups, and the
  sliding-window groups dominate it (a 348k prompt wants ~17,800 chunk-rows in
  total, of which 91% are g3/g4 at 32- and 64-token chunks).

  Note this is **not** what `VLLM_OFFLOAD_STREAM_WAVE_CHUNKS` controls — wave
  size governs the *load* path and is not referenced in the store path at all.
  Halving it frees ~32 rows against a deficit in the hundreds.

  Not introduced by 03 — but 03 makes small tiers useful enough that you may
  now run one small enough to hit this.
- Written with AI assistance and verified on the hardware above: a collaboration
  between Claude Opus 5 and DeepSeek-V4-Flash reviewing each other's work. Every
  number here is measured, not asserted by a model.

---

## Patch 04 — wave readiness evaluated at load (2026-09-21)

**Fixes an engine crash in 03's wave driver that we hit after 20.6 h under
concurrent load.**

### The bug

```
File ".../offloading/scheduler.py", in build_connector_meta
    src_spec = self.manager.prepare_load(w.keys, req_status.req_context)
File ".../kv_offload/tiering/manager.py", in prepare_load
    return self.primary_tier.prepare_load(keys, req_context)
File ".../kv_offload/cpu/manager.py", in prepare_load
    assert block is not None, f"Block {key!r} not found in cache"
AssertionError: Block b'...' not found in cache
```

`EngineCore` dies and every in-flight request gets `EngineDeadError`. Seen once,
after 20.6 h of production traffic on our own two-DGX-Spark cluster
(DeepSeek-V4-Flash-0731, TP=2, one GPU per node, `fs` tier), running 01–03 with
`VLLM_OFFLOAD_STREAM_WAVE_CHUNKS=64`. Our image also carries unrelated local
patches outside the offload path.

03's driver *remembers* readiness. It records a wave's keys in `wave_ready_keys`
as they become resident in the primary tier (their promotion completed, they
were already resident, or another request promoted them) and ships the wave
once every key has been seen. But a resident row is only pinned by
`prepare_load()` (ref_cnt 0 → 1). Until then it is evictable, so a key that
became ready while the rest of its wave was still promoting can be evicted by
any other request's `prepare_store`/`prepare_write` before the wave's last key
lands. The driver then loads a key that is gone. As far as we can see, this is
the only way the wave path can reach that assert; the eviction of the failing key
itself was not logged.

### The fix

Evaluate readiness, never remember it. `TieringOffloadingManager.wave_lookup()`
asks the primary tier about the whole wave in the same pass as the
`prepare_load()` that pins it:

| `wave_lookup` | driver |
|---|---|
| `HIT` (every key resident and readable) | `prepare_load`, ship the wave |
| `HIT_PENDING` (a write is still in flight: a promotion, or another request's store) | wait |
| `MISS` (evicted, failed promotion, `reset_cache` wipe) | re-stage the wave; `promote_for_staging` re-reads only the absent keys |

`wave_ready_keys`, `pop_ready_keys` and `WaveSpec.promoted` are removed.

Stuck waves are now visible. Every step a wave makes no progress, staging
refused or re-staged after a `MISS`, is counted in
`vllm:kv_offload_wave_retry_total{reason="promote_refused"|"miss"}`. The log warns
at `VLLM_OFFLOAD_STREAM_WAVE_MAX_RETRIES` (default 50) and at each doubling.
Before, only refusals incremented the retry count, and it warned once, at the
ceiling.

### Verified

- `test-wave-lookup.py` replays the eviction on vLLM's real
  `CPUOffloadingManager`: `HIT_PENDING` while the wave promotes, `MISS` after
  another request's store evicts a ready key (where `prepare_load` raises the
  exact assertion above), `HIT` after re-staging. It also checks that the counter
  is exported. No GPU needed; from `mods/fix-kv-offload-disk-tier`:
  `docker run --rm --entrypoint python3 -v "$PWD/test-wave-lookup.py:/t.py:ro" <image> /t.py`
- 01–04 apply with `git apply` on vLLM `e2666d9a65f41fc376607531453cbd57c4c71016`,
  all touched files compile, and a second `run.sh` skips.
- vLLM's `tests/v1/kv_connector/unit/offloading_connector` and
  `tests/v1/kv_offload`: the same pass/fail set with and without 04 (403
  passed). We ran them in a container without a GPU, so the GPU-dependent tests
  could not run either way.
- Live, on that cluster: after a full restart, a 175,022-token prompt restored from
  disk in 3.2 s through the new path, with no errors. The prefill that first stored
  it took 139.5 s.
  The original crash took 20.6 h to appear, so hours without a recurrence are
  weak evidence; the replay is the proof of mechanism.

### Honest scope

- **Retries are still unbounded.** A key the secondary tier has lost re-reads
  every step while its request holds the GPU blocks allocated up front. The
  abort-and-recompute fallback (03's OPEN A) is still not implemented; 04 makes
  that case visible (`reason="miss"` climbing on one wave), not bounded.
- Unrelated to 04, found while running the tests: with 03 applied, 23 tests in
  vLLM's `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` fail with
  `TypeError: _make_scheduler_with_lookup.<locals>.<lambda>() got an unexpected keyword argument 'promote'`,
  because the test's `lookup` mock does not accept 03's new `promote=` argument.
- vLLM #54914 reports the same assertion on stock vLLM without this mod. 04 fixes
  only the wave path and says nothing about that report's cause.
- Written with AI assistance (Claude) and verified on that cluster.

## Patch 05 — fs tier completes zero-task jobs instead of leaking them (2026-09-25)

**Fixes an idle engine burning ~1.4 cores forever.**

### The bug

`DualQueueThreadPool` reports a job finished only from `task_done()` of its
**last** task. A job with zero tasks has no last task: it is never reported,
never leaves `_inflight_jobs` (so `wait_idle()` would never return either), and
`TieringOffloadingManager` keeps it in `_transfer_jobs` forever.
`has_pending_work()` then stays `True`, and the engine steps an empty batch every
~1 ms for the rest of its life.

Zero-key jobs are real. `prepare_write()` drops keys that are already resident or
in flight, so a wave whose keys were all staged by another request flushes as an
empty promotion. `FileSystemTierManager` also passed `len(keys)` as the task
count while `zip(keys, block_ids)` stops at the shorter of the two: a count
above the real number of tasks is the same never-finishes leak.

Measured on our cluster: `_transfer_jobs` grew from 4 to 127 over 25 h with no
request in flight, and the idle engine sat at ~140% CPU.

### The fix

`FileSystemTierManager` materialises the task list, enqueues it with its real
length, and completes a job with no tasks itself, on the next
`get_finished_jobs()` poll, without touching the pool. Each one is logged
(the first 20, then every 100th).

### Verified

- Live since 2026-09-25: idle container CPU **3.3%** (was ~140%); in ~17 h, 38
  empty jobs were completed this way, all of them loads, and `_transfer_jobs`
  drained.

### Honest scope

- The alternative, making `DualQueueThreadPool.enqueue_*` finish a zero-task job
  immediately, is arguably cleaner. We shipped the manager-side fix because it is
  the one that ran in production.

## Patch 06 — a stuck wave kills the engine after 120 s (2026-09-26)

**Bounds 03's OPEN A (see 04's "Honest scope": retries are unbounded).**

### The bug

If a wave's keys can never be loaded, because their files are missing from the
secondary tier, the wave retries every scheduler step forever. There is no
abort-and-recompute fallback. We hit it on 2026-09-26: for 95 minutes the engine
retried one wave ~1.6M times at ~170% CPU, held 97% of GPU KV, served nothing,
and logged ~900k `block I/O failed ... ENOENT` lines, while the client's
15-minute resends walked into the same hole six times.

In our case the missing files were caused by a bug in an unrelated local vision
patch, which lowered the load boundary below the window the lookup had
certified. The failure mode is generic, though: anything that removes tier files
under a running engine ends up here.

### The fix

A wave that has made no progress for `VLLM_OFFLOAD_STREAM_WAVE_STUCK_S` seconds
(default **120**) raises `RuntimeError("KVWAVE STUCK ...")`, which kills
`EngineCore`. `0` restores retry-forever. The stuck-wave warning now says how
long is left.

Why a crash and not recompute: falling back means reporting the request's
still-unloaded GPU blocks as failed loads so vLLM recomputes them. That runs
through the worker-side load-error path on every rank, and whether vLLM
re-enters cleanly on already-allocated blocks has not been verified.
Unverified KV reuse produces garbage output; a crash cannot. A dead engine is
visible and restartable; a spinning one looks alive.

### Verified

- Over 16 days of our production journal, no wave outside that incident ever
  reached even `VLLM_OFFLOAD_STREAM_WAVE_MAX_RETRIES` (50 steps), so 120 s is far
  from normal contention.
- Deployed since 2026-09-26 with no false trigger.

### Honest scope

- **The raise itself has never fired in production.** Nothing has gone stuck
  since it was deployed, so the crash path is verified only by reading it.
- It is a bound, not a fix: the real fix is still the recompute fallback.

## Patch 07 — optional SWA store stride (2026-09-28)

**Cuts the disk tier to ~¼ on DeepSeek V4, with no change to resume latency.
Off by default.**

### The problem

01's alignment filter keeps each sliding-window group's tail (and the eagle
group's segment head) at **every** full-attention boundary, so any of them can be
a load point. On DeepSeek V4 with `blocks_per_chunk=8` that is, per 2048 tokens,
7 chunk files of 8.6 MB: g0 ×1, g1 ×1, g2 ×2, g3 ×1, g4 ×2. The four SWA groups
are 6/7 of the tier. Under agent traffic (250–350k-token prompts) our tier grew
~50 GB/h and filled a 2 TB disk in a day. Almost none of the interior boundaries
are ever load points.

### The fix

`VLLM_OFFLOAD_SWA_STORE_STRIDE=S` keeps the SWA tails/heads only at boundaries
that are multiples of `S × alignment` tokens, **plus the last two boundaries of
every request's prompt**. A multi-turn client's next request resumes exactly
there. Only prompt tokens are offloaded by default (`offload_prompt_only`), so
that end is known when the chunk is stored.

The load path is unchanged: `_sliding_window_lookup` scans backward for the last
stored window, every SWA group lands on the same boundary, and the eagle pair is
kept at exactly those boundaries. The admissible sets coincide, so the lookup
converges in one pass (unlike the 2026-08-30 ratchet, which came from
**disjoint** sets). Files already stored with stride 1 stay valid.

### Measured (S=8)

- **Disk:** after 73 min of live traffic the SWA groups had written 2,746 files
  against g0's 3,195. At stride 1 that would have been ~19,000. Per 2048 tokens:
  ~1.8 files instead of 7.
- **Resume latency, live:** 7 of 19 tier hits in that window resumed at the end
  of the previous prompt (off the 16k grid), re-prefilling 1.5–7k tokens,
  including the turn's new content.
- **Resume latency, simulated** on the real group geometry (3,000 agent turns,
  20% diverging inside the previous prompt's last 1.5k tokens): re-prefill median
  **1,533** / max 2,557 tokens, against stride 1's 1,536 / 2,559. **0**
  uncertified loads.
- 0 errors live.

### Do not drop the prompt-end boundaries

The first version kept only the stride boundaries. That made every resumed turn
snap down to a 16k boundary: up to 14k tokens (~9 s) of extra prefill **per
turn**, in both the CPU and the disk tier, because the filter runs before either.
Averaged over all requests it looked cheap; per turn it was not.

### Honest scope

- Measured on DeepSeek V4 only. The arithmetic is generic for any
  full-attention + SWA model, but the ~¼ figure depends on the group mix.
- A resume that diverges in the **middle** of an earlier prompt (e.g. a context
  fold rewriting old history) snaps down to the stride grid: on average
  `(S−1) × 1024` extra tokens.
- 07 reduces growth but does not bound it; 08 does.

## Patch 08 — free-space LRU eviction for the fs tier (2026-09-28)

**Gives the fs tier a bound. Off by default.**

### The problem

`FileSystemTierManager` has no capacity limit, no eviction and no quota: the
tier grows until the filesystem is full. A full disk turned out to be harmless to
the engine. We measured 2 h at 0 bytes free, with 17,425 failed stores (ENOSPC),
0 load errors, and serving continued at a ~63% external hit rate. But the tier
stops caching, and it starves everything else on that disk. The only bound was a
manual wipe, which throws away the tier exactly when it is most valuable: right
after a restart, when the GPU cache is empty.

### The fix

`VLLM_OFFLOAD_FS_MIN_FREE_GB=N` (0 / unset = off): below N GB free on the tier's
filesystem, delete the **least recently used** chunk files until
`VLLM_OFFLOAD_FS_TARGET_FREE_GB` (default 1.5 × N) is free. A free-space floor
rather than a tier size, because the disk is usually shared.

- **Recency lives in memory, never in file metadata.** A hot conversation is
  served from the GPU/CPU tiers and never reads its files, so atime would call
  it cold, and writing atime on every use would add SSD wear. The scheduler
  already calls `touch()` on every lookup; the fs tier now implements it, and
  stores and loads count as uses too. At startup the list is seeded from the
  files' atime (read-only `stat`), so a tier kept across a restart is ordered
  from the first step. ~150 B of memory per file.
- **Safety.** A file deleted between a lookup HIT and its load would leave a
  wave with nothing to load (see 06). A key is therefore never evicted while the
  async lookup cache holds it (some live request looked it up), while a load job
  reads it, or within `VLLM_OFFLOAD_FS_EVICT_MIN_AGE_S` (600) of its last use.
  Once chosen, it answers MISS to lookups until the unlink is done.
- Victims are picked on the scheduler thread; a background thread does the
  unlinks and a `statvfs` every 30 s. Files already queued count as freed, so a
  stale `statvfs` cannot overshoot the target.
- Tier directories of another model or layout are never touched, but they are
  named in a startup warning.
- Log lines: `KVFS EVICT enabled/start/done`, `KVFS SEED`.

### Verified

- Unit test against a temp directory and a simulated disk: deletes strictly
  oldest-used first, keeps looked-up / loading / recently used keys, stops within
  one file of the target, never writes file metadata.
- Live since 2026-09-28 17:38 (MIN 200 / TARGET 300 GB): the seed picked up the
  kept tier (7,790 files) at startup. Growth with 07 was ~4 GB/h, so it did not
  trigger on its own for 36 h. A forced round (a `fallocate` ballast dropped free
  space to 195 GB): **2,560 files / 20.5 GB deleted in 92 s** while serving,
  oldest-used first (the oldest victim was last used before the restart, as
  seeded), 0 protected keys hit, 0 `block I/O failed`, 0 stuck waves. The round
  ended within one step of removing the ballast.

### Honest scope

- Eviction advances once per scheduler step, so an **idle** engine does not
  evict. That is fine for the tier's own growth, which only happens while it
  serves, but a different process filling the disk is not answered until the
  next request arrives.
- The forced round did not end by itself: we removed the ballast after 92 s,
  which put free space back above the target, and the round stopped on the next
  step. A round that deletes all the way down to the target on its own has not
  run live yet. It is the same loop, only longer.
