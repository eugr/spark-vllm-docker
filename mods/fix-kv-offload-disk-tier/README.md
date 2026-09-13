# Disk-Backed KV Offload Tier Fix

**Last updated:** `2026-09-02`

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
