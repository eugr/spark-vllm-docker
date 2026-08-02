# spark — Roadmap

**Status:** Draft · **Original design:** 2026-06-14

> Implemented phases (see `docs/SPARK.md`):
> - Phase 0 — `--node` solo placement (remote solo over SSH)
> - Phase 1 — `fleet.yaml` + `apply`/`down`/`restart` + `state.json`

---

## R2 — Memory Budgeting ("use the space properly")

**One GPU per node**, ~**120 GB** unified memory (DGX Spark / GB10).
Models are small relative to memory: a 35B **nvfp4** model ≈ ~20 GB weights.
Two models can co-tenant a single node.

### Primary lever: absolute KV cache size in GB

vLLM's `--kv-cache-memory-bytes` sizes the KV cache exactly, overriding
`--gpu-memory-utilization`. Absolute KV + fixed weights = known footprint.

| Config field | vLLM flag | Notes |
|---|---|---|
| `kv_cache_gb: 5` | `--kv-cache-memory-bytes 5368709120` | `gb × 1024³` |
| `kv_dtype: fp8` | `--kv-cache-dtype fp8` | |
| `mtp: true` | speculative/MTP config | model-dependent |

Per-node validation on `apply` (prevents co-tenant OOM):

```
node footprint = Σ over deployments ( weights + activations + kv_cache )
require: node footprint ≤ capacity - reserved
```

### Memory model & profiling

Per-instance GPU memory:

```
total = weights + activations(peak) + non_torch + KV cache
        └ file read  ┘  └ measured ─────┘   └ you set ┘
```

- **weights** — read from HF cache (`Σ *.safetensors` blobs).
- **activations** — measured via `spark profile` (boot once, scrape vLLM logs).
- **non_torch** — CUDA context + NCCL, ~0.5–2 GB.
- **KV** — `kv_cache_gb`, user-set.

`spark profile <recipe>` boots the model once, scrapes the memory breakdown,
and caches `weights_gb / activation_peak_gb / non_torch_gb` in state.

### KV geometry

```
KV bytes = 2 × layers × kv_heads × head_dim × dtype_bytes × tokens
```

`spark scan --tokens 1M` already computes this from `config.json`.
Inverts to: *"pick `kv_cache_gb` from a concurrency × context target."*

> Caveat: **hybrid models** (linear + full attention layers) have constant
> state in linear layers — the formula is an **upper bound**.

### Schema additions (v2, backward compatible)

```yaml
kv_cache_gb: 20     # → --kv-cache-memory-bytes
kv_dtype: fp8       # → --kv-cache-dtype
mtp: false          # → speculative config
mem_gb: 40          # OPTIONAL hard ceiling (gpu-mem-utilization-gb)
```

---

## R4 — Dual-rail RoCE NCCL

`autodiscover.sh` currently hardcodes four RoCE ports for one mesh layout.
Change: detect present ConnectX-7 netdevs/HCAs, set `IB_IF` / `NCCL_IB_HCA`
to all detected rails.

- Verify both rails carry traffic: `NCCL_DEBUG=INFO` topology dump + 2-node
  all-reduce bandwidth test. Assert ~2× single-rail BW.
- Keep `NCCL_IB_MERGE_NICS=0` (separate rails for aggregate BW).

---

## Router (LiteLLM)

- Runs as a container on the master, fronts all instances.
- `/v1/models` → aggregate of deployment `model` fields.
- Routes by `model` field to the instance's `node:port`.
- OpenAI (`/v1/chat/completions`) + Anthropic (`/v1/messages`).
- Config **generated from `state.json`** on every `apply`/`down`.
- **No token validation** initially.

Auth (deferred): Keycloak → JWT → LiteLLM + Postgres.

---

## Additional Commands

```
spark diff   -f fleet.yaml    # dry-run: show plan without executing
spark sync                 # reconcile state.json vs docker ps (drift repair)
spark ls                   # observed deployments + free memory per node
spark up <recipe> ...      # ad-hoc single deployment (sugar over one-entry apply)
```

---

## Open Questions

- **Q1** Single master assumption: `fleet.yaml` + `state.json` live on one
  control host. Confirm always exactly one control host (no HA)?
- **Q3** Port range for auto-allocation (e.g. 8001–8099)?
- **Q4** On master reboot: who re-runs `spark apply` (manual? cron?) since
  there's no daemon? `sync` only repairs state drift.
