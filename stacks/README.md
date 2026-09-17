# Stacks — serving multiple recipes at once

A **stack** runs several vLLM recipes at once, each in its own Docker container
on its own port, sharing the same on-disk model cache. Each recipe picks its
own **placement** independently — distributed across the cluster, pinned alone
to one node (local or remote), or left to the default. The simplest stack
co-locates two models on one machine (e.g. a large chat model on `:8000` and a
small tool/coder model on `:8001`); the same manifest format also spreads one
model across the cluster while pinning another to a specific node.

Stacks are a thin layer over the normal recipe flow: `run-stack.py` calls
`run-recipe.py` once per recipe, so everything recipes already do — mods, model
download, container image selection, command templating — works unchanged.
`run-stack.py` passes each distributed recipe its own `--master-port` so
co-located distributed recipes don't collide. See **Placement** below.

## Quick start

```bash
./run-stack.sh --list                                    # list stacks/*.yaml
./run-stack.sh stacks/example-dual-stack.yaml --dry-run  # show the ordered launch plan
./run-stack.sh stacks/example-dual-stack.yaml --setup    # build/download as needed, then bring up
./run-stack.sh stacks/example-dual-stack.yaml --status   # per-container state + /health
./run-stack.sh stacks/example-dual-stack.yaml --stop     # tear the whole stack down
```

A stack can be named by path, as above, or by bare name — `run-stack.sh` looks
for a literal path first, then `stacks/<name>`, `stacks/<name>.yaml`, and
`stacks/<name>.yml`. So `./run-stack.sh example-dual-stack --status` is
equivalent to the third line above.

## Why order matters (and why it's sequential)

Recipes are launched **in the order listed**, and that order **must be
descending memory usage**. vLLM profiles free memory at startup, so the largest
model must claim its slice of the unified memory pool first — and fail fast if it
won't fit. Each recipe is started only after the previous one's `/health`
endpoint reports ready (`health_timeout` seconds, default 1200). This also
serializes first-time kernel compilation so co-located instances don't race on
the shared compile caches (`~/.cache/vllm`, `flashinfer`, `triton`, `tilelang`).

## Manifest format

```yaml
stack_version: "1"
name: my-stack                 # optional; defaults to the file name
description: ...               # optional
health_timeout: 1200          # optional, seconds per recipe

recipes:                      # required; listed in DESCENDING memory order
  - recipe: <name-or-path>    # required; resolved like run-recipe.py (recipes/<name>.yaml)
    container_name: <name>    # optional; Docker --name, must be unique across the stack
                              #   (defaults to a sanitized recipe name)
    port: <int>               # optional; defaults to the recipe's defaults.port
    placement: all | <IP>     # optional; defaults to 'all' -- see "Placement" below
    gpu_mem: <float>          # optional; fraction override -> run-recipe.py --gpu-mem
    master_port: <int>        # optional; distributed recipes only, see "Placement" below
    mods: [<path>, ...]       # optional; extra --apply-mod paths (repo-relative)
    volumes: [<mapping>, ...] # optional; Docker -v host:container[:options] mounts
    extra_args: [<arg>, ...]  # optional; appended after `--` to run-recipe.py (vLLM args)
```

The **served model name** (`--served-model-name`) stays recipe-owned —
`run-stack.py` doesn't invent it. The **port** defaults to the recipe's
`defaults.port` and can be overridden per entry; `run-stack.py` never picks one
itself. `container_name`s must be unique across the whole stack (so
`--stop`/`--status` and `docker logs <name>` are unambiguous); `port`s must be
unique **per node** (two entries pinned to different nodes may reuse a port),
or the run aborts with a clear error before anything launches.

### Extra mounts (`volumes`)

`volumes` forwards Docker's `-v host:container[:options]` to `run-recipe.py`
per entry — for local weights, datasets, or an output directory the HF cache
mounts don't already cover:

```yaml
  - recipe: qwen3.6-35b-a3b-nvfp4
    volumes:
      - /srv/models:/models:ro
      - datasets/eval:/eval      # repo-relative, resolved to an absolute path
```

A relative host path is resolved against the repo root, like `mods`; `~` is
expanded. A bare token with no `/` (e.g. `hfcache:/root/.cache/huggingface`) is
left alone, since Docker reads that as a **named volume**. The container path
must be absolute, and mappings containing whitespace or shell metacharacters are
refused — `launch-cluster.sh` expands the mount unquoted into the `docker run`
line, so they would word-split or execute rather than mount. All of these are
checked before anything launches.

> **Host paths are resolved on the machine running `run-stack`, but mounted on
> the placement node.** An entry with `placement: all` runs a container on every
> node in the cluster and applies the same `-v` to each, so the host path must
> exist on **all** of them. A `placement: <IP>` entry has the same trap in
> smaller form: a relative or `~` path is expanded here and then shipped to that
> node over ssh. Docker does not fail on a missing host path — it silently
> creates an empty root-owned directory and mounts that. **Prefer absolute host
> paths for any entry that is not local**, and confirm the path exists there.
> Mounts are per entry: they never leak to other recipes in the stack.

### Memory budgeting: fractions vs absolute GiB

`gpu_mem` sets vLLM's `--gpu-memory-utilization` as a **fraction of total**
memory; the fractions across co-located recipes must sum to **< 1** with headroom
for CUDA context and non-torch allocations.

Absolute GiB budgets compose more predictably when stacking. The
[`mods/gpu-mem-util-gb`](../mods/gpu-mem-util-gb) mod patches vLLM to accept
`--gpu-memory-utilization-gb`; add it per entry and pass the flag via
`extra_args` (see the commented block in `example-dual-stack.yaml`).

## Placement

Each recipe entry picks its own `placement`, independent of every other entry
in the stack — a single manifest can distribute one model across the cluster
while pinning another alone to a specific node (see
`example-cluster-stack.yaml`).

- **`placement: all`** (the default when the key is omitted) distributes the
  recipe across every node in `.env`'s `CLUSTER_NODES` (or the `--config` file
  passed to `run-stack.py`). **If `CLUSTER_NODES` has fewer than two nodes
  (unset, empty, or a single node), it runs solo on this host** — the normal
  case on a single Spark, so this is not an error and not a warning. A recipe
  marked `cluster_only:` in that position *is* refused outright, before
  anything launches. For an unflagged recipe genuinely sized to spread across
  a cluster, though, nothing stops the attempt to fit it on one host, and it
  can OOM. The placement summary printed by every sub-command names the target
  of each entry (`-> distributed across N nodes` / `-> this host` / `-> node
  <IP>`); check it, or `--dry-run` first, before trusting that a stack is
  actually running distributed.

- **`placement: <IP>`** pins the recipe alone to that one node — not
  distributed — whether the IP is this machine (`LOCAL_IP` in `.env`/`--config`)
  or a remote node. The IP **must** be either `LOCAL_IP` or one of
  `CLUSTER_NODES`; an unrecognized IP is rejected at load time, before
  anything launches, naming the IP and the nodes it was checked against.

- **Pinning to the invoking host requires `LOCAL_IP` to be set.** Without
  `LOCAL_IP` in `.env`/`--config`, `run-stack.py`, `run-recipe.py`, and
  `launch-cluster.sh` have no way to know that a listed IP refers to this
  machine, and may disagree on whether it's local — treating a pin meant for
  "here" as remote and ssh'ing back into this machine instead of running
  locally.

- **Ports are unique per node, not per stack.** Two entries may reuse the same
  `port` as long as they resolve to different nodes (e.g. one distributed
  across the cluster, one pinned elsewhere) — `run-stack.py` tracks
  `(node, port)` pairs, not just `port`.

- **`master_port` applies only to distributed recipes** (`placement: all`
  resolving to 2+ nodes); a `single`-placement entry (solo or pinned) never
  uses cluster coordination and ignores it. If you don't set it, `run-stack.py`
  auto-assigns one per distributed entry as base + index in manifest order,
  where the base is `.env`'s `MASTER_PORT` if set, else 29501 — so with the
  default base entries get 29501, 29502, …, and co-located distributed
  recipes never collide with each other by default. Override it explicitly
  when the auto-assigned value collides with something else on the network,
  or you need a stable port across manifest edits.

- **`--status` and `--stop` target each entry's own resolved node** — ssh'ing
  to a remote pinned node or to every worker of a distributed entry as needed
  — not a single stack-wide mode. The health gate that watches for a startup
  failure while bringing an entry up also targets that entry's node (its head
  node if distributed, its pinned node if pinned).

- **`--setup` fans out per entry**, and per node for a distributed entry. Each
  recipe with `--setup` re-runs image build and model distribution to every
  node it resolves to, not just once for the stack. This is correct (each
  entry may need different images/mods, and a distributed entry needs the
  image/model on every node it spans) but slow on a cold cluster — expect the
  first `--setup` run to take substantially longer than subsequent ones once
  images and models are cached everywhere.

- **Run `run-stack.sh` on the node you expect distributed entries to head
  from.** A distributed entry's head is not `nodes[0]` — it is whichever
  *listed* node is the invoking machine (`launch-cluster.sh` derives it from
  the local IP). Invoking from a different listed node silently makes *that*
  node the head for that entry, inverting the intended roles; invoking from a
  machine that isn't in the node list at all makes `launch-cluster.sh` refuse
  to launch before anything starts. `--host` (used to poll `/health` for
  non-pinned entries) defaults to `localhost`, which matches the head only
  when you run the stack from it.

- **`gpu_mem` budgeting is per node, not per stack or per cluster.** A node
  that hosts a shard of a distributed recipe *and* a recipe pinned to it must
  fit both: under tensor-parallel sharding every node in a distributed
  recipe's cluster holds a shard of that recipe, so a pinned entry sharing one
  of those nodes needs its `gpu_mem` sized against what the shard already
  uses there, not against the distributed recipe's total footprint across the
  whole cluster. A recipe's *default* `gpu_memory_utilization` is tuned for
  running that model *alone*, so every co-located entry on a node still needs
  an explicit `gpu_mem` (or an absolute-GiB budget via
  `mods/gpu-mem-util-gb`) chosen so the totals on that node leave headroom.
  See `example-cluster-stack.yaml` for a worked example (two `placement: all`
  entries plus one pinned to a node also carrying a distributed shard).

- **A stray top-level `solo_only:` key is ignored, with a warning.** Older
  manifests set a stack-wide `solo_only:` to force every recipe solo; that key
  has been replaced by per-recipe `placement:` and is no longer read for
  anything but the warning — remove it and set `placement:` per entry instead.

- **A recipe's own `cluster_only`/`solo_only` (in the *recipe* YAML, not the
  stack) still applies.** These are separate, per-recipe restrictions
  unrelated to stack placement — some models genuinely cannot run distributed
  (or cannot run solo) regardless of what a stack asks for. A recipe whose
  resolved placement contradicts its own restriction is rejected at load time,
  naming the conflict and how to fix it (use `placement: all` with a
  configured cluster, or pin with `placement: <node-IP>`).

## Limitations when co-locating

- **Total memory** — the sum of all recipes co-located on a given node
  (weights + KV cache + activations) must fit that node, with headroom. This
  is the main constraint on a GB10's 128 GB unified pool.
- **Cluster coordination port** — each distributed recipe binds a master port on
  its head node (`--master-port`, default 29501), used by both the no-Ray
  PyTorch backend and the Ray head. Co-located distributed recipes therefore need
  distinct ports; stacks assigns base + index automatically (base = `.env`'s
  `MASTER_PORT`, else 29501 — so `29501`, `29502`, … by default), and
  `master_port:` overrides an entry if something else owns that range.
- **Shared `/dev/shm` (`--ipc=host`)** — vLLM's engine↔API-server IPC uses shared
  memory; heavy co-location may need `--non-privileged` (per-container
  `--shm-size`), which the underlying recipe/launcher already supports.
- **One GPU** — compute is time-sliced across containers (no MIG on GB10);
  functionally fine, but throughput contends under simultaneous load.
- **`cluster_only`/`solo_only` recipes** (the recipe's own field) still can't
  be placed in a way that contradicts them — see the last two bullets under
  **Placement** above.
