# spark — Fleet Manager

`spark` is a CLI tool for managing multi-model vLLM deployments across a DGX Spark fleet.

## Architecture

```
control machine ──ssh──> master ──ssh──> fleet nodes (DGX Sparks)
                            │
                     run-recipe.sh
                     docker run
```

There are two roles:

- **Control machine** — where you run `./spark`. It orchestrates everything over SSH.
  Can be your laptop, a management server, or one of the fleet nodes itself.
- **Master** — the host that actually runs `run-recipe.sh` and `docker run` commands.
  By default this is the first node listed in `fleet.yaml`. Override with `--master`.

The control machine never loads models itself. It pushes configurations and launches
containers by SSHing into the master, which in turn SSHs into fleet nodes to manage Docker.

### Typical setups

| Setup | Control machine | Master |
|-------|----------------|--------|
| Laptop → cluster | Your laptop | Head Spark node |
| Management server | Dedicated server | Head Spark node |
| Local (single Spark) | Your DGX Spark | `localhost` (auto-detected) |

### SSH requirements

- Passwordless SSH from the control machine to the master.
- Passwordless SSH from the master to all fleet nodes.
- The control machine does **not** need direct SSH to fleet nodes — only to the master.

```bash
./spark <command> [options]
```

## Usage Scenarios

`spark` has no hardcoded fleet size limit — it manages however many nodes you list in `fleet.yaml`.

| Scenario | Models | Nodes | How |
|----------|--------|-------|-----|
| **Single Spark, multiple models** | Many | 1 | Assign all to the same node. Co-tenant containers share the ~120 GB GPU memory. |
| **Small cluster** | Few | 2–4 | One model per node, or multi-node sharding for big models. |
| **Large fleet** | Many | 50+ | `spark apply` scales to arbitrary fleet sizes. SSH is the only requirement. |

### Single Spark example

Two models on one DGX Spark (120 GB total):

```
Node: 10.0.0.10
└─ glm-flash        ── ~40 GB
└─ nemotron-nano    ── ~40 GB
   ≈ 80 GB / 120 GB total
```

Assign both to the same node, with explicit ports:

```yaml
deployments:
  - name: glm-flash
    recipe: glm-4.7-flash-awq
    node: 10.0.0.10
    port: 8001
  - name: nemotron-nano
    recipe: nemotron-3-nano-nvfp4
    node: 10.0.0.10
    port: 8002
```

Then `./spark apply -f fleet.yaml --execute`. Both run in separate containers, co-tenanted on the same GPU. Memory budgets can be tightened later with `kv_cache_gb` and `kv_dtype` (see Phase 2 in `docs/SPARK-ROADMAP.md`).

### Multi-node cluster example

The same `fleet.yaml` pattern scales to any size. A 10-node fleet looks identical — just add nodes and assign deployments:

```yaml
fleet:
  - ip: 10.0.0.10
  - ip: 10.0.0.11
  # ... more nodes

deployments:
  - name: glm-flash
    recipe: glm-4.7-flash-awq
    node: 10.0.0.10
  - name: big-model
    recipe: qwen3.5-397b-int4-autoround
    nodes:
      - 10.0.0.11
      - 10.0.0.12
```

`spark apply` launches deployments **in parallel across nodes** and **sequentially within each node**. The tool doesn't care if you have 1 or 50 Sparks — it just SSHes into the master and orchestrates from there.

### Heterogeneous fleets

Not all nodes need to be DGX Sparks. Any machine with Docker and an SSH server can be a fleet
member — x86 workstations, discrete-GPU servers, ARM nodes, whatever you want. Your recipes just
need to target the container image for that hardware.

```yaml
fleet:
  - { ip: 10.0.0.10, mem_gb: 120 }   # DGX Spark (GB10)
  - { ip: 10.0.0.11, mem_gb: 120 }   # DGX Spark (GB10)
  - { ip: 10.0.0.12, mem_gb: 24 }    # x86 w/ RTX 4090 (24GB VRAM)

deployments:
  # Big models stay on the Sparks
  - { name: qwen35, recipe: qwen3.5-122b-a10b-nvfp4, node: 10.0.0.10, port: 8001 }

  # Smaller models, embedding, reranking — no problem on x86
  - { name: embed,  recipe: qwen3-embedding-8b-x86,       node: 10.0.0.12, port: 8005 }
  - { name: rerank, recipe: qwen3-reranker-8b-x86,        node: 10.0.0.12, port: 8002 }
```

`spark` treats every node the same regardless of architecture — it SSHes in and runs Docker.
The only requirement is that the recipe and container image match the target hardware.

## Commands

### scan

List downloaded models in the HuggingFace cache with metadata (size, quantization, capabilities).

```bash
# Basic scan
./spark scan

# Custom cache directory
./spark scan --hub /path/to/hub

# Show KV cache cost at a specific token budget
./spark scan --tokens 1M

# Machine-readable output
./spark scan --json
```

Table output includes:

| Column | Description |
|--------|-------------|
| MODEL | HuggingFace repo ID |
| SIZE | Total safetensors weight size |
| QUANT | Detected quantization format (nvfp4, fp8, awq, gptq, int4, bf16) |
| ARCH | Model architecture type |
| CTX | Max position embeddings |
| KV/1K | KV cache MB (fp8) per 1,000 tokens |
| CAPS | Capabilities: `moe`, `dense`, `vision`, `hybrid-attn`, `mtp`, `embed`, `rerank` |

### apply

Start deployments defined in a fleet YAML file (`fleet.yaml` by default).

**Dry-run (default):** prints the launch plan without executing.

```bash
./spark apply -f fleet.yaml
```

**Execute:** actually launches the containers.

```bash
./spark apply -f fleet.yaml --execute
```

Options:

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to fleet YAML (default: `fleet.yaml`) |
| `--master` | Control host that runs `run-recipe.sh` (default: first fleet node) |
| `--remote-dir` | Project directory on the master (default: `~/projects/spark-vllm-docker`) |
| `--execute` | Execute the plan (default: dry-run) |

Deployments launch **in parallel across nodes** and **sequentially within each node** (shared GPU). Containers that are already serving are skipped. State is written to `.state/cluster.json`.

### down

Stop fleet YAML deployments.

```bash
# Dry-run
./spark down -f fleet.yaml

# Execute
./spark down -f fleet.yaml --execute
```

Options:

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to fleet YAML (default: `fleet.yaml`) |
| `--execute` | Execute (default: dry-run) |

### restart

Restart a single deployment by name.

```bash
./spark restart -f fleet.yaml my-model-name
```

Options:

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to fleet YAML (default: `fleet.yaml`) |
| `--master` | Control host (default: first fleet node) |
| `--remote-dir` | Project directory on the master |

The deployment is stopped and relaunched on the same node and port. Waits for the model to become ready before returning.

## fleet.yaml format

```yaml
fleet:
  - ip: 192.168.1.10
  - ip: 192.168.1.11

deployments:
  - name: my-model
    recipe: qwen3.5-122b-fp8
    node: 192.168.1.10
    # Optional overrides
    tp: 1
    port: 8001
    max_model_len: 128000
    # Extra args passed to run-recipe.py after --
    extra_args:
      - --enable-prefix-caching
```

Multiple nodes per deployment for multi-node sharding:

```yaml
deployments:
  - name: big-model
    recipe: qwen3.5-397b-int4-autoround
    nodes:
      - 192.168.1.10
      - 192.168.1.11
    max_model_len: 128000
```

## Requirements

- **PyYAML** — auto-installed by the `spark` launcher script on first use.
- Passwordless SSH from the control machine to the master, and from the master to all fleet nodes.
- A built vLLM container image available on the master node.

## Container Naming

Each deployment gets its own container named **`spark-<deployment-name>`**, derived from the
deployment's `name:` field and sanitized to Docker-safe characters. The `spark-` prefix is
the manager's ownership namespace — `apply`, `down` and `restart` only operate on containers
matched by `docker ps --filter name=spark-`. This keeps the manager from touching
hand-started containers or the legacy `vllm_node`.

## Placement

Placement is explicit in `fleet.yaml`:

- **Solo** (`node:`) — one container on one node. The `--node <ip>` flag supports both
  local and remote solo containers. When the target is remote, the master orchestrates
  the container over SSH with no Ray or cluster peers.
- **Cluster** (`nodes:`) — sharded across multiple nodes (Ray or PyTorch distributed).
  The existing `launch-cluster.sh` path handles multi-node TP/PP/DP.

`apply` launches deployments **in parallel across nodes** and **sequentially within each
node** (shared GPU). Already-serving containers are skipped, so re-applying is idempotent.

## State

Applied deployments are tracked in `.state/cluster.json`. This file is written
automatically by `apply` and can be inspected to see what is currently deployed.

## What's not yet implemented

The full design covers more phases. See `docs/SPARK-ROADMAP.md` for:

- Phase 2 — memory budgeting + profiling (`kv_cache_gb`, `kv_dtype`, `spark profile`)
- Phase 3 — LiteLLM router (single `/v1/models` entry point)
- Phase 4 — dual-rail RoCE generalization
- Additional commands: `diff`, `sync`, `ls`, `up`
