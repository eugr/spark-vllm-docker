# Recipes

Recipes provide a **one-click solution** for deploying models with pre-configured settings. Each recipe is a YAML file that specifies:

- HuggingFace model to download
- Container image and build arguments
- Required mods/patches
- Default parameters (port, host, tensor parallelism, etc.)
- Environment variables
- The vLLM serve command

## Quick Start

```bash
# List available recipes
./run-recipe.sh --list

# Run a recipe in solo mode (single node)
./run-recipe.sh glm-4.7-flash-awq --solo

# Full setup: build container + download model + run
./run-recipe.sh glm-4.7-flash-awq --solo --setup

# Run with overrides
./run-recipe.sh glm-4.7-flash-awq --solo --port 9000 --gpu-mem 0.8

# Cluster deployment
./run-recipe.sh glm-4.7-nvfp4 -n 192.168.1.10,192.168.1.11 --setup
```

## Cluster Node Discovery

The recipe runner can automatically discover cluster nodes:

```bash
# Auto-discover nodes and save to .env
./run-recipe.sh --discover

# Show current .env configuration
./run-recipe.sh --show-env

# Run recipe (uses nodes from .env automatically)
./run-recipe.sh glm-4.7-nvfp4 --setup
```

When you run `--discover`, it:
1. Detects active CX7 interfaces and determines mesh vs. standard topology.
2. Scans the network for peers that are both SSH-reachable **and** have an NVIDIA GB10 GPU.
3. In mesh mode, separately discovers `COPY_HOSTS` on the direct IB-attached interfaces.
4. Prompts for per-node confirmation for `CLUSTER_NODES` and `COPY_HOSTS`.
5. Saves the full configuration (including mesh NCCL settings if applicable) to `.env`.

Future recipe runs will automatically use nodes from `.env` unless you specify `-n` or `--solo`.

When distributing the container image or model files, the runner uses `COPY_HOSTS` from `.env` (which may differ from `CLUSTER_NODES` in mesh mode) to ensure transfers go over the fastest available path.

## Workflow Modes

### Solo Mode (Single Node)
```bash
# Explicitly run in solo mode
./run-recipe.sh glm-4.7-flash-awq --solo

# If no nodes configured, defaults to solo
./run-recipe.sh minimax-m2-awq
```

### Cluster Mode (Multiple Nodes)
```bash
# Specify nodes directly (first IP is head node)
./run-recipe.sh glm-4.7-nvfp4 -n 192.168.1.10,192.168.1.11 --setup

# Or use auto-discovered nodes from .env
./run-recipe.sh --discover  # First time only
./run-recipe.sh glm-4.7-nvfp4 --setup
```

When using cluster mode with `--setup`:
- Container is built locally and copied to all worker nodes
- Model is downloaded locally and copied to all worker nodes

### Cluster-Only Recipes

Some models are too large to run on a single node. These recipes have `cluster_only: true` and will fail with a helpful error if you try to run them in solo mode:

```bash
$ ./run-recipe.sh glm-4.7-nvfp4 --solo
Error: Recipe 'GLM-4.7-NVFP4' requires cluster mode.
This model is too large to run on a single node.

Options:
  1. Specify nodes directly:  ./run-recipe.sh glm-4.7-nvfp4 -n node1,node2
  2. Auto-discover and save:  ./run-recipe.sh --discover
     Then run:                ./run-recipe.sh glm-4.7-nvfp4
```

## Setup Options

| Flag | Description |
|------|-------------|
| `--setup` | Full setup: build (if missing) + download (if missing) + run |
| `--build-only` | Only build/copy the container, don't run |
| `--download-only` | Only download/copy the model, don't run |
| `--force-build` | Rebuild even if container exists |
| `--force-download` | Re-download even if model exists |
| `--dry-run` | Show what would happen without executing |

## Recipe Format

```yaml
# Required fields
name: Human-readable name
container: docker-image-name
command: |
  vllm serve model/name \
      --port {port} \
      --host {host}

# Optional fields
description: What this recipe does
model: org/model-name              # HuggingFace model ID for --setup downloads
cluster_only: false                # Set to true if model requires cluster mode
build_args:                        # Extra args for build-and-copy.sh
  - --pre-tf                       # e.g., for transformers 5.0
  - --exp-mxfp4                    # e.g., for MXFP4 Dockerfile
mods:
  - mods/some-patch
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.85
  max_model_len: 32000
env:
  SOME_VAR: "value"
```

### Build Arguments

The `build_args` field passes flags to `build-and-copy.sh`:

| Flag | Description |
|------|-------------|
| `--pre-tf` | Use transformers 5.0 (required for GLM-4.7 models) |
| `--exp-mxfp4` | Use MXFP4 Dockerfile (for MXFP4 quantized models) |
| `--use-wheels` | Use pre-built wheels instead of building from source |

### Parameter Substitution

Use `{param_name}` in the command to substitute values from defaults or CLI overrides:

```yaml
defaults:
  port: 8000
  tensor_parallel: 2

command: |
  vllm serve my/model \
      --port {port} \
      -tp {tensor_parallel}
```

Override at runtime:
```bash
./run-recipe.sh my-recipe --port 9000 --tp 4
```

## CLI Reference

```
Usage: ./run-recipe.sh [OPTIONS] [RECIPE]

Cluster discovery:
  --discover                  Auto-detect cluster nodes and save to .env
  --show-env                  Show current .env configuration
  --config FILE               Path to .env configuration file (default: .env in repo directory)

Recipe overrides:
  --port PORT                 Override port
  --host HOST                 Override host
  --tensor-parallel, --tp N   Override tensor parallelism
  --gpu-memory-utilization N  Override GPU memory utilization (--gpu-mem)
  --max-model-len N           Override max model length

Setup options:
  --setup                     Full setup: build + download + run
  --build-only                Only build/copy container, don't run
  --download-only             Only download/copy model, don't run
  --force-build               Rebuild even if container exists
  --force-download            Re-download even if model exists

Launch options:
  --solo                      Run in solo mode (single node, no Ray)
  --no-ray                    Multi-node without Ray (PyTorch distributed backend)
  -n, --nodes IPS             Comma-separated node IPs (first = head)
  -d, --daemon                Run in daemon mode
  -t, --container IMAGE       Override container from recipe
  --name NAME                 Override container name
  --nccl-debug LEVEL          NCCL debug level (VERSION, WARN, INFO, TRACE)
  --master-port PORT          Cluster coordination port: Ray head port or PyTorch
                              distributed master port (default: 29501).
                              Alias: --head-port
  --eth-if IFACE              Override Ethernet interface
  --ib-if IFACE               Override InfiniBand interface
  -e VAR=VALUE                Pass environment variable to container (repeatable)
  -j N                        Number of parallel build jobs
  --no-cache-dirs             Do not mount ~/.cache/vllm, ~/.cache/flashinfer, ~/.triton
  --non-privileged            Run container without --privileged
  --mem-limit-gb N            Memory limit in GB (only with --non-privileged)
  --mem-swap-limit-gb N       Memory+swap limit in GB (only with --non-privileged)
  --pids-limit N              Process limit (only with --non-privileged)
  --shm-size-gb N             Shared memory size in GB (only with --non-privileged)

Extra vLLM arguments:
  -- ARGS...                  Pass additional arguments directly to vLLM

Other:
  --dry-run                   Show what would be executed
  --list, -l                  List available recipes
```

## Extra vLLM Arguments

Use the Unix-style `--` separator to pass additional arguments directly to vLLM. Any arguments after `--` are appended verbatim to the vLLM command.

```bash
# Override load format
./run-recipe.sh my-recipe --solo -- --load-format safetensors

# Set a custom served model name
./run-recipe.sh my-recipe --solo -- --served-model-name my-api-name

# Configure CUDA graph mode
./run-recipe.sh my-recipe --solo -- -cc.cudagraph_mode=PIECEWISE

# Multiple extra arguments
./run-recipe.sh my-recipe --solo -- --load-format auto --enforce-eager --seed 42
```

These arguments are appended to the end of the generated vLLM command after all template substitutions.

**Duplicate Detection**: If you pass an argument that conflicts with a CLI override (e.g., `--port` when you also used `--port`), a warning will be shown since your CLI override value may be replaced by the extra arg.

## Creating a Recipe

1. Create a new `.yaml` file in `recipes/`
2. Specify required fields: `name`, `container`, `command`
3. Add `build_args` if your model needs special build options
4. Add `mods` if your model needs patches
5. Set `cluster_only: true` if model is too large for single node
6. Set sensible `defaults`
7. Add `env` variables if needed

Example:
```yaml
name: My Model
description: My custom model setup
container: vllm-node-tf5

build_args:
  - --pre-tf

mods:
  - mods/my-fix

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  gpu_memory_utilization: 0.85

command: |
  vllm serve org/my-model \
      --port {port} \
      --host {host} \
      -tp {tensor_parallel} \
      --gpu-memory-utilization {gpu_memory_utilization}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  autodiscover.sh                                        │
│  - Interface detection (standard / mesh topology)       │
│  - GB10 peer verification via SSH                       │
│  - CLUSTER_NODES and COPY_HOSTS discovery               │
│  - Interactive .env save with per-node confirmation     │
└──────────────────────────┬──────────────────────────────┘
                           │ sourced by
                           ▼
┌─────────────────────────────────────────────────────────┐
│  run-recipe.sh / run-recipe.py                          │
│  - Parses YAML recipe                                   │
│  - Loads / triggers cluster discovery (--discover)      │
│  - Handles --setup (build + download + run)             │
│  - Generates launch script from template                │
│  - Applies CLI overrides                                │
└──────────┬────────────────────────┬─────────────────────┘
           │ calls (for build)      │ calls (for download)
           ▼                        ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  build-and-copy.sh   │  │  hf-download.sh               │
│  - Docker build      │  │  - HuggingFace model download │
│  - Copy to COPY_HOSTS│  │  - Rsync to COPY_HOSTS        │
└──────────────────────┘  └───────────────────────────────┘
           │
           │ then calls (for run)
           ▼
┌─────────────────────────────────────────────────────────┐
│  launch-cluster.sh                                      │
│  - Cluster orchestration                                │
│  - Container lifecycle (trimmed to required node count) │
│  - Mod application                                      │
│  - Launch script execution                              │
└─────────────────────────────────────────────────────────┘
```

This separation follows the Unix philosophy: `run-recipe.sh` provides convenience, while the underlying scripts remain focused on their specific tasks.
