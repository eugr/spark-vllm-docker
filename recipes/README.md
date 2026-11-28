# Recipes

## NVIDIA Local AI Inference Recipes

`run-nvidia-recipe.sh` accepts a local NVIDIA recipe YAML file, an HTTPS raw
YAML URL, or a GitHub `blob` URL. It converts the NVIDIA `runtime.image` and
`serve.command` fields into this repository's recipe format, then passes the
result to `run-recipe.sh`. For example:

```bash
./run-nvidia-recipe.sh https://github.com/NVIDIA/local-ai-inference-recipes/blob/main/models/qwen/qwen3.8-27b/dgx-spark-2gpu/linux/vllm/128gb/nvfp4/recipe.yaml --dry-run --config /dev/null -n 10.0.0.1,10.0.0.2
./run-nvidia-recipe.sh ./nvidia-recipe.yaml --solo -t my-vllm-image
./run-nvidia-recipe.sh ./nvidia-recipe.yaml --export ./converted.yaml
./run-nvidia-recipe.sh ./nvidia-recipe.yaml --export
```

The NVIDIA image is used by default. `-t IMAGE` overrides it.
`--remap-container` maps `eugr/spark-vllm-b12x:*` to `vllm-node-b12x`,
`eugr/spark-vllm:*` to `vllm-node`, and every other image to `vllm-node`.
An explicit `-t` takes precedence over the mapping. `--export FILE` writes a
standalone internal recipe and refuses to overwrite an existing file.
`--export` without a filename prints the YAML to stdout. Neither form launches
the recipe. Other launch options pass through to `run-recipe.sh`.

The translator uses the main `serve.command`, transfers simple `export KEY=VALUE`
lines to recipe environment variables, and removes vLLM rank, master, headless,
and distributed-backend flags supplied by our launcher. It sets solo or cluster
constraints from `topology.node_count`; one-node recipes launch in solo mode by
default. For a cluster dry run, provide synthetic nodes and `--config /dev/null`
so the command does not read your local cluster configuration.

The converted recipe tracks the main checkpoint for the runner's model cache
check. If the NVIDIA recipe uses a separate draft checkpoint, download that
model separately before launching. For an unmodified NVIDIA image, the runner
pulls it when missing and copies it to cluster workers. Remapped or overridden
local images follow the normal build workflow.

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

# First cluster deployment: discover once, then run from saved configuration
./run-recipe.sh --discover
./run-recipe.sh minimax-m2-awq --setup
```

### Inkling 2 Mini on a Single DGX Spark

The [Inkling 2 Mini recipe](inkling-2-mini.yaml) serves text and image inputs from
the official Thinking Machines checkpoint. It uses one GPU (TP1), a 262,144-token
context, the Inkling reasoning parser, and seven DSpark draft tokens with
probabilistic sampling. The draft is bundled with the checkpoint.
[vLLM's Spark configuration](https://github.com/vllm-project/vllm/blob/main/recipes/models/thinkingmachines/Inkling-2-Mini.yaml)

Setup pulls the prebuilt `vllm/vllm-openai:inkling-2-mini` image and downloads the
checkpoint. The recipe sets vLLM's GPU memory utilization to `0.8` to leave
headroom on Spark's shared memory.
[vLLM's Inkling 2 Mini deployment guide](https://vllm.ai/blog/2026-11-20-inkling2-mini)

```bash
./run-recipe.sh inkling-2-mini --solo --setup
```

Automatic tool calling is enabled by default with the Inkling tool parser.

For Inco AI's external draft, use the
[Inkling 2 Mini DFlash2 recipe](inkling-2-mini-dflash2.yaml). It pins the vLLM
image, target revision, and
[draft checkpoint](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2). It
applies the Inkling capture and scoring adapter and uses seven probabilistic
draft tokens with the same 262,144-token context.

```bash
./run-recipe.sh inkling-2-mini-dflash2 --solo --setup
```

The recipe uses Humming for Inkling's NVFP4 experts. In a single-Spark
comparison with surrounding controls, Humming improved 25.5K-token prefill by
20% and four-request throughput by 10%. Pass `-- --moe-backend auto` to use the
original expert backend selection.
[Benchmark findings](../docs/inkling-2-mini-spark-findings.html)

Two optimized single-Spark variants layer [mods/inkling2-perf-opt](../mods/inkling2-perf-opt/README.md)
on top of the DFlash2 recipe:

- [Inkling 2 Mini DFlash2 Lossless](inkling-2-mini-dflash2-lossless.yaml):
  quantizes only the DFlash2 draft model at load time (`NVFP4` draft `lm_head` +
  `MXFP8` 5-layer draft backbone), keeping the target model and target `lm_head`
  untouched (**4/4 bitwise-identical greedy hashes** with the baseline DFlash2
  recipe; **+5.9%** coding decode, **+6.5%** sampled reasoning, **+9.2%** decode
  at 41K context).
- [Inkling 2 Mini DFlash2 MXFP8](inkling-2-mini-dflash2-mxfp8.yaml): fastest
  tested single-Spark configuration. Adds Humming `MXFP8` on the four coda MoE
  layers with cached W4A4 alignment, online block-32 `MXFP8 W8A16` on target
  dense attention projections (`qkvr`/`wo_ud`) and `lm_head`, and sparse prefill
  CUDA graphs (**+31.6%** coding decode at `73.40 tok/s`, **+34.4%** sampled
  reasoning at `68.41 tok/s`, **+16.9%** 4-req and **+7.1%** 8-req throughput,
  **+8.2%** 32K prefill, and **+27.8%** decode at 41K context). Pass
  `-e INKLING2_OPT_TARGET_DENSE=mxfp8` to keep the target `lm_head` in BF16, or
  `-- --max-num-batched-tokens 4096` for peak 32K prefill (`3,336 tok/s`).

```bash
# Strictly target-lossless (quantized DFlash2 draft only)
./run-recipe.sh inkling-2-mini-dflash2-lossless --solo --setup

# Fastest overall (full Humming MXFP8 dense + coda + quantized DFlash2)
./run-recipe.sh inkling-2-mini-dflash2-mxfp8 --solo --setup
```

vLLM fetches the draft checkpoint separately during startup. To pre-cache it for
offline use, download the pinned revision into the Hugging Face cache before launch.

```bash
uvx hf download incoai/Inkling-2-Mini-DFlash2 config.json model.safetensors \
  --revision 4f9088ea66c21f2b99ed657d1941c306b03e84d2
```

### Nemotron 3.5 on a Single DGX Spark

These recipes use the standard `vllm-node` image and require `--solo`:

- [Nemotron 3.5 Super VL NVFP4](nemotron-3.5-super-vl-nvfp4.yaml):
  `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`, with image and video input.
- [Nemotron 3.5 Bolt 24GB](nemotron-3.5-bolt-24gb.yaml):
  `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB`.

Both use their corresponding LiLiCorr draft model with DFlash and seven
speculative tokens, FP8 KV cache, InstantTensor loading, automatic context
length, eight concurrent sequences, and GPU memory utilization of `0.8`.

```bash
./run-recipe.sh nemotron-3.5-super-vl-nvfp4 --solo --setup
./run-recipe.sh nemotron-3.5-bolt-24gb --solo --setup
```

Run one recipe at a time. Omit `--setup` when the image and model are already
available, or use `--dry-run` instead to inspect the generated command.

## Cluster Node Discovery

Autodiscovery is the default way to configure cluster nodes:

```bash
# Auto-discover nodes and save to .env
./run-recipe.sh --discover

# Run recipe (uses nodes from .env automatically)
./run-recipe.sh minimax-m2-awq --setup
```

When you run `--discover`, it:
1. Detects active CX7 interfaces and determines mesh vs. standard topology.
2. Scans the network for peers that are both SSH-reachable **and** have an NVIDIA GB10 GPU.
3. In mesh mode, separately discovers `COPY_HOSTS` on the direct IB-attached interfaces.
4. Prompts for per-node confirmation for `CLUSTER_NODES` and `COPY_HOSTS`.
5. Saves the full configuration (including mesh NCCL settings if applicable) to `.env`.

Future recipe runs automatically use nodes from `.env`. Do not pass IP addresses
with `-n` unless you were specifically instructed to use a manual node list or
the network topology prevents autodiscovery from working.

When distributing the container image or model files, the runner uses `COPY_HOSTS` from `.env` (which may differ from `CLUSTER_NODES` in mesh mode) to ensure transfers go over the fastest available path.

## Workflow Modes

### Solo Mode (Single Node)
```bash
# Explicitly run in solo mode
./run-recipe.sh glm-4.7-flash-awq --solo
```

### Cluster Mode (Multiple Nodes)
```bash
# Discover nodes and network topology on first use
./run-recipe.sh --discover  # First time only

# Deploy using the saved .env automatically
./run-recipe.sh minimax-m2-awq --setup
```

Manual `-n` / `--nodes` input is a fallback for explicitly instructed or
non-discoverable network topologies, not the normal cluster workflow.

When using cluster mode with `--setup`:
- Container is built locally and copied to all worker nodes
- Model is downloaded locally and copied to all worker nodes

### Cluster-Only Recipes

Some models are too large to run on a single node. These recipes have `cluster_only: true` and will fail with a helpful error if you try to run them in solo mode:

```bash
$ ./run-recipe.sh minimax-m2-awq --solo
Error: Recipe 'MiniMax-M2-AWQ' requires cluster mode.
This model is too large to run on a single node.

Options:
  1. Auto-discover and save:  ./run-recipe.sh --discover
     Then run:                ./run-recipe.sh minimax-m2-awq
  2. If autodiscovery cannot support the topology, specify nodes as instructed.
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
pull_image: false                 # Pull an upstream image; copy to workers instead of building
cluster_only: false                # Set to true if model requires cluster mode
build_args:                        # Extra args for build-and-copy.sh
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
| `--exp-mxfp4` | Use MXFP4 Dockerfile (for MXFP4 quantized models) |
| `--exp-b12x` | Use the B12X image profile, which defaults to the `vllm-node-b12x` tag |
| `--use-wheels` | Build the runner image from prebuilt or local wheels instead of pulling `eugr/spark-vllm:latest` |
| `--apply-vllm-pr PR` | Build vLLM with a specific upstream pull request (experimental recipes) |

For example, the single-Spark Qwen3.8 DFlash2 recipe keeps the target's FP8
`lm_head` and uses a separate image built from the pending DFlash2 pull request:

```bash
./run-recipe.sh qwen3.8-27b-nvfp4-dflash2-fp8-head --solo --setup
```

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
  --ray                       Opt into Ray for multi-node vLLM
  --no-ray                    Default multi-node no-Ray mode (accepted for compatibility)
  -n, --nodes IPS             Manual fallback/override: comma-separated node IPs
                              (first = head)
  -d, --daemon                Run in daemon mode
  -t, --container IMAGE       Override container from recipe
  --name NAME                 Override container name
  --nccl-debug LEVEL          NCCL debug level (VERSION, WARN, INFO, TRACE)
  --apply-mod PATH            Apply an extra mod directory or zip (repeatable)
  --apply-vllm-pr PR_OR_URL   Apply a vLLM PR number or full GitHub PR URL
                              (repeatable, runtime-only, ordered with --apply-mod)
  -p, --publish HOST:CONTAINER
                              Publish a container port in solo mode (repeatable)
  -v, --volume LOCAL:CONTAINER
                              Map a volume using Docker syntax (repeatable)
  --master-port PORT          Cluster coordination port: Ray head port or PyTorch
                              distributed master port (default: 29501).
                              Alias: --head-port
  --eth-if IFACE              Override Ethernet interface
  --ib-if IFACE               Override InfiniBand interface
  -e VAR=VALUE                Pass environment variable to container (repeatable)
  -j N                        Number of parallel build jobs
  --no-cache-dirs             Do not mount ~/.cache/vllm, ~/.cache/flashinfer, ~/.cache/b12x, ~/.triton, ~/.tilelang
  --keep-entrypoint           Keep the Docker image entrypoint
  --earlyoom                  Run earlyoom as the container foreground process
  --earlyoom-args ARGS        Arguments passed to earlyoom
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

`--earlyoom` uses the same optional monitor as `launch-cluster.sh`. The default arguments are `-M 524288,102400 -s 100 -r 60`; override them with `--earlyoom-args "..."` or `VLLM_SPARK_EARLYOOM_ARGS`. `-M` values are KiB, so the default sends SIGTERM below 512 MiB available memory and SIGKILL below 100 MiB. For example:

```bash
./run-recipe.sh minimax-m2-awq --solo \
  --earlyoom --earlyoom-args "-M 786432,196608 -s 100 -r 120"
```

`--apply-vllm-pr` is a launch option, including when combined with `--setup`.
It creates an ephemeral runtime mod and does not pass the PR to the image build.
Bare numbers select PRs from `vllm-project/vllm`. A full public
`https://github.com/OWNER/REPO/pull/NUMBER` URL downloads the PR from the named
repository instead.
It accepts Python/package-only changes under `vllm/`; tests, docs, CI files,
and `setup.py` are ignored. The container must already provide dependencies
declared by an ignored `setup.py` change. PRs that modify native code or other
build/source-tree files must use `build-and-copy.sh --apply-vllm-pr` instead.
Recipe-declared mods are applied first, then command-line `--apply-mod` and
`--apply-vllm-pr` layers in their specified order:

```bash
./run-recipe.sh my-recipe --solo \
  --apply-mod mods/use-official-vllm \
  --apply-vllm-pr https://github.com/local-inference-lab/vllm/pull/669 \
  --apply-mod mods/my-followup-fix
```

## Environment Variables and Volumes

Both `run-recipe.sh` and `launch-cluster.sh` accept repeatable `-e VAR=VALUE`
container environment variables and repeatable `-v LOCAL:CONTAINER` Docker
volume mappings. Put `launch-cluster.sh` options before `exec`.

Recipe `env:` entries are passed to every container using `-e`, so they are also
available to mods and Ray workers. Explicit `run-recipe.sh -e VAR=VALUE` arguments
override recipe values; if a variable is repeated, the last CLI value wins.
These values override saved `CONTAINER_*` settings as well. Put recipe overrides
before the `--` separator for extra vLLM arguments.

The [Qwen3.8 Flash Next solo recipe](qwen3.8-flash-next-nvfp4-solo.yaml) and
[GLM 5.3 Flash recipe](glm-5.3-flash.yaml)
set `PYTORCH_ALLOC_CONF=per_process_memory_fraction:0.8` to cap total PyTorch
CUDA reservation, including cached memory, during A16 weight preparation on
Spark's shared RAM. This preserves A16 decode and compressed scales. The cap
also applies during serving and is separate from vLLM's
`--gpu-memory-utilization` budget; increasing `--gpu-mem` does not raise it.
Override it explicitly with `-e PYTORCH_ALLOC_CONF=per_process_memory_fraction:VALUE`
when tuning memory use. This setting is specific to these recipes.

B12X source builds also limit each autotuning batch to one challenger plus
the incumbent. This bounds simultaneous trial buffers for large vision shapes
while retaining autotuning and image/video support. It can increase tuning time.

The DeepSeek V4 Flash 0731 and Vision recipes use `gpu_memory_utilization: 0.8`
to leave room for host and distributed-worker allocations in Spark's shared
RAM. A higher budget can fail vLLM's free-memory check on a worker before
weights load, leaving the head with a Gloo peer-disconnect error. Override with
`--gpu-mem VALUE` only when every node has enough free memory at startup.

The Qwen3.8 Flash Next cluster recipe includes
`--additional-config '{"ple_table_memory":"device"}'` in its command, so a
normal recipe launch allocates PLE tables on the GPU. This explicit setting
takes precedence over `VLLM_PLE_TABLE_MEMORY` environment overrides.

To override the cluster recipe with disk offload:

```bash
./run-recipe.sh qwen3.8-flash-next-nvfp4-cluster \
  -e B12X_DISK_BACKEND=io_uring \
  -- --additional-config '{"ple_table_memory":"disk"}'
```

```bash
./run-recipe.sh my-recipe --solo \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/models:/models"

./launch-cluster.sh --solo \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/models:/models" \
  exec vllm serve /models/my-model --port 8000
```

In cluster mode, each volume mapping is applied to every node. Ensure the local
path exists with suitable contents and permissions on each host.

For a gated Hugging Face model, export the token for host-side downloads and
also pass it to the container:

```bash
export HF_TOKEN="YOUR_TOKEN"
./run-recipe.sh my-recipe -e HF_TOKEN="$HF_TOKEN" --setup
```

Avoid printing or sharing the expanded command because it contains the token.

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

These arguments are appended to the end of the generated vLLM command after all template substitutions. If an option occurs more than once, vLLM uses its last occurrence.

**Duplicate Detection**: If you pass an argument that conflicts with a CLI override (e.g., `--port` when you also used `--port`), the runner warns that the later extra argument wins. Prefer the built-in `--port`, `--host`, `--tp`, `--gpu-mem`, and `--max-model-len` overrides when they cover the setting you need.

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
container: vllm-node

# New recipes should use the default vllm-node image and omit legacy TF5 build args.

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
