
# vLLM Docker Optimized for DGX Spark (single or multi-node)

This repository contains the Docker configuration and startup scripts to run vLLM on DGX Spark, from a single node to multi-node clusters using Ray or vLLM's native PyTorch distributed mode. It supports InfiniBand/RDMA (NCCL), custom environment configuration, and high-performance model loading through fastsafetensors and InstantTensor.
Cluster setup supports direct connections between dual Sparks, QSFP/RoCE switch configurations, and 3-node mesh configurations.

While it was primarily developed to support multi-node inference, it works just as well on single-node setups.

If you point an AI agent at this repository to prepare a host or run a recipe,
use the [operational agent runbook](docs/AGENT_RUNBOOK.md). Agent-assisted
repository changes use the separate [development guide](docs/AGENT_DEVELOPMENT.md);
[AGENTS.md](AGENTS.md) routes agents to the appropriate guide.

## Table of Contents

- [DISCLAIMER](#disclaimer)
- [QUICK START](#quick-start-shortcut)
- [CHANGELOG](#changelog)
- [1. Building the Docker Image](#1-building-the-docker-image)
- [2. Launching the Cluster (Recommended)](#2-launching-the-cluster-recommended)
- [3. Running the Container (Manual)](#3-running-the-container-manual)
- [4. Configuration Details](#4-configuration-details)
- [5. Mods and Patches](#5-mods-and-patches)
- [6. Launch Scripts](#6-launch-scripts)
- [7. Using cluster mode for inference](#7-using-cluster-mode-for-inference)
- [8. Model Loading](#8-model-loading)
- [9. Benchmarking](#9-benchmarking)
- [10. Downloading Models](#10-downloading-models)

## DISCLAIMER

This repository is not affiliated with NVIDIA or their subsidiaries. This is a community effort aimed to help DGX Spark users to set up and run the most recent versions of vLLM on Spark cluster or single nodes. 

By default, `build-and-copy.sh` pulls the tested nightly runner image from DockerHub: `eugr/spark-vllm:latest`. Nightly images are built and tested on multiple models in both cluster and solo configuration before `latest` is advanced.
We will expand the selection of models we test in the pipeline, but since vLLM is a rapidly developing platform, some things may break.

Selecting `--exp-b12x` without local-build flags or customizations pulls the separately tested
`eugr/spark-vllm-b12x:latest` image and tags it as `vllm-node-b12x`.

If you want to build only the runner from precompiled vLLM and FlashInfer wheels, specify `--use-wheels`. This option never falls back to compiling missing wheels: if a wheel cannot be downloaded or found locally, the command stops with an error. To build the latest vLLM from the main branch, use `--rebuild-vllm`; to target a specific repository, release, or commit, set `--vllm-repo` and/or `--vllm-ref`.

Similarly, `--rebuild-flashinfer`, `--flashinfer-ref`, and `--apply-flashinfer-pr` control the FlashInfer build and force the local build path.

## QUICK START SHORTCUT

If you are here to run DeepSeek V4 Flash (07/31 version), follow these instructions, otherwise skip to the next section.

Before you start, make sure you connect your Sparks together and enable passwordless SSH as described in our [Networking Guide](docs/NETWORKING.md). You can also check out NVIDIA's [Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks), but using our guide is the best way to get started. The guide includes instructions for 3-node Spark mesh clusters.

Check out locally. Do it on the head node of the cluster.
This will build the image, download and distribute the model and launch the cluster.

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./run-recipe.sh recipes/deepseek-v4-flash-0731.yaml --setup
```

## REGULAR QUICK START

### Build

Check out locally. If using DGX Spark cluster, do it on the head node.

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

Prepare the container image.

**If you have only one DGX Spark:**

```bash
./build-and-copy.sh
```

**On DGX Spark cluster:**

Make sure you connect your Sparks together and enable passwordless SSH as described in our [Networking Guide](docs/NETWORKING.md). You can also check out NVIDIA's [Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks), but using our guide is the best way to get started. The guide includes instructions for 3-node Spark mesh clusters.

Then run the following command to pull, tag, and distribute the image across the cluster.

```bash
./build-and-copy.sh -c --copy-parallel
```

The default image preparation speed depends mostly on your Internet connection and whether `eugr/spark-vllm:latest` is already present locally.

For slower internet connections it can be faster to build from the precompiled wheels by using `--use-wheels` parameter. An initial build speed depends on your Internet connection speed and whether the base image is already present on your machine. After base image pull, the build should take only 2-3 minutes.

If `--use-wheels`, `--rebuild-vllm`, `--rebuild-flashinfer`, or another build customization is used, the script keeps the local wheel-based runner path. `--use-wheels` by itself only downloads or reuses precompiled wheels; source compilation occurs only for a dependency explicitly selected by a source-build flag. Full source rebuilds can take 20-40 minutes, but subsequent builds are faster.

### Run

**On a single node**:

`launch-cluster.sh` supports solo mode, which is now a recommended way to run the container on a single Spark:

```bash
./launch-cluster.sh --solo exec \
  vllm serve \
    QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization 0.8 \
    --load-format fastsafetensors
```

To publish the server port instead of using host networking, pass the Docker port mapping before `exec`:

```bash
./launch-cluster.sh --solo -p 8000:8000 exec \
  vllm serve \
    QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
    --port 8000 --host 0.0.0.0
```

**On a cluster**

It's recommended to download the model on one node and distribute across the cluster using ConnectX interconnect prior to launching. This is to avoid re-downloading the model from the Internet on every node in the cluster.

This repository provides a convenience script, `hf-download.sh`. The following
command will download the model and distribute it across the cluster using autodiscovery.

```bash
./hf-download.sh QuantTrio/MiniMax-M2-AWQ -c --copy-parallel
```

To launch the model:

```bash
./launch-cluster.sh exec vllm serve \
  QuantTrio/MiniMax-M2-AWQ \
  --port 8000 --host 0.0.0.0 \
  --gpu-memory-utilization 0.8 \
  -tp 2 \
  --max-model-len 128000 \
  --load-format fastsafetensors \
  --enable-auto-tool-choice --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2
```

The launcher will use the number of nodes required by the parallelism flags. In a 2-node cluster, this command uses both nodes; in a larger configured cluster, extra nodes are not utilized. Do not pass `--distributed-executor-backend`, `--nnodes`, `--node-rank`, `--master-addr`, `--master-port`, or `--headless` to `vllm serve`; `launch-cluster.sh` supplies the correct backend and per-node multiprocessing arguments automatically.

**NOTE:** do not use `--load-format fastsafetensors` if you are loading models that would take >0.85 of available RAM (without KV cache) as it may result in out of memory situation.

**Also:** You can use other vLLM containers with the launch script as long as they have `bash` available. The launcher clears image entrypoints by default, to prevent containers such as `vllm-openai` to start vLLM before all necessary initialization is complete. However, it's recommended to build the container using this repository for best compatibility and most up-to-date features.

**IMPORTANT**

You may want to prune your build cache every once in a while, especially if you've been using these container builds since the beginning.

You can check the build cache size by running:

```bash
docker system df
```

To prune the cache for the first time or if you notice unusually big cache size, use:

```bash
docker builder prune
```

Don't do it every time you rebuild, because it will slow down compilation times.

For periodic maintenance, I recommend using a filter: `docker builder prune --filter until=72h`

## CHANGELOG

### 2026-08-19

#### Launch-time vLLM PR application

`launch-cluster.sh` and `run-recipe.sh` now accept repeatable
`--apply-vllm-pr <pr-num>` options. The launcher fetches each upstream PR once,
validates that it only changes installed `vllm/` runtime files, and applies it
to every newly created container in command-line layer order. PRs that require
native compilation, dependency changes, or packaging changes are rejected with
instructions to use the existing build-time option instead.

Example:

```bash
./launch-cluster.sh --solo --apply-vllm-pr 52816 exec vllm serve Inferact/Qwen3.8-27B-NVFP4   --host 0.0.0.0   --port 8000   --trust-remote-code   --kv-cache-dtype fp8   --gpu-memory-utilization 0.7   --max-model-len 262144   --max-num-seqs 8   --max-num-batched-tokens 16384   --enable-chunked-prefill   --async-scheduling   --enable-prefix-caching   --speculative-config '{"method":"dflash","model": "z-lab/Qwen3.8-27B-DFlash2", "num_speculative_tokens":8}'   --load-format instanttensor   --reasoning-parser qwen3   --tool-call-parser qwen3_xml   --enable-auto-tool-choice
```

### 2026-08-14

#### B12X source branch update

B12X source builds now use `local-inference-lab/vllm@dev/infernal-invocation`.

### 2026-08-12

#### PyTorch 2.13 and CUTLASS DSL 4.7

Regular and B12X builds now use PyTorch 2.13.0, torchvision 0.28.0,
torchaudio 2.11.0, and CUTLASS DSL 4.7.0. 

### 2026-08-11

#### Nemotron 3.5 Lightning recipe

Added a recipe for NVIDIA's Nemotron 3.5 Lightning with DSpark and 1M context.

Quick start:

```bash
git pull
./hf-download.sh nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 # add -c for cluster
./hf-download.sh nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark # speculator model; add -c for cluster
./run-recipe.sh recipes/nemotron-3.5-lightning.yaml --solo
```

### 2026-08-06

#### GLM-5.2 NVFP4 8x Spark cluster recipe

Added the cluster-only `recipes/8x-spark-cluster/glm-5.2-nvfp4.yaml` recipe for serving `nvidia/GLM-5.2-NVFP4`. Requires 8x nodes. The recipe uses the `vllm-node-b12x` image.

```bash
./run-recipe.sh recipes/8x-spark-cluster/glm-5.2-nvfp4.yaml
```

### 2026-08-03

#### New B12X image

Added `--exp-b12x` (alias: `--experimental-b12x`) as an alternative version built from [a fork by Luke Alonso](https://github.com/local-inference-lab/vllm/tree/dev/infernal-invocation). This fork supports a collection of experimental high-performance B12X kernels for sm12x architecture.

Since it is built from a forked vLLM branch, it will be supported in parallel to the main ("regular") build, at least for the time being.

Specifying `--exp-b12x` without arguments will pull `eugr/spark-vllm-b12x:latest` from Dockerhub, which is now built and tested together with the main image by CI pipeline on nightly basis (if there are any updates to the source branch or B12X).

Add `--rebuild-vllm` to compile from the source. 

The preset defaults the image tag to `vllm-node-b12x`; an explicit `-t` still takes precedence. Additional vLLM changes can be layered onto the preset with one or more `--apply-vllm-pr` flags. PRs are applied from the upstream vLLM repository, not from Luke's fork! If any PRs are specified, the script will build from the source, not prebuilt images.

vLLM wheels are not published for this build, but it will reuse published Flashinfer wheels if you build from the source (unless `--rebuild-flashinfer` is specified).

For now, we have only one recipe using this build, with more to come.

#### DeepSeek V4 Flash 0731 B12X cluster recipe

Added the cluster-only `deepseek-v4-flash-0731` recipe for serving `deepseek-ai/DeepSeek-V4-Flash-0731` on a dual DGX Spark cluster. The recipe requires B12X container (vllm-node-b12x) that can be pulled by using `./build-and-copy.sh --exp-b12x -c` (or just allow the recipe system to pull it for you). See details on b12x build above.

```bash
./run-recipe.sh deepseek-v4-flash-0731
```

### 2026-07-30

#### Inkling Small NVFP4 support

Added the support for
`thinkingmachines/Inkling-Small-NVFP4`. It requires at least dual DGX Spark setup.

```bash
./run-recipe.sh inkling-small-nvfp4
```

Add `--setup` on the first run to prepare the container and download and
distribute the model.

#### Official vLLM image support with earlyoom, InstantTensor, and SciPy

`mods/use-official-vllm` now installs `earlyoom`, InstantTensor, and SciPy in
addition to the compatibility packages needed by other mods. This makes
`launch-cluster.sh --earlyoom`, `--load-format instanttensor`, and SciPy-based
functionality available when launching official vLLM images such as
`vllm-openai`. The Python package install pins the image's existing Torch
packages so the CUDA-enabled build is not replaced during dependency
resolution.

#### Repeatable Docker volume mappings

`launch-cluster.sh` and `run-recipe.sh` now accept repeatable `-v` / `--volume` mappings using Docker's `local_path:container_path` syntax. In cluster mode, each mapping is applied to every launched node.

```bash
./launch-cluster.sh --solo \
  -v "$PWD/models:/models" \
  exec vllm serve /models/qwen3.6-35b-a4b-nvfp4 ...
```

### 2026-07-14

#### Custom vLLM repositories and PyTorch versions

`build-and-copy.sh` can now build vLLM from a fork with `--vllm-repo`. Custom repositories bypass the shared upstream Git checkout cache, force a vLLM source build, and suppress the Dockerfile's upstream preset PRs unless `--apply-preset-vllm-prs` is explicitly requested.

`--torch-version`, `--torchvision-version`, and `--torchaudio-version` select the packages installed in both the source-build environment and final runner image. The defaults are PyTorch 2.13.0, torchvision 0.28.0, and torchaudio 2.11.0, matching current upstream vLLM CUDA requirements.

Builds from any ref in `local-inference-lab/vllm` also clone and build the `master` ref of `lukealonso/b12x` automatically. The repository produces the `b12x` distribution. The source layer is refreshed on every applicable runner build so a previously cached clone cannot hide newer upstream commits. Only the locally built B12X wheel is installed. Its CUTLASS DSL metadata is adjusted to the image-wide 4.7.0 pin before installation. The checked-out commit is recorded at `/workspace/b12x-source-commit` in the image. B12X kernels remain JIT-compiled at runtime; building its Python wheel does not add another CUDA compilation phase to the image build.

### 2026-07-10

#### Optional earlyoom monitor

Runner images now include `earlyoom`, and `launch-cluster.sh --earlyoom` can run it as the container foreground process instead of `sleep infinity`. This keeps the existing launcher flow, where Ray and vLLM are started with `docker exec`, while allowing the container to monitor low host memory continuously. `run-recipe.py` and `run-recipe.sh` pass the same `--earlyoom` and `--earlyoom-args` options through to the launcher.

The default policy is conservative and absolute-memory based: `-M 524288,102400 -s 100 -r 60`. That sends SIGTERM when available memory drops below 512 MiB, escalates to SIGKILL below 100 MiB, does not wait for swap to fill before acting, and prints one memory report per minute. You can override the default per launch with `--earlyoom-args` or set `VLLM_SPARK_EARLYOOM_ARGS`.

#### Build and runtime compatibility updates

`--gpu-arch` now also drives the NCCL `NVCC_GENCODE` build argument, so non-default local builds compile NCCL for the same target architecture as Torch and FlashInfer.

The source-build KV-cache cleanup is now embedded directly in the Dockerfile, while `mods/kv-cache-prealloc-cleanup` keeps only the runtime policy tweaks.

`mods/gpu-mem-util-gb` was refreshed against current vLLM memory profiling code so fixed-GiB GPU memory reservations continue to work with the newer startup path.

### 2026-07-02

#### Prebuilt runner image by default

`build-and-copy.sh` now pulls prebuilt `eugr/spark-vllm:latest` by default and tags it locally as `vllm-node` or the tag requested with `-t`. The `latest` tag points at the latest tested nightly image. The prebuilt image is updated at the same time as prebuilt wheels by the CI pipeline, so they all stay in sync.

Use `--use-wheels` to keep the previous wheel-based runner build path. Build customization flags such as `--exp-mxfp4`, non-default `--gpu-arch`, `--vllm-ref`, `--flashinfer-ref`, rebuild/download flags, and PR application flags also keep the local build path. `--tf5` remains a tag-compatibility alias and pulls the prebuilt image as `vllm-node-tf5`.

Copy now checks the image ID locally and on each remote host before saving the image. Hosts that already have the same image ID are skipped, and `docker save` is skipped entirely when every target is already current. `--no-build` still skips image preparation and only copies an already-local tag when needed.

### 2026-07-01

#### No-Ray is now the default multi-node backend

`launch-cluster.sh` and `run-recipe.sh` now default to no-Ray multi-node launches. Use `--ray` to opt into Ray; Ray mode ensures vLLM commands include `--distributed-executor-backend ray` when they omit it. `--no-ray` remains accepted for compatibility in multi-node launches.

#### Build and dependency updates

We now use NCCL `main` branch and include new experimental vLLM Rust frontend in the builds. DeepGEMM now tracks the `nv_dev` branch.

#### Transformers 5 flag deprecation

`--tf5`, `--pre-tf`, and `--pre-transformers` are now deprecated compatibility aliases. They no longer override dependency resolution; they only preserve the legacy default image tag. Recipes that previously used `vllm-node-tf5` now use the standard `vllm-node` image.

#### Recipe updates

Added the `gemma4-26b-a4b-nvfp4` recipe, reverted the Qwen3.6-35B-A3B-NVFP4 recipes to `--kv-cache-dtype fp8`, and cleaned up stale TF5 build args from affected recipes.

### 2026-06-22

#### Deepseek V4 Flash support

Support and a recipe for Deepseek V4 Flash has been added based on a newly merged vLLM PR. Please note that this PR requires the DeepGEMM `nv_dev` branch, which is not present in default vLLM builds, but included in this community build.

To run DSV4F you will need a Spark cluster (2 or more nodes).

To run:

```bash
git pull
./build-and-copy.sh -c
./hf-download.sh deepseek-ai/DeepSeek-V4-Flash -c
./run-recipe.sh deepseek-v4-flash --no-ray
```

#### DeepGEMM support added

vLLM now uses NVIDIA branch for DeepGEMM that includes support for sm12x GPU family, including DGX Spark.
Please note that it is currently not compatible with NVRTC compiler, so `DG_JIT_USE_NVRTC` has been turned off for new builds.

#### MiniMax AWQ Weight-Shape Loader Workaround

Added a Dockerfile-level workaround for a vLLM nightly regression where compressed-tensors MoE `weight_shape` metadata is treated as a scalar during load, causing affected MoE quantized models to fail with `shape '[]' is invalid for input of size 2`.

### 2026-06-18

#### KV Cache Preallocation Cleanup Mod & Updated Qwen3.5-397B recipe for dual Sparks

Added KV cache preallocation cleanup for source-built vLLM wheels, which clears cached CUDA allocator memory before vLLM sizes and allocates KV cache blocks. The dual-node Qwen3.5-397B INT4 AutoRound recipe also applies `mods/kv-cache-prealloc-cleanup` after `mods/gpu-mem-util-gb` for its model-specific memory policy tweaks.

The same recipe now keeps the 108 GiB startup reservation, sets `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`, and uses a manual 2.25 GiB KV-cache allocation to bypass vLLM's conservative profiler-derived KV budget. The source build handles profiling-only graph-pool cleanup before KV-cache sizing, while the recipe mod makes the env var skip CUDA graph memory profiling entirely and allows the fixed-GiB reservation argument to coexist with `--kv-cache-memory-bytes` for this model to load. This may result in a bit of swap to be used to offload unused resources, so make sure swap is enabled.

#### Added recipes for nvidia/Qwen3.6-35B-A3B-NVFP4

Added recipes for NVIDIA's mixed-precision quant for Qwen3.6-35B model. It has a higher precision and an improved performance compared to the earlier NVFP4 quants.

Use:
- `qwen3.6-35b-a3b-nvfp4` to run with MTP on
- `qwen3.6-35b-a3b-nvfp4-no-mtp` to run without MTP.

### 2026-06-10

#### DiffusionGemma Recipes and Mod

Added day0 support for Google DeepMind's DiffusionGemma model via `mods/diffusiongemma`. Check out [NVIDIA blog](https://developer.nvidia.com/blog/run-diffusiongemma-on-nvidia-for-developer-ready-high-throughput-text-generation/) for details!

Added four solo-only DiffusionGemma recipes:

- `diffusion-gemma-bf16-thinking` for `google/diffusiongemma-26B-A4B-it` with thinking enabled.
- `diffusion-gemma-bf16` for `google/diffusiongemma-26B-A4B-it` with thinking disabled.
- `diffusion-gemma-nvfp4-thinking` for `nvidia/diffusiongemma-26B-A4B-it-NVFP4` with thinking enabled.
- `diffusion-gemma-nvfp4` for `nvidia/diffusiongemma-26B-A4B-it-NVFP4` with thinking disabled.

The non-thinking variants still keep `--reasoning-parser gemma4`, since these models can emit Gemma4 channel markers even when thinking is disabled.

Example:

```bash
./hf-download.sh google/diffusiongemma-26B-A4B-it
./run-recipe.sh diffusion-gemma-bf16-thinking --solo
```

#### `run-recipe.sh` Launch Flag Passthrough

`run-recipe.sh` now passes additional `launch-cluster.sh` flags through when running recipes:
`--apply-mod`, `-p` / `--publish`, and `--keep-entrypoint`.

Port publishing is still solo-only, matching `launch-cluster.sh` behavior.

### 2026-06-09

#### Recipe Memory Defaults

Raised the default `gpu_memory_utilization` from `0.7` to `0.8` across the main single-node and two-node recipes to match the current vLLM memory allocation behavior.

### 2026-06-07

#### Docker Base Image Compatibility

The default CUDA base image was changed to `nvidia/cuda:13.0.2-devel-ubuntu24.04` for broader host compatibility.

The Dockerfile also now passes `--allow-change-held-packages` when installing the custom NCCL Debian packages, avoiding apt failures when replacing held CUDA/NCCL packages during image builds.

### 2026-06-06

#### MiniMax Multi-Node Regression Workaround

Added a targeted Dockerfile patch that disables the MiniMax QK RMSNorm CUDA IPC fused path introduced by vLLM PR #43410. The fused path can fail when tensor parallelism spans DGX Spark nodes; the workaround preserves MiniMax multi-node TP while avoiding a full upstream revert.

### 2026-06-03

#### Solo Port Publishing

`launch-cluster.sh` now supports Docker-style `-p` / `--publish` port mappings in solo mode. When port publishing is used, the launcher switches from host networking to Docker bridge networking for that solo container.

Example:

```bash
./launch-cluster.sh --solo -p 8000:8000 exec vllm serve ...
```

### 2026-05-29

#### Wheel Freshness Detection

Improved `build-and-copy.sh` wheel freshness checks so newer locally built wheels are not overwritten just because their filenames differ from the latest release assets. The script now compares local wheel mtimes against remote release asset timestamps before deciding to download. Also switched from using GitHub API to regular HTTP checks to avoid throttling.

#### `gpu-mem-util-gb` Patch Refresh

Refreshed `mods/gpu-mem-util-gb` so it applies against newer vLLM `CacheConfig` code after upstream line/context drift.

### 2026-05-28

#### StepFun Step 3.7 Flash Support

Added support for StepFun [Step 3.7 Flash multimodal model](https://developer.nvidia.com/blog/run-step-3-7-flash-on-nvidia-gpus-with-enterprise-ready-multimodal-ai).

Requires at least 2 Sparks in a cluster. Both FP8 and NVFP4 checkpoints are supported. FP8 requires more memory, so using NVFP4 is recommended.

Update the repo and build a fresh container first:

```bash
git pull
./build-and-copy.sh --cleanup -c
```

To run NVFP4 version:

Download the model:

```bash
./hf-download.sh stepfun-ai/Step-3.7-Flash-NVFP4 -c
```

Run:

```bash
./run-recipe.sh step-3.7-flash-nvfp4 --no-ray
```

To run FP8 version:

Download the model:

```bash
./hf-download.sh stepfun-ai/Step-3.7-Flash-FP8 -c
```

Run:

```bash
./run-recipe.sh step-3.7-flash-fp8 --no-ray
```

Please note that `--no-ray` is required for FP8 to fit with full context!

#### `use-official-vllm` NCCL Workaround

Updated `mods/use-official-vllm` to also handle the NCCL load-order bug tracked in [vllm-project/vllm#42354](https://github.com/vllm-project/vllm/issues/42354). When both the pip-installed `nvidia/nccl/lib/libnccl.so.2` and system `libnccl2` are present, the mod redirects the pip-installed NCCL path to the system `/usr/lib` soname, matching the manual workaround that fixes multi-node DGX Spark hangs.

Use it with official vLLM images before starting the model:

```bash
./launch-cluster.sh -t vllm/vllm-openai:latest \
  --apply-mod mods/use-official-vllm \
  exec vllm serve ...
```

#### Torch Pinning During Wheel Install

Pinned the already-installed CUDA torch build via `uv --override` when installing locally built wheels and final runtime dependencies in both Dockerfiles. This prevents transitive dependencies from re-resolving torch to a CPU wheel during image builds.

### 2026-05-22

#### New Mod: `use-official-vllm`

Added `mods/use-official-vllm`, a prerequisite mod for applying patches inside official vLLM Docker containers (e.g. `vllm-openai`). Official containers do not ship `git`, which several mods require. This mod installs `git` via `apt-get` if it is not already present.

Apply it before any other mod that requires `git`:

```bash
./launch-cluster.sh -t vllm/vllm-openai:latest \
  --apply-mod mods/use-official-vllm \
  --apply-mod mods/gpu-mem-util-gb \
  exec vllm serve ...
```

#### `gpu-mem-util-gb` Updated for Latest vLLM Main

Updated `mods/gpu-mem-util-gb` patch to apply cleanly against the current vLLM `main` branch. The mod now checks for `git` at startup and prints a hint to apply `mods/use-official-vllm` first if `git` is missing (relevant when using official vLLM containers).

### 2026-05-18

#### NCCL Updated to NVIDIA `v2.30u1`

The Dockerfile now builds NCCL from NVIDIA's `v2.30u1` branch instead of the custom NCCL fork. This branch incorporates all features of the custom fork and is based on the latest NCCL release. The networking guide's NCCL test commands have been updated to use the same branch for 3-node mesh clusters.

### 2026-05-14

#### Default Entrypoint Clearing

`launch-cluster.sh` now clears the Docker image entrypoint by default when starting idle cluster containers. This allows images with server-style entrypoints, such as `vllm-openai`, to work with the same cluster launcher flow. Use `--keep-entrypoint` to preserve the image entrypoint.

### 2026-05-10

#### Qwen3.5-397B Recipe Memory Updates

Updated Qwen3.5-397B AutoRound recipes to reduce OOM risk. The dual-node recipe now uses standard fractional `--gpu-memory-utilization`, and the 3-node pipeline-parallel recipe uses InstantTensor loading with lower memory pressure.

### 2026-05-06

#### Qwen3.6-35B-A3B-FP8 Recipes

Added `qwen3.6-35b-a3b-fp8` and `qwen3.6-35b-a3b-fp8-dflash` recipes plus a dedicated Qwen3.6 chat-template mod. The DFlash recipe has prefix caching disabled because it caused accuracy issues.

### 2026-04-29

#### Gemma4 Recipe Fixes and Experimental b12x Mod

The Gemma4-26B-A4B recipe now uses `safetensors` loading and no longer applies the obsolete tool parser mod by default.

### 2026-04-25

#### MiniMax-M2.7-AWQ Recipe

Added `minimax-m2.7-awq`, a cluster-only MiniMax M2.7 AWQ recipe using `cyankiwi/MiniMax-M2.7-AWQ-4bit`.

### 2026-04-14

Added `--load-format instanttensor` support to vLLM - thanks @SeraphimSerapis. 
An experimental option for now, but allows for faster loading than the current fastsafetensors default. You need to rebuild the container to start using the option, but you don't have to trigger the source build.

### 2026-04-12

#### Drop-caches mod for Qwen3.5-397B

Updated Qwen3.5-397B recipe (for dual node configuration) to use the new mod `mods/drop-caches` which clears filesystem caches every minute while the container is running, resolving fastsafetensors getting stuck during loading and a few other bugs when operating close to max memory limit.

### 2026-04-11

#### Pinned PyTorch Version

Pinned PyTorch to version 2.11.0 (previously using nightly builds) to fix incompatibility with transformers 5.x and avoid torch version mismatch in builds.

### 2026-04-02

A new recipe for Gemma4-26B-A4B in "on-the-fly" FP8 quantization:

Single Spark:

```bash
./run-recipe.sh gemma4-26b-a4b --solo
```

Dual Sparks: 

```bash
./run-recipe.sh gemma4-26b-a4b --no-ray
```

### 2026-03-31

#### Flags to specify Flashinfer ref and apply PRs

`build-and-copy.sh` gains two new flags that mirror the existing vLLM equivalents:

- `--flashinfer-ref <ref>` — build FlashInfer from a specific commit SHA, branch, or tag instead of `main`. Forces a local FlashInfer build (skips prebuilt wheel download).
- `--apply-flashinfer-pr <pr-num>` — fetch and apply a FlashInfer GitHub PR patch before building. Can be specified multiple times. Forces a local FlashInfer build.

Both flags are incompatible with `--exp-mxfp4`.

#### Default image tag in `build-and-copy.sh`

`build-and-copy.sh` now automatically sets a sensible default image tag when `-t` is not specified:

- `--tf5` / `--pre-tf` - deprecated compatibility flag; normal build, tag defaults to `vllm-node-tf5`
- `--exp-mxfp4` - tag defaults to `vllm-node-mxfp4`
- in all other cases - tag defaults to `vllm-node` (no change)

An explicit `-t <tag>` always takes precedence.

#### Support for 3-node mesh setups

Added initial support for setups where 3 Sparks are connected in a ring-like mesh without an additional switch.
See [Networking Guide](docs/NETWORKING.md) for instructions on how to connect and set up networking in such cluster.

Autodiscover function in both `launch-cluster.sh` and `run-recipe.sh` now can detect mesh setups and configure parameters accordingly.

You can try running a model on all 3 nodes in pipeline-parallel configuration using the following recipe:

```bash
./run-recipe.sh --discover # force mesh discovery
./run-recipe.sh recipes/3x-spark-cluster/qwen3.5-397b-int4-autoround.yaml --setup --no-ray --force-build # you can drop --setup and --force-build on subsequent calls
```

Please note that `--tensor-parallel-size 3` or `-tp 3` is not supported by any commonly used model, so the only two viable options to utilize all three nodes for a single model are:

- `--pipeline-parallel 3` will let you run a model that can't fit on dual Sparks, but without additional speed improvements (total throughtput may improve though).
- `--data-parallel 3` (possibly with `--enable-expert-parallel`) will let you run a model that can fit on a single Spark, but allow for better concurrency.

You can also run models with `--tensor-parallel 2` in a 3-node configuration - in this case only first two nodes (from autodiscovery/.env or from the CLI parameters) will be utilized.

#### GB10 Verification During Node Discovery

Node discovery now confirms each SSH-reachable peer is a GB10 system before adding it to the cluster:
Only hosts reporting `NVIDIA GB10` are included. This prevents accidentally adding non-Spark machines that happen to be on the same subnet.

#### Separate COPY_HOSTS Discovery

Autodiscover now determines the host list used for image and model distribution separately from `CLUSTER_NODES`:

- **Non-mesh**: `COPY_HOSTS` mirrors `CLUSTER_NODES` (no change in behaviour).
- **Mesh**: scans the direct IB-attached `enp1s0f0np0` and `enp1s0f1np1` interfaces (not the OOB ETH interface), so large file transfers use the faster direct InfiniBand path.

`COPY_HOSTS` is saved to `.env` and respected by `build-and-copy.sh`, `hf-download.sh`, and `run-recipe.py`.

#### Interactive Configuration Save in `autodiscover.sh`

`autodiscover.sh` now handles `.env` creation with a guided interactive flow, replacing the previous logic in `run-recipe.py`:

- Runs automatically when `.env` is absent.
- Asks per-node confirmation for both `CLUSTER_NODES` and `COPY_HOSTS`.
- Skips if `.env` already exists (use `--setup` to force).

`run-recipe.py` no longer contains its own `.env`-save prompt — it delegates entirely to `autodiscover.sh`.

#### `--setup` Flag in `launch-cluster.sh` and `build-and-copy.sh`

Both scripts now accept `--setup` to force a full autodiscovery run and overwrite the existing `.env`:

```bash
./launch-cluster.sh --setup exec vllm serve ...
./build-and-copy.sh --setup -c
```

This is equivalent to the existing `--setup` in `run-recipe.sh`.

#### `--config` Flag

`hf-download.sh`, `build-and-copy.sh` and `launch-cluster.sh` now accept `--config <file>` to load a custom `.env` configuration file. `COPY_HOSTS` from the config is used for model distribution:

```bash
./hf-download.sh QuantTrio/MiniMax-M2-AWQ --config /path/to/cluster.env -c --copy-parallel
```

#### Parallelism-Aware Node Trimming

`launch-cluster.sh` now parses `-tp` / `--tensor-parallel-size`, `-pp` / `--pipeline-parallel-size`, and `-dp` / `--data-parallel-size` from the exec command or launch script and adjusts the active node count accordingly — for both Ray and no-Ray modes.

- If **fewer nodes are needed** than configured, only the required nodes get containers started (excess nodes are left idle).
- If **more nodes are needed** than available, an error is raised before anything starts.

```
Note: Command requires 2 node(s) (tp=2 * pp=1 * dp=1); using 2 of 3 configured node(s).
Error: Command requires 4 nodes (tp=4 * pp=1 * dp=1) but only 3 node(s) are configured.
```

No flags required — the check is automatic whenever parallelism arguments are present in the command.

### 2026-03-18

#### `--master-port` / `--head-port` Parameter

Added `--master-port` (synonym: `--head-port`) to both `launch-cluster.sh` and `run-recipe.sh` to configure the port used for cluster coordination:

- In **Ray mode**: sets the Ray head node port (previously hardcoded to 6379)
- In **No-Ray mode**: sets the PyTorch distributed `--master-port` passed to vLLM

Default is `29501`.

```bash
./launch-cluster.sh --master-port 29501 --no-ray exec vllm serve ...
./run-recipe.sh qwen3.5-122b-fp8 --no-ray --master-port 29501
```

#### `--network` Parameter in Build Arguments

Added `--network <name>` to `build-and-copy.sh` to allow using host networking during builds. 
Thanks @apairmont for the PR.

### 2026-03-17

#### EXPERIMENTAL Intel/Qwen3.5-397B-A17B-int4-AutoRound Recipe

You can run full 397B Qwen3.5 model on just two Sparks with vision and full context, however you need to make sure your Sparks don't run anything extra that can take a lot of RAM. That means that you don't want to log into the graphical interface or use remote desktop. Connect to the head node via ssh.

Alternatively, you can run in non-graphical mode (runlevel 3) by using `sudo systemctl isolate multi-user.target` to switch (you can use `sudo systemctl set-default graphical.target` to switch back to graphical mode), however this is known to reduce performance a bit.

You can run the model with the following command on the head node:

```bash
./run-recipe.sh qwen3.5-397b-int4-autoround.yaml --no-ray
```

Please, note `--no-ray` is necessary to fit full context. It also improves inference speed by ~1 t/s.
By default it will try to allocate 108 GiB for vLLM on each node. You can change this by changing `gpu_memory_utilization` in the recipe or passing `--gpu-mem`; this recipe maps that value to `--gpu-memory-utilization-gb`, so it is GiB rather than a percentage.

**KNOWN ISSUES**:

1. The current firmware may cause sudden shutdown event on one or both Sparks during heavy inference. If you have this issue, you will need to lower GPU clock frequency on the affected unit(s), e.g. `sudo nvidia-smi -lgc 200,2150`. This command will reduce max GPU frequency to 2150 MHz. You can play with higher values to see what works for you (default is 2411 MHz, but can boost to 3000 MHz). Please note that this setting only survives until the next reboot, but can be applied at any time.
2. You will need to use the new `--no-ray` argument to fit full context.
3. If the model gets stuck loading weights, clearing the cache on both nodes can "unstuck" it. Use `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` to clear the cache. 


#### Major Cluster Orchestration Refactoring

Significantly refactored the internal cluster startup logic in `launch-cluster.sh`:
- Removed the standalone `run-cluster-node.sh` script; its logic is now fully integrated into `launch-cluster.sh`.
- Ray head/worker startup, environment variable injection, and launch script distribution are now handled by `launch-cluster.sh` directly.
- Worker containers are started with proper per-node environment variables (`VLLM_HOST_IP`, `NCCL_SOCKET_IFNAME`, etc.) injected via `docker run`/`docker exec` instead of relying on `.bashrc`.
- You will now be able to run other vLLM containers without applying `use-ngc-vllm` mod (current version is just an empty stub).

#### No-Ray Multi-Node Mode

Added `--no-ray` flag to `launch-cluster.sh` to run multi-node vLLM clusters without Ray, using PyTorch's native distributed backend instead. It slightly improves inference performance for most models and reduces memory requirements.

```bash
./launch-cluster.sh --no-ray exec vllm serve ...
```

`--no-ray` is incompatible with `--solo` (which already runs without Ray).

#### `run-recipe.sh` No-Ray Mode and Extended Flag Passthrough

`run-recipe.sh` now supports `--no-ray` flag for running multi-node inference without Ray (uses PyTorch distributed backend instead):

```bash
./run-recipe.sh qwen3.5-122b-fp8 --no-ray
```

The following `launch-cluster.sh` flags are now also passed through from `run-recipe.sh`:
`--master-port`, `--name`, `--eth-if`, `--ib-if`, `-j`, `--no-cache-dirs`, `--non-privileged`, `--mem-limit-gb`, `--mem-swap-limit-gb`, `--pids-limit`, `--shm-size-gb`.

#### Nemotron-3-Nano-NVFP4 Switched to Marlin Backend

The `nemotron-3-nano-nvfp4` recipe has been updated to use the Marlin backend for better performance and reliability (until Flashinfer fully supports NVFP4 on sm121).

### 2026-03-12

#### Experimental `--gpu-memory-utilization-gb` Mod

Added a new mod `mods/gpu-mem-util-gb` that adds a `--gpu-memory-utilization-gb` flag to vLLM, allowing you to specify GPU memory reservation in GiB instead of as a fraction. This is particularly useful on DGX Spark's unified memory architecture where available memory changes dynamically.

```bash
./launch-cluster.sh --apply-mod mods/gpu-mem-util-gb exec vllm serve ... \
  --gpu-memory-utilization-gb 110
```

Cannot be used simultaneously with `--kv-cache-memory-bytes`.

#### Qwen3.5-397B INT4-AutoRound TP=4 Recipe (4× Spark Cluster)

Added `recipes/4x-spark-cluster/qwen3.5-397b-int4-autoround.yaml` for running Intel/Qwen3.5-397B-A17B-int4-AutoRound across 4 DGX Spark nodes with tensor parallelism (TP=4).

Benchmarked at ~37 tok/s single-user, ~103 tok/s aggregate (4 concurrent users).

Includes a new mod `mods/fix-qwen35-tp4-marlin` that resolves a Marlin kernel constraint (`MIN_THREAD_N=64`) that breaks certain projection layers at TP=4.

**Note:** Requires NVIDIA driver 580.x. Driver 590.x has a CUDAGraph capture deadlock on GB10 unified memory.

```bash
./run-recipe.sh 4x-spark-cluster/qwen3.5-397b-int4-autoround
```
Thanks @sonusflow for the contribution.

#### Nemotron-3-Super-120B NVFP4 Recipe

Added a new recipe `nemotron-3-super-nvfp4` for running `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` with Marlin kernels. Supports both solo and cluster modes. Includes a custom reasoning parser (`super_v3_reasoning_parser.py`) fetched from the model repository. Supports both dual and single Spark configurations.

```bash
./run-recipe.sh nemotron-3-super-nvfp4
```

### 2026-03-11

#### Qwen3-Coder-Next INT4-AutoRound Recipe

Added a new recipe `qwen3-coder-next-int4-autoround` for running Intel/Qwen3-Coder-Next-int4-AutoRound. Supports single Spark only (use with `--solo` switch), since split weights are too small for Marlin kernel.

```bash
./run-recipe.sh qwen3-coder-next-int4-autoround --solo
```

### 2026-03-06

#### `-e/--env` Passthrough in `run-recipe.py`

`run-recipe.sh` now accepts one or more `-e VAR=VALUE` flags to pass environment variables directly to the container, mirroring the existing behaviour of `launch-cluster.sh`.

```bash
./run-recipe.sh qwen3.5-122b-int4-autoround --solo -e HF_TOKEN="$HF_TOKEN"
```

#### Unsloth Chat Template for Qwen3.5

Added a new mod `mods/fix-qwen3.5-chat-template` that applies the Unsloth chat template to Qwen3.5 models for better compatibility with modern clients. The template is now included in the `qwen3.5-122b-fp8`, `qwen3.5-122b-int4-autoround`, and `qwen3.5-35b-a3b-fp8` recipes.

#### Fix Shell Quoting for Exec Command Arguments

Fixed shell quoting for exec command arguments in `launch-cluster.sh` and `run-recipe.py` to correctly handle arguments containing spaces or special characters.

### 2026-03-05

#### Qwen3.5-35B-A3B-FP8 Recipe

Added a new recipe `qwen3.5-35b-a3b-fp8` for running Qwen3.5-35B-A3B in FP8 format.

```bash
./run-recipe.sh qwen3.5-35b-a3b-fp8
```

#### 4× Spark Cluster Recipes

Added a `recipes/4x-spark-cluster/` subdirectory with recipes optimised for a 4-node Spark cluster:
- `minimax-m2.5` — MiniMax M2.5 on 4× Spark
- `qwen3.5-397b-a17B-fp8` — Qwen3.5-397B-A17B in FP8 on 4× Spark

#### More Robust Wheels Check Before Download

Improved the wheels availability check in `build-and-copy.sh` to be more reliable when deciding whether to download remote wheels.

### 2026-03-04

#### Prebuilt vLLM Wheels via GitHub Releases

`build-and-copy.sh` automatically downloads prebuilt vLLM wheels from the [GitHub releases](https://github.com/eugr/spark-vllm-docker/releases/tag/prebuilt-vllm-current) for wheel-based runner builds. It does not implicitly fall back to a source build.

The download logic mirrors the FlashInfer behaviour:
- If prebuilt wheels are available and newer than any locally cached version, they are downloaded automatically.
- If the download fails (e.g. no network, release not found, GPU arch not supported), the script reuses an existing local wheel when available; otherwise it exits and identifies the explicit source-build flag.
- `--rebuild-vllm`, `--vllm-ref`, or `--apply-vllm-pr` skip the download entirely and force a local build.

No new flags are required — the download happens transparently.

All prebuilt wheels are now tested with multiple models in both solo and cluster configuration as a part of automated deployment pipeline which will now run nightly. The wheels are released only if they pass all the tests and no significant performance regressions are detected.

#### Qwen3.5-122B-FP8 Recipe

Added a new recipe `qwen3.5-122b-fp8` for running Qwen3.5-122B in FP8 format.

```bash
./run-recipe.sh qwen3.5-122b-fp8
```

### 2026-03-02

#### Qwen3.5-122B-INT4-Autoround Support

Added support for Intel/Qwen3.5-122B-A10B-int4-AutoRound model with a new mod `mods/fix-qwen3.5-autoround` that fixes a ROPE syntax error.

Recipe available at `recipes/qwen3.5-122b-int4-autoround.yaml`.

### 2026-02-26

#### Daemon Mode Improvements

- You can now use daemon mode (both solo and in the cluster) when exec action is specified.
- Piping exec command to docker logs when running in daemon mode.

### 2026-02-25

#### HF_HOME Support

Added support for using `$HF_HOME` environment variable as huggingface cache directory.

#### Intel/Qwen3-Coder-Next-INT4-Autoround Mod

Added a new mod for Intel/Qwen3-Coder-Next-INT4-Autoround model support: `mods/fix-qwen3-next-autoround`


### 2026-02-21

#### Minimax Reasoning Parser Update

Changed reasoning parser in Minimax for better compatibility with modern clients (like coding tools).

### 2026-02-18

#### Completely Redesigned Build Process

`build-and-copy.sh` automatically downloads prebuilt FlashInfer wheels from the [GitHub releases](https://github.com/eugr/spark-vllm-docker/releases/tag/prebuilt-flashinfer-current) for wheel-based runner builds. It does not implicitly fall back to a source build.

The download logic:
- If prebuilt wheels are available and newer than any locally cached version, they are downloaded automatically.
- If the download fails (e.g. no network, release not found, GPU arch is not compatible), the script reuses an existing local wheel when available; otherwise it exits and suggests `--rebuild-flashinfer`.
- `--rebuild-flashinfer` skips the download entirely and forces a fresh local build.

No new flags are required - the download happens transparently unless `--rebuild-flashinfer` is specified.

Wheels are cached by component and profile under `./.wheel-cache` for subsequent reuse.

- `--rebuild-flashinfer` will force FlashInfer rebuild from the flashinfer `main` branch.
- `--rebuild-vllm` will force vLLM rebuild from vLLM `main` branch or specific commit in `--vllm-ref`.

Please, note that specifying `--vllm-ref` or `--apply-vllm-pr` will force vLLM rebuild every time.

### 2026-02-17

#### Non-Privileged Mode Support

Added `--non-privileged` flag to `launch-cluster.sh` for running containers without full privileged access while maintaining RDMA/InfiniBand functionality:

- Replaces `--privileged` with `--cap-add=IPC_LOCK`
- Replaces `--ipc=host` with `--shm-size=64g` (configurable via `--shm-size-gb`)
- Exposes RDMA devices via `--device=/dev/infiniband`
- Adds resource limits: memory (110GB), memory+swap (120GB), pids (4096)

Example usage:
```bash
./launch-cluster.sh --non-privileged exec vllm serve ...
./launch-cluster.sh --non-privileged --mem-limit-gb 120 --shm-size-gb 64 exec vllm serve ...
```

May result in a slightly reduced performance (within 2%) in exchange for better reliability and stability.

#### Qwen3-Coder-Next recipe update

Updated `qwen3-coder-next-fp8` recipe: KV cache type changed to `fp8` and maximum context length reduced to 131072 tokens to reliably fit within a single Spark's memory.

### 2026-02-16

#### MiniMax M2.5 AWQ recipe

Added a new recipe `minimax-m2.5-awq` for running MiniMax-Text-01-AWQ (M2.5). Usage:

```bash
./run-recipe.sh minimax-m2.5-awq
```

#### GLM-4.7-Flash-AWQ mod extended with vLLM crash fix

The `fix-glm-4.7-flash-AWQ` mod now also applies the fix from [PR #34695](https://github.com/vllm-project/vllm/pull/34695), which addresses a crash in `mla_attention.py` when running GLM models with AWQ quantization. The patch is applied automatically alongside the existing speed fix, and is skipped if it has already been merged into the installed vLLM version.

### 2026-02-13

#### FlashInfer cubin caching

FlashInfer cubins (pre-compiled GPU kernels) are now cached via a Docker bind mount and reused across rebuilds. Previously, all cubins were recompiled from scratch on every FlashInfer rebuild even if unchanged. This significantly reduces FlashInfer rebuild times when only minor source changes are made.

### 2026-02-12

Added a mod for Qwen3-Coder-Next-FP8 that fixes:

- A bug with Triton allocator (https://github.com/vllm-project/vllm/issues/33857) that prevented the model to run in a cluster.
- A bug that introduced crash when `--enable-prefix-caching` is on (https://github.com/vllm-project/vllm/issues/34361).
- A bug that significantly impacted the performance on Spark (https://github.com/vllm-project/vllm/issues/34413).

This mod was included in `qwen3-coder-next-fp8` recipe.

### 2026-02-11

#### Configurable GPU Architecture

Added `--gpu-arch <arch>` flag to `build-and-copy.sh`. This allows specifying the target GPU architecture (e.g., `12.0f`) during the build process, instead of being hardcoded to `12.1a`. This argument controls both `TORCH_CUDA_ARCH_LIST` and `FLASHINFER_CUDA_ARCH_LIST` build arguments.

### 2026-02-10

#### Cache Directory Mounting

`launch-cluster.sh` now automatically mounts default cache directories to the container to improve cold start times:
- `~/.cache/vllm`
- `~/.cache/flashinfer`
- `~/.triton`
- `~/.tilelang`

To disable this behavior (clean start), use `--no-cache-dirs` flag.

### 2026-02-09

- Migrated to a new base image with PyTorch 2.10 compiled with Spark support. With this change, wheels build is no longer a recommended way - please use a source build instead.
- Triton 3.6.0 is now default.
- Removed temporary fastsafetensors patch, as proper fix is now merged into vLLM main branch.

### 2026-02-04

#### Recipes support

A major contribution from @raphaelamorim - model recipes. 
Recipes allow to launch models with preconfigured settings with one command.

Example:

```bash
# List available recipes
./run-recipe.sh --list

# Run a recipe in solo mode (single node)
./run-recipe.sh glm-4.7-flash-awq --solo

# Full setup: build container + download model + run
./run-recipe.sh glm-4.7-flash-awq --solo --setup

# Run with overrides
./run-recipe.sh glm-4.7-flash-awq --solo --port 9000 --gpu-mem 0.8

# Apply an extra launch-cluster mod on top of recipe mods
./run-recipe.sh glm-4.7-flash-awq --solo --apply-mod mods/use-official-vllm

# Publish ports in solo mode
./run-recipe.sh glm-4.7-flash-awq --solo -p 8000:8000

# Cluster deployment
./run-recipe.sh glm-4.7-nvfp4 --setup
```

Please refer to the [documentation](recipes/README.md) for the details.

#### Launch script option

You can now specify a launch script to execute on head node instead of specifying a command directly via `exec` action. 
Example: 

```bash
./launch-cluster.sh --launch-script examples/vllm-openai-gpt-oss-120b.sh
```

Thanks @raphaelamorim for the contribution!


#### Ability to apply vLLM PRs during build

`./build-and-copy.sh` now supports ability to apply vLLM PRs to builds. PR patches are applied to the selected vLLM ref (`--vllm-ref`, default `main`) without carrying the PR branch's original base history. This does NOT apply to MXFP4 special build!

To use, just specify `--apply-vllm-pr <pr_num>` in the arguments. Dockerfile preset vLLM PRs are applied automatically for an ordinary `main` source build. Specifying either `--vllm-ref` or `--apply-vllm-pr` suppresses the preset PRs unless `--apply-preset-vllm-prs` is also specified; when enabled, both preset and requested PR patches are applied on top of the selected vLLM ref. Please note that a PR patch may fail if it does not apply cleanly to the selected ref. Use with caution!

Example:

```bash
./build-and-copy.sh -t vllm-node-20260204-pr31740 --apply-vllm-pr 31740 -c
```

### 2026-02-02

#### Nemotron Nano mod

Added a mod for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B support. It supports all Nemotron Nano models/quants using the same reasoning parser.
To use, add `--apply-mod mods/nemotron-nano` to `./launch-cluster.sh` arguments.

For example, to run nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 on a single node:

```bash
./launch-cluster.sh --solo --apply-mod mods/nemotron-nano \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e VLLM_FLASHINFER_MOE_BACKEND=throughput \
  exec vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --max-num-seqs 8 \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --port 8888 --host 0.0.0.0 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser-plugin nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.7 \
    --load-format fastsafetensors 
```

Please note, that NVFP4 models on Spark are not fully supported on vLLM (any build) yet, so the performance will not be optimal. You will likely see Flashinfer errors during load. This model is also known to crash sometimes.

#### Ability to use launch-cluster.sh with NVIDIA NGC containers

Added a new mod that enables using cluster launch script with NVIDIA NGC vLLM or any other vLLM container that includes Infiniband libraries and Ray support.

To use, add `--apply-mod mods/use-ngc-vllm` to `./launch-cluster.sh` arguments. It can be combined with other mods.
For example, to launch Nemotron Nano in the cluster using NGC container, you can use the following command:

```bash
./launch-cluster.sh \
   -t nvcr.io/nvidia/vllm:26.01-py3 \
   --apply-mod mods/use-ngc-vllm \
   --apply-mod mods/nemotron-nano \
   -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
   -e VLLM_FLASHINFER_MOE_BACKEND=throughput \
   exec vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
       --max-model-len 262144 \
       --port 8888 --host 0.0.0.0 \
       --trust-remote-code \
       --enable-auto-tool-choice \
       --tool-call-parser qwen3_coder \
       --reasoning-parser-plugin nano_v3_reasoning_parser.py \
       --reasoning-parser nano_v3 \
       --kv-cache-dtype fp8 \
       --gpu-memory-utilization 0.7 \
       --tensor-parallel-size 2 \
       --distributed-executor-backend ray
```

Make sure you have the container pulled on both nodes!

At this point it doesn't seem like NGC container performs any better for this model than a custom build.

### 2026-01-29

#### New Parameters for launch-cluster.sh

- Added **solo mode** to `launch-cluster.sh` to launch models on a single node. Just use `--solo` flag  or if you have only a single Spark, it will default to Solo mode if no other nodes are found.
- Added `-e` / `--env` parameter to `launch-cluster.sh` to pass environment variables to the container.

#### New Mod for GLM-4.7-Flash-AWQ

Added a mod to prevent severe inference speed degradation when using cyankiwi/GLM-4.7-Flash-AWQ-4bit (and potentially other AWQ quants of this model).
See (this post on NVIDIA forums)[https://forums.developer.nvidia.com/t/make-glm-4-7-flash-go-brrrrr/359111] for implementation details.

Build the standard image first:

```bash
# Standard image tag defaults to vllm-node
./build-and-copy.sh -c
```

Then, to run on a single node:

```bash
./launch-cluster.sh -t vllm-node --solo \
  --apply-mod mods/fix-glm-4.7-flash-AWQ \
  exec vllm serve cyankiwi/GLM-4.7-Flash-AWQ-4bit \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --served-model-name glm-4.7-flash \
  --max-model-len 202752 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --host 0.0.0.0 --port 8888 \
  --gpu-memory-utilization 0.7
```

To run on cluster:

```bash
./launch-cluster.sh -t vllm-node \
  --apply-mod mods/fix-glm-4.7-flash-AWQ \
  exec vllm serve cyankiwi/GLM-4.7-Flash-AWQ-4bit \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --served-model-name glm-4.7-flash \
  --max-model-len 202752 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --host 0.0.0.0 --port 8888 \
  --gpu-memory-utilization 0.7 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2
```

**NOTE**: vLLM implementation is suboptimal even with the patch. The model performance is still significantly slower than it should be for the model with this number of active parameters. Running in the cluster increases prompt processing performance, but not token generation. You can expect ~40 t/s generation speed in both single node and cluster.

#### Experimental Optimized MXFP4 Build

Added an experimental build option, optimized for DGX Spark and gpt-oss models by [Christopher Owen](https://github.com/christopherowen/spark-vllm-mxfp4-docker/blob/main/Dockerfile).

It is currently the fastest way to run GPT-OSS on DGX Spark, achieving 60 t/s on a single Spark.

To use this build, first build the container with `--exp-mxfp4` flag. I recommend using a separate label as it is currently not recommended to use this build for models other than gpt-oss:

```bash
# Image tag defaults to vllm-node-mxfp4 when --exp-mxfp4 is used
./build-and-copy.sh --exp-mxfp4 -c
```

Then, to run on a single Spark:

```bash
 docker run \
  --privileged \
  --gpus all \
  -it --rm \
  --network host --ipc=host \
  -v  ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node-mxfp4 \
  bash -c -i "vllm serve openai/gpt-oss-120b \
        --host 0.0.0.0 \
        --port 8888 \
        --enable-auto-tool-choice \
        --tool-call-parser openai \
        --reasoning-parser openai_gptoss \
        --gpu-memory-utilization 0.70 \
        --enable-prefix-caching \
        --load-format fastsafetensors \
        --quantization mxfp4 \
        --mxfp4-backend CUTLASS \
        --mxfp4-layers moe,qkv,o,lm_head \
        --attention-backend FLASHINFER \
        --kv-cache-dtype fp8 \
        --max-num-batched-tokens 8192"
```

On a Dual Spark cluster:

```bash
./launch-cluster.sh -t vllm-node-mxfp4 exec vllm serve \
  openai/gpt-oss-120b \
        --host 0.0.0.0 \
        --port 8888 \
        --enable-auto-tool-choice \
        --tool-call-parser openai \
        --reasoning-parser openai_gptoss \
        --gpu-memory-utilization 0.70 \
        --enable-prefix-caching \
        --load-format fastsafetensors \
        --quantization mxfp4 \
        --mxfp4-backend CUTLASS \
        --mxfp4-layers moe,qkv,o,lm_head \
        --attention-backend FLASHINFER \
        --kv-cache-dtype fp8 \
        --max-num-batched-tokens 8192 \
        --distributed-executor-backend ray \
        --tensor-parallel-size 2
```

### 2025-12-24

- Added `hf-download.sh` script to download models from HuggingFace using `uvx` and optionally copy them to other cluster nodes.

Example usage. This will download model and distribute in parallel across all nodes in the cluster:

```bash
./hf-download.sh QuantTrio/GLM-4.7-AWQ -c --copy-parallel
```

### 2025-12-23

- Added mods/patches functionality allowing custom patches to be applied via `--apply-mod` flag in `launch-cluster.sh`, enabling model-specific compatibility fixes and experimental features without rebuilding the entire image.

- Added support for [Salyut1/GLM-4.7-NVFP4](https://huggingface.co/Salyut1/GLM-4.7-NVFP4) quant.

To run, use the new `--apply-mod` flag to apply a patch that fixes incompatibility due to glm4 parser expecting separate k and v scales, while this model uses fused quantization scheme. See [this issue on Huggingface](https://huggingface.co/Salyut1/GLM-4.7-NVFP4/discussions/3#694ab9b6e2efa04b7ecb0c4b) for details.

After downloading the model on both nodes (to avoid excessive wait times during launch), use this command:

```bash
./launch-cluster.sh --apply-mod ./mods/fix-Salyut1-GLM-4.7-NVFP4 \
exec vllm serve Salyut1/GLM-4.7-NVFP4 \
        --attention-config.backend flashinfer \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --enable-auto-tool-choice \
        -tp 2 \
        --gpu-memory-utilization 0.88 \
        --max-model-len 32000 \
        --distributed-executor-backend ray \
        --host 0.0.0.0 \
        --port 8000
```

### 2025-12-21

- Added `--pre-tf` / `--pre-transformers` flag to `build-and-copy.sh` for early Transformers 5 testing. Historical note: this flag is now deprecated; current builds use the vLLM default Transformers dependency and the flag only preserves legacy inputs/tagging.
- Pre-built wheels now support release versions. Use with `--use-wheels release`.
- Using nightly wheels or building from source is recommended for better performance.

### 2025-12-20

- Limited ccache to 50G when building from source to reduce build cache size.
- Added `--pre-flashinfer` flag to `build-and-copy.sh` to use pre-release versions of FlashInfer.
- Added `--use-wheels [mode]` flag to `build-and-copy.sh`.
  - Allows building the container using pre-built vLLM wheels instead of compiling from source.
  - Reduced build time and container size.
  - `mode` is optional and defaults to `nightly`.
  - Supported modes: `nightly` (release wheels are broken with CUDA 13 currently). UPDATE: `release` also works now.
### 2025-12-19

Updated `build-and-copy.sh` to support copying to multiple hosts (thanks @ericlewis for the contribution).
- Added `-c, --copy-to` (accepts space- or comma-separated host lists) and kept `--copy-to-host` as a backward-compatible alias.
- Added `--copy-parallel` to copy to all hosts concurrently.
- Added autodiscovery support: if no hosts are provided to `--copy-to`, the script detects other cluster nodes automatically.
- **BREAKING CHANGE**: Short `-h` argument is now used for help. Use `-c` for copy.

### 2025-12-18

- Added `launch-cluster.sh` convenience script for basic cluster management - see details below.
- Added `-j` / `--build-jobs` argument to `build-and-copy.sh` to control build parallelism.
- Added `--nccl-debug` option to specify NCCL debug level. Default is none to decrease verbosity.

### 2025-12-15

Updated `build-and-copy.sh` flags:
- Renamed `--triton-sha` to `--triton-ref` to support branches and tags in addition to commit SHAs.
- Added `--vllm-ref <ref>`: Specify vLLM commit SHA, branch or tag (defaults to `main`).

### 2025-12-14

Converted to multi-stage Docker build with improved build times and reduced final image size. The builder stage is now separate from the runtime stage, excluding unnecessary build tools from the final image.

Added timing statistics to `build-and-copy.sh` to track Docker build and image copy durations, displaying a summary at the end.

Triton is now being built from the source, alongside with its companion triton_kernels package. The Triton version is set to v3.5.1 by default, but it can be changed by using `--triton-sha` parameter.

Added new flags to `build-and-copy.sh`:
- `--triton-sha <sha>`: Specify Triton commit SHA (defaults to v3.5.1 currently)
- `--no-build`: Skip building and only copy existing image (requires `--copy-to`)

### 2025-12-11 update

PR for MiniMax-M2 has been merged into main, so removed the temporary patch from Dockerfile.

### 2025-12-11

Applied a patch to fix broken MiniMax-M2 in some quants after [this commit](https://github.com/vllm-project/vllm/commit/d017bceb08eaac7bae2c499124ece737fb4fb22b) until [this PR](https://github.com/vllm-project/vllm/pull/30389) is approved. 
See [this issue](https://github.com/vllm-project/vllm/issues/30445) for details.

### 2025-12-05

Added `build-and-copy.sh` for convenience.

### 2025-11-26

Initial release.
Updated RoCE configuration example to include both interfaces in the list.
Applied patch to enable FastSafeTensors in cluster configuration (EXPERIMENTAL) and added documentation on fastsafetensors use.

## 1\. Building the Docker Image

### Building Manually

Building the container manually is no longer supported due to Dockerfile complexity. Please use the provided build script.

### Using the Build Script

The `build-and-copy.sh` script prepares the runner image and optionally copies it to one or more nodes. By default it pulls `eugr/spark-vllm:latest` and tags it locally. Use `--use-wheels` to build only the runner from downloaded or local precompiled wheels. Source compilation happens only when an explicit source-build flag such as `--rebuild-vllm`, `--vllm-ref`, `--apply-vllm-pr`, `--rebuild-flashinfer`, `--flashinfer-ref`, or `--apply-flashinfer-pr` is supplied.

**Basic usage (prepare local image):**

```bash
./build-and-copy.sh
```

**Prepare with a custom local tag:**

```bash
./build-and-copy.sh -t my-vllm-node
```

**Build runner image from wheels:**

```bash
./build-and-copy.sh --use-wheels
```

**Prepare and copy to Spark node(s) using autodiscovery (recommended):**

If you omit the host list after `--copy-to`, the script discovers the other
nodes in the cluster and excludes the current node.

```bash
./build-and-copy.sh -c --copy-parallel
```

`-c` / `--copy-to` requests distribution. With no addresses, the script uses
`COPY_HOSTS` from `.env` or autodiscovery; `--copy-parallel` transfers to all
resolved hosts concurrently.

**Manual host fallback:**

Pass addresses only when you were specifically instructed to do so or the
network topology prevents autodiscovery from working.

Using the same username as currently logged-in user (single host):

```bash
./build-and-copy.sh --copy-to 192.168.177.12
```

Copy to multiple hosts (space- or comma-separated after the flag):

```bash
./build-and-copy.sh --copy-to 192.168.177.12 192.168.177.13
```

Copy to multiple hosts in parallel:

```bash
./build-and-copy.sh --copy-to 192.168.177.12 192.168.177.13 --copy-parallel
```

Using a different username:

```bash
./build-and-copy.sh --copy-to 192.168.177.12 --user your_username
```

**Force rebuild vLLM from source:**

```bash
./build-and-copy.sh --rebuild-vllm
```

**Force rebuild FlashInfer from source (skips prebuilt wheel download):**

```bash
./build-and-copy.sh --rebuild-flashinfer
```

**Combined example (rebuild vLLM and copy to another node):**

```bash
./build-and-copy.sh --rebuild-vllm -c
```

**Build for specific GPU architecture:**

```bash
./build-and-copy.sh --gpu-arch 12.0f
```

**Build a vLLM fork with custom PyTorch versions:**

```bash
./build-and-copy.sh \
  --vllm-repo https://github.com/local-inference-lab/vllm.git \
  --vllm-ref dev/fathomless-firmament \
  --torch-version 2.12.0 \
  --torchvision-version 0.27.0 \
  --torchaudio-version none
```

For the maintained experimental B12X combination, the equivalent shortcut is:

```bash
./build-and-copy.sh --exp-b12x
```

Without local-build flags, this pulls `eugr/spark-vllm-b12x:latest` and tags it
as `vllm-node-b12x` unless `-t` is supplied. To build the maintained combination
from `local-inference-lab/vllm@dev/infernal-invocation` and the `master` branch of the
B12X repository, run:

```bash
./build-and-copy.sh --exp-b12x --rebuild-vllm
```

It can be combined with `--apply-vllm-pr <pr-num>` to build custom vLLM patches.
The preset preserves selected Blackwell subarchitectures in vLLM's CUDA 13
CMake configuration. This prevents targets such as `10.3a` and the default
`12.1a` from being reduced to generic `sm_100` or `sm_120`, which cannot compile
the B12X branch's NVFP4 cache writer. Explicit `--gpu-arch 12.0a` and
`--gpu-arch 12.0f` selections remain supported and are forwarded unchanged.
FlashInfer architecture validation applies to every standard-Dockerfile build,
including B12X: alternate targets rebuild FlashInfer when no matching
architecture marker is present, and the cached wheel records its architecture
so a later build cannot silently reuse a wheel for a different target.

Custom vLLM repositories are cloned fresh instead of using the shared upstream checkout cache. Specifying a custom repository forces a vLLM source build. Upstream preset PRs are skipped by default for custom repositories and refs.

Wheel profiles are selected automatically:

```text
.wheel-cache/
├── flashinfer/
│   ├── regular/   # shared by regular and B12X builds
│   └── custom/    # custom ref/PR or non-default GPU target
└── vllm/
    ├── regular/
    ├── b12x/
    └── custom/    # custom repository/ref/PR, Torch family, or GPU target
```

Only regular vLLM wheels are downloaded from the published wheel release.
`--exp-b12x` is therefore incompatible with `--use-wheels`: use bare
`--exp-b12x` for the published image or add `--rebuild-vllm` for a source build.

For any branch, tag, or commit selected from `local-inference-lab/vllm`, the runner freshly clones the `master` ref of `https://github.com/lukealonso/b12x.git`, builds and installs its `b12x` distribution automatically. A per-build cache key prevents Docker from reusing a stale source checkout. Before the `--no-deps` install, its package metadata is updated to the image-wide CUTLASS DSL 4.7.0 pin. The exact source commit is recorded at `/workspace/b12x-source-commit`. B12X requires PyTorch 2.12 or newer; the preset uses 2.13.0.

**Copy existing image without rebuilding:**

```bash
./build-and-copy.sh --no-build --copy-to
```

**Available options:**

| Flag | Description |
| :--- | :--- |
| `-t, --tag <tag>` | Local image tag (default: `vllm-node`; auto-set to `vllm-node-tf5` with `--tf5`, `vllm-node-mxfp4` with `--exp-mxfp4`, or `vllm-node-b12x` with `--exp-b12x`) |
| `--use-wheels` | Build only the runner image from downloaded or local precompiled wheels; never implicitly compile missing wheels. Incompatible with `--exp-b12x`. |
| `--gpu-arch <arch>` | Target GPU architecture for wheel/source builds. Non-default targets rebuild FlashInfer unless the local cache is marked for that architecture. The default `12.1a` still uses the prebuilt image unless another build-forcing flag is set. |
| `--rebuild-flashinfer` | Skip prebuilt wheel download; force a fresh local FlashInfer build |
| `--rebuild-vllm` | Force rebuild vLLM from source |
| `--force-flashinfer-download` | Force download FlashInfer wheels, skipping cached wheel checks |
| `--force-vllm-download` | Force download vLLM wheels, skipping cached wheel checks |
| `--force-download` | Force download all prebuilt wheels, skipping cached wheel checks |
| `--vllm-repo <url>` | vLLM Git repository. Defaults to `https://github.com/vllm-project/vllm.git`; custom repositories bypass the shared checkout cache and force a source build. |
| `--vllm-ref <ref>` | vLLM commit SHA, branch or tag (default: `main`) |
| `--torch-version <version>` | PyTorch version installed in source-build and runner stages (default: `2.13.0`) |
| `--torchvision-version <version>` | Optional torchvision version (default: `0.28.0`) |
| `--torchaudio-version <version>` | Optional torchaudio version (default: `2.11.0`; use `none` to omit it) |
| `--flashinfer-ref <ref>` | FlashInfer commit SHA, branch or tag (default: `main`) |
| `--apply-vllm-pr <pr-num>` | Apply a vLLM PR patch during the image build. Can be specified multiple times. This is distinct from the launch-time option accepted by `launch-cluster.sh` and `run-recipe.sh`. |
| `--apply-preset-vllm-prs` | Apply preset vLLM PRs even when `--vllm-repo`, `--vllm-ref`, or `--apply-vllm-pr` would otherwise suppress them |
| `--apply-flashinfer-pr <pr-num>` | Apply a FlashInfer PR patch during build. Can be specified multiple times. |
| `--tf5` | Deprecated compatibility flag; pulls/tags the prebuilt image as `vllm-node-tf5` unless another build-forcing flag is set. Aliases: `--pre-tf, --pre-transformers`. |
| `--exp-mxfp4` | Build with experimental native MXFP4 support. Alias: `--experimental-mxfp4`. |
| `--exp-b12x` | Select the B12X profile. Pulls `eugr/spark-vllm-b12x:latest` unless a local wheel/image build is requested; defaults to local tag `vllm-node-b12x`. Alias: `--experimental-b12x`. |
| `-c, --copy-to <hosts>` | Host(s) to copy the image to after preparation (space- or comma-separated). Hosts with the same image ID are skipped. |
| `--copy-to-host` | Alias for `--copy-to` (backwards compatibility). |
| `--copy-parallel` | Copy to all specified hosts concurrently. |
| `-j, --build-jobs <jobs>` | Number of parallel build jobs (default: 16) |
| `-u, --user <user>` | Username for SSH connection (default: current user) |
| `--full-log` | Enable full Docker build output (`--progress=plain`) |
| `--no-build` | Skip image preparation entirely, only copy an existing local image tag (requires `--copy-to`) |
| `--network <name>` | Docker network to use during build (e.g. `host`). |
| `--cleanup` | Remove cached wheel files and provenance markers from every `.wheel-cache` profile; this does not force a local build by itself. |
| `--config <file>` | Path to `.env` configuration file (default: `.env` in script directory) |
| `--setup` | Force autodiscovery and save configuration to `.env` (even if `.env` already exists) |
| `-h, --help` | Show help message |

**IMPORTANT**: When copying to another node manually, use the IP assigned to a ConnectX 7 interface (`enp1s0f*`), not the 10G/wireless interfaces. When using `-c` without addresses, autodiscovery selects the correct interface automatically — in mesh mode it uses the direct IB-attached interfaces (`enp1s0f0np0`, `enp1s0f1np1`) for maximum transfer speed.

### Copying the container to another Spark node (Manual Method)

Alternatively, you can manually copy the image directly to your second Spark node via ConnectX 7 interface by using the following command:

```bash
docker save vllm-node | ssh your_username@another_spark_hostname_or_ip "docker load"
```

**IMPORTANT**: make sure you use Spark IP assigned to it's ConnectX 7 interface (enp1s0f1np1) , and not 10G one (enP7s7)!

-----

## 2\. Launching the Cluster (Recommended)

The `launch-cluster.sh` script simplifies the process of starting the cluster nodes. It handles Docker parameters, network interface detection, and node configuration automatically. It loads the default `.env` when present and runs autodiscovery when the file is absent.

### Basic Usage

**Start idle cluster containers (auto-detects everything):**

```bash
./launch-cluster.sh start
```

This will:
1.  Auto-detect the active InfiniBand and Ethernet interfaces.
2.  Auto-detect the node IP.
3.  Verify that the selected Docker image has the same content-addressable image ID on the head and every worker.
4.  Launch idle containers on the head and worker nodes.
5.  Use native no-Ray multiprocessing by default, or start Ray when `--ray` is selected.

Assumptions and limitations:

- It assumes that you've already set up passwordless SSH access on all nodes. If not, follow NVIDIA's [Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks). I recommend setting up static IPs in the configuration instead of automatically assigning them every time, but this script should work with automatically assigned addresses too.
- By default, it assumes that the container image name is `vllm-node`. If it differs, you need to specify it with `-t <name>` parameter.
- Before launching a multi-node cluster, it compares `docker image inspect` IDs for the selected image on the head and every active worker. The launch is aborted if an image is missing or any ID differs.
- If both ConnectX **physical** ports are utilized, and both have IP addresses, it will use whatever interface it finds first. Use `--eth-if` to override.
- It will ignore IPs associated with the 2nd "clone" of the physical interface. For instance, the outermost port on Spark has two logical Ethernet interfaces: `enp1s0f1np1` and `enP2p1s0f1np1`. Only `enp1s0f1np1` will be used. To override, use `--eth-if` parameter.
- It assumes that the same physical interfaces are named the same on all nodes (IOW, enp1s0f1np1 refers to the same physical port on all nodes). If it's not the case, you will have to launch cluster nodes manually or modify the script.
- It clears the Docker image entrypoint by default so images that define an entrypoint, such as `vllm-openai`, can still start as idle cluster containers before commands are executed. Use `--keep-entrypoint` to keep the image entrypoint.
- In solo mode, `-p` / `--publish` can be used to publish ports in Docker format, for example `-p 8000:8000`. When port publishing is used, the launcher does not use host networking. Port publishing is not supported in cluster mode.
- It sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in each container by default to reduce allocator fragmentation on DGX Spark. Override it with `-e PYTORCH_CUDA_ALLOC_CONF=<value>` when needed.
- It mounts `~/.cache/huggingface`, `~/.cache/vllm`, `~/.cache/flashinfer`, `~/.triton`, and `~/.tilelang` by default. Use `--no-cache-dirs` to skip the vLLM/FlashInfer/Triton/TileLang cache mounts. Add other mounts with repeatable Docker-style `-v` / `--volume` options, e.g. `-v "$HOME/my-data:/data"`.


**Start in daemon mode (background):**

```bash
./launch-cluster.sh -d start
```

**Stop the container:**

```bash
./launch-cluster.sh stop
```

**Check status:**

```bash
./launch-cluster.sh status
```

**Execute a command inside the running container:**

```bash
./launch-cluster.sh exec vllm serve ...
```

### Running `vllm serve` Commands

Pass ordinary model, serving, and parallelism arguments through `exec`:

```bash
./launch-cluster.sh exec vllm serve MODEL_NAME \
  --host 0.0.0.0 \
  --port 8000 \
  -tp 2
```

For a cluster, the launcher uses `.env` or autodiscovery and derives the required
node count from `-tp` / `--tensor-parallel-size`, `-pp` /
`--pipeline-parallel-size`, and `-dp` / `--data-parallel-size`. Do not add
`--distributed-executor-backend`, `--nnodes`, `--node-rank`, `--master-addr`,
`--master-port`, or `--headless` to the vLLM command. Default no-Ray mode adds
the correct per-node arguments automatically; launcher option `--ray` starts Ray
and adds its executor backend. Use `--solo` for a single-node launch.

### Environment Variables and Volume Mappings

Both `launch-cluster.sh` and `run-recipe.sh` accept repeatable `-e VAR=VALUE`
options to pass environment variables into the container and repeatable
`-v LOCAL_PATH:CONTAINER_PATH` options to add Docker volume mappings:

```bash
./launch-cluster.sh \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/models:/models" \
  exec vllm serve /models/MODEL_NAME --port 8000 -tp 2

./run-recipe.sh RECIPE_NAME \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/models:/models" \
  --setup
```

Options for `launch-cluster.sh` must appear before `exec`. In cluster mode, each
volume is mapped on every launched node, so its local path must be suitable on
every host.

For gated Hugging Face models, export the token for host-side downloads and pass
it into the container with `-e`:

```bash
export HF_TOKEN="YOUR_TOKEN"
./run-recipe.sh RECIPE_NAME -e HF_TOKEN="$HF_TOKEN" --setup

# Direct launch
./launch-cluster.sh -e HF_TOKEN="$HF_TOKEN" exec vllm serve MODEL_NAME -tp 2
```

Do not print or share the expanded command because it contains the token.

### Recipe Overrides and Additional vLLM Arguments

`run-recipe.sh` directly supports `--port`, `--host`, `--tp` /
`--tensor-parallel`, `--gpu-mem` / `--gpu-memory-utilization`, and
`--max-model-len` overrides. Add any other vLLM arguments after the `--`
separator:

```bash
./run-recipe.sh RECIPE_NAME \
  --port 9000 \
  --gpu-mem 0.85 \
  -- --max-num-seqs 16 --enforce-eager
```

Extra arguments are appended to the generated vLLM command. When an option is
defined more than once, vLLM uses the last occurrence; the recipe runner warns
about detected duplicates of its built-in overrides.

### Auto-Detection

The script attempts to automatically detect:
*   **Ethernet Interface (`ETH_IF`):** Determined by the number of active CX7 interfaces:
    - **2 active** (standard): the `enp*` interface (no capital P) that has an IP address.
    - **4 active** (mesh topology): `enP7s7` (preferred) or `wlP9s9` (wireless, shown with a warning) — the cluster coordination interface is separate from the CX7 ports in this configuration.
*   **InfiniBand Interface (`IB_IF`):** All active RoCE devices. In mesh mode this is always `rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`.
*   **Cluster peers:** Discovered by scanning the `ETH_IF` subnet for hosts with SSH access **and** a GB10 GPU (`nvidia-smi --query-gpu=name` must return `NVIDIA GB10`).
*   **Copy hosts (`COPY_HOSTS`):** In standard mode, same as cluster peers. In mesh mode, scanned separately on `enp1s0f0np0` and `enp1s0f1np1` subnets so that image/model transfers use the direct InfiniBand path.

### Manual Overrides

Use manual overrides only when instructed or when the topology cannot be
discovered correctly:

```bash
./launch-cluster.sh --nodes "10.0.0.1,10.0.0.2" --eth-if enp1s0f1np1 --ib-if rocep1s0f1 -e MY_ENV=123
```

| Flag | Description |
| :--- | :--- |
| `-n, --nodes` | Manual fallback/override: comma-separated node IPs (head node first). |
| `-t` | Docker image name (default: `vllm-node`). |
| `--name` | Container name (default: `vllm_node`). |
| `--eth-if` | Ethernet interface name. |
| `--ib-if` | InfiniBand interface name. |
| `-e, --env` | Environment variable to pass to container (e.g. `-e VAR=val`). Can be used multiple times. |
| `-j` | Number of parallel jobs for build environment variables (optional). |
| `--apply-mod` | Apply mods/patches from specified directory. Can be used multiple times to apply multiple mods. |
| `--apply-vllm-pr <pr-num>` | Fetch and apply an upstream vLLM PR to the installed runtime package before launch. Runtime-only; repeatable and ordered with `--apply-mod`. |
| `--nccl-debug` | NCCL debug level (e.g., INFO, WARN). Defaults to INFO if flag is present but value is omitted. |
| `--check-config` | Check configuration and auto-detection without launching. |
| `--solo` | Solo mode: skip autodetection, launch only on current node, do not launch Ray cluster |
| `-p, --publish` | Publish a container port in Docker format, for example `-p 8000:8000`. Solo mode only; replaces host networking. Can be used multiple times. |
| `-v, --volume` | Map a volume in Docker format, for example `-v /local/path:/container/path`. Applies to every launched node and can be used multiple times. |
| `--ray` | Opt into Ray for multi-node vLLM and add `--distributed-executor-backend ray` when missing. |
| `--no-ray` | Default multi-node no-Ray mode; accepted for compatibility. |
| `--master-port` / `--head-port` | Port for cluster coordination: Ray head port or PyTorch distributed master port (default: 29501). |
| `--no-cache-dirs` | Do not mount default cache directories (~/.cache/vllm, ~/.cache/flashinfer, ~/.triton, ~/.tilelang). |
| `--keep-entrypoint` | Keep the Docker image entrypoint instead of clearing it before launching the idle cluster container. |
| `--earlyoom` | Run `earlyoom` as the container foreground process instead of `sleep infinity`. |
| `--earlyoom-args` | Arguments passed to `earlyoom` (default: `-M 524288,102400 -s 100 -r 60`). Implies `--earlyoom`. |
| `--launch-script` | Path to bash script to execute in the container (from examples/ directory or absolute path). If launch script is specified, action should be omitted. |
| `-d` | Run in daemon mode (detached). |
| `--non-privileged` | Run in non-privileged mode (removes `--privileged` and `--ipc=host`). |
| `--mem-limit-gb` | Memory limit in GB (default: 110, only with `--non-privileged`). |
| `--mem-swap-limit-gb` | Memory+swap limit in GB (default: mem-limit + 10, only with `--non-privileged`). |
| `--pids-limit` | Process limit (default: 4096, only with `--non-privileged`). |
| `--shm-size-gb` | Shared memory size in GB (default: 64, only with `--non-privileged`). |
| `--config <file>` | Path to `.env` configuration file (default: `.env` in script directory). |
| `--setup` | Force autodiscovery and save configuration to `.env` (even if `.env` already exists). |
| `start \| stop \| status \| exec` | Action to perform. Use `start` for idle containers or `exec` to run a command. Not compatible with `--launch-script`. |
| `command` | Command to execute inside the container (only for `exec` action). |

### Early OOM Monitor

The `--earlyoom` flag starts the idle container with `earlyoom` as PID 1 instead of `sleep infinity`, so it monitors memory while Ray and vLLM are launched with `docker exec`. This is optional; without `--earlyoom`, containers still use the plain idle command.

Default policy:

```bash
earlyoom -M 524288,102400 -s 100 -r 60
```

What those arguments mean:

- `-M 524288,102400`: send SIGTERM when available memory is below 524288 KiB (512 MiB), then SIGKILL when it is below 102400 KiB (100 MiB).
- `-s 100`: do not wait for swap to fill before acting. Earlyoom normally requires both memory and swap to be below their thresholds; `-s 100` makes the swap threshold effectively satisfied so host RAM pressure is enough. Use a paired value such as `-s 70,50` if you want swap-free thresholds for TERM and KILL.
- `-r 60`: print a memory report every 60 seconds. Use `-r 0` to disable periodic reports.

Tune the policy per launch with `--earlyoom-args`:

```bash
./launch-cluster.sh --earlyoom \
  --earlyoom-args "-M 1048576,262144 -s 70,50 -r 30" \
  exec vllm serve ...
```

The same options pass through recipe launches:

```bash
./run-recipe.sh minimax-m2-awq --solo \
  --earlyoom --earlyoom-args "-M 786432,196608 -s 100 -r 120"
```

You can also set the default arguments through the environment:

```bash
VLLM_SPARK_EARLYOOM_ARGS="-M 786432,196608 -s 100 -r 120" \
  ./launch-cluster.sh --earlyoom exec vllm serve ...
```

Other useful upstream options include `--prefer REGEX` and `--avoid REGEX` to bias victim selection, `--dryrun` to log what would be killed without killing it, and `-g` to kill the selected process group instead of only the selected process. Keep custom argument values shell-safe because they are passed through the launcher as command arguments.

`--earlyoom` is not compatible with `--keep-entrypoint`: the launcher must clear the image entrypoint so `earlyoom` can be the container foreground process.

### Non-Privileged Mode

The `--non-privileged` flag allows running containers without full privileged access while maintaining RDMA/InfiniBand functionality:

```bash
./launch-cluster.sh --non-privileged exec vllm serve ...
```

When `--non-privileged` is specified:
- `--privileged` is replaced with `--cap-add=IPC_LOCK`
- `--ipc=host` is replaced with `--shm-size=64g` (configurable via `--shm-size-gb`)
- RDMA devices are exposed via `--device=/dev/infiniband`
- Resource limits are applied: memory (110GB), memory+swap (120GB), pids (4096)

All launch modes set Docker's open-file ulimit to `1048576`, which avoids loader failures when highly sharded models are opened in parallel by Ray workers. Override it with `VLLM_SPARK_NOFILE_LIMIT` if your host requires a different value.

These resource limits can be customized:
```bash
./launch-cluster.sh --non-privileged \
  --mem-limit-gb 120 \
  --mem-swap-limit-gb 130 \
  --shm-size-gb 64 \
  exec vllm serve ...
```

## 3\. Running the Container (Manual)

Manual `docker run` can be useful if you want full control over Docker parameters, but it's not recommended even for single Sparks. For multi-node Ray or no-Ray launches, use `launch-cluster.sh`; the old standalone `run-cluster-node.sh` flow has been removed and its logic is now integrated into the launcher.

```bash
docker run -it --rm \
  --gpus all \
  --net=host \
  --ipc=host \
  --privileged \
  --ulimit nofile=1048576:1048576 \
  --name vllm_node \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node bash
```

Inside the container, run `vllm serve ...` directly for solo inference.

**IMPORTANT**: for cluster commands, use the IP addresses associated with ConnectX 7 interfaces, not the 10G or wireless interfaces.


**Flags Explained:**

  * `--net=host`: **Required for cluster commands.** Ray and NCCL need full access to host network interfaces.
  * `--ipc=host`: **Recommended.** Allows shared memory access for PyTorch/NCCL. As an alternative, you can set it via `--shm-size=16g`.
  * `--privileged`: **Recommended for InfiniBand.** Grants the container access to RDMA devices (`/dev/infiniband`). As an alternative, you can pass `--ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/infiniband`.
  * `--ulimit nofile=1048576:1048576`: **Recommended for large sharded models.** Prevents `Too many open files` errors during parallel safetensors metadata loading.

-----

## 4\. Configuration Details

### Cluster Configuration (`.env` file)

The scripts share a `.env` file (default: `.env` in the repo directory) for persistent cluster configuration. It is created automatically by autodiscovery — run `--discover` (via `run-recipe.sh`) or `--setup` (via `launch-cluster.sh` / `build-and-copy.sh`) on first use.

Autodiscovery and the saved `.env` are the default user workflow. Do not pass
node IPs manually unless you were specifically instructed to do so or the
network topology prevents autodiscovery from working.

**Supported variables:**

| Variable | Description |
| :--- | :--- |
| `CLUSTER_NODES` | Comma-separated node IPs used for Ray/vLLM cluster (head node first). |
| `COPY_HOSTS` | Comma-separated node IPs used for image and model distribution. In mesh mode these are the IPs on the direct IB-attached interfaces, which may differ from `CLUSTER_NODES`. |
| `LOCAL_IP` | IP address of the local node. |
| `ETH_IF` | Ethernet interface for cluster coordination (e.g. `enp1s0f1np1` or `enP7s7`). |
| `IB_IF` | Comma-separated RoCE/IB device names (e.g. `rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`). |
| `CONTAINER_*` | Any variable prefixed with `CONTAINER_` (except `CONTAINER_NAME`) is passed as `-e VAR=VALUE` to the container. Example: `CONTAINER_NCCL_DEBUG=INFO` → `-e NCCL_DEBUG=INFO`. |

**Mesh-mode NCCL variables** (written automatically when mesh topology is detected):

```
CONTAINER_NCCL_NET_PLUGIN=none
CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING=1
CONTAINER_NCCL_IB_MERGE_NICS=0
```

**Example `.env` for a standard 2-node cluster:**

```
CLUSTER_NODES=192.168.177.11,192.168.177.12
COPY_HOSTS=192.168.177.12
LOCAL_IP=192.168.177.11
ETH_IF=enp1s0f1np1
IB_IF=rocep1s0f1,roceP2p1s0f1
```

To use a custom config file path, pass `--config /path/to/file.env` to any script.

### Autodiscovery Workflow

On first run, if no `.env` is present, the scripts will automatically trigger autodiscovery. You can also run it explicitly:

```bash
# Via run-recipe.sh
./run-recipe.sh --discover

# Via launch-cluster.sh or build-and-copy.sh (force re-run even if .env exists)
./launch-cluster.sh --setup exec vllm serve ...
./build-and-copy.sh --setup -c
```

Autodiscovery:
1. Detects active CX7 interfaces and determines mesh vs. standard topology.
2. Scans the network for SSH-reachable GB10 peers.
3. In mesh mode, separately discovers `COPY_HOSTS` on direct IB-attached interfaces.
4. Prompts for per-node confirmation for both `CLUSTER_NODES` and `COPY_HOSTS`.
5. Saves the result to `.env`.

### Environment Persistence

The launcher injects node-specific environment variables with Docker `-e` flags when each container starts. If you need to open a second terminal into the running container for debugging, run:

```bash
docker exec -it vllm_node bash
```

The new shell inherits the container environment, including NCCL, Ray, and vLLM network settings.

## 5\. Mods and Patches

The vLLM Docker setup supports applying custom mods and patches to address specific model compatibility issues or apply experimental features. This functionality is primarily managed through the `--apply-mod` option in the cluster launch script, and `run-recipe.sh` can pass additional `--apply-mod` flags through to `launch-cluster.sh`.

### Available Mods

The repository includes several pre-configured mods in the `mods/` directory:

- **fix-Salyut1-GLM-4.7-NVFP4/**: Fixes the GLM4MoE parser for Salyut1/GLM-4.7-NVFP4 fused QKV quantization.
- **fix-glm-4.7-flash-AWQ/**: Applies GLM-4.7-Flash-AWQ compatibility and performance fixes.
- **fix-qwen3.5-chat-template/** and **fix-qwen3.6-chat-template/**: Install fixed chat templates used by the Qwen3.5 and Qwen3.6 recipes.
- **fix-qwen3.5-autoround/**, **fix-qwen3-next-autoround/**, and **fix-qwen35-tp4-marlin/**: Model-specific Qwen AutoRound and Marlin compatibility fixes.
- **fix-qwen3-coder-next/**: Qwen3-Coder-Next runtime and performance fixes.
- **radixark-dspark/**: Routes Qwen DSpark checkpoints such as `RadixArk/Qwen3.8-27B-DSpark` to vLLM's Qwen3 DSpark loader instead of the DeepSeek-V4 loader.
- **dspark-instanttensor/**: Filters embedded `mtp.*` DSpark draft weights before InstantTensor or safetensors I/O, preventing a second full-checkpoint load.
- **gpu-mem-util-gb/**: Adds experimental `--gpu-memory-utilization-gb` support.
- **kv-cache-prealloc-cleanup/**: Applies model-specific manual KV-cache startup tweaks: skip CUDA graph profiling when disabled by env and allow `--gpu-memory-utilization-gb` with `--kv-cache-memory-bytes`.
- **uma-fix/**: Uses CUDA/NVML memory accounting under WSL and skips host-memory UMA accounting there.
- **drop-caches/**: Periodically clears filesystem caches for large models running near the memory limit.
- **diffusiongemma/**: Adds DiffusionGemma support, dynamic causal attention compatibility, and Gemma4 reasoning/content-channel fixes used by the DiffusionGemma recipes.
- **nemotron-nano/** and **nemotron-super/**: Nemotron reasoning parser and model support helpers.
- **inkling-sm12-paged-kv/**: Routes Inkling's SM12 paged-KV relative attention through a vendored FA4 implementation while leaving other models and GPU architectures unchanged.
- **instanttensor-hybrid-draft-loader/**: Keeps a target model on InstantTensor while using lazy safetensors for eligible speculative draft weights, including embedded MTP drafts.
- **exp-b12x/**: Experimental FlashInfer b12x support for builds that include the required upstream vLLM support.
- **use-official-vllm/**: Installs `git`, `earlyoom`, InstantTensor, and SciPy inside official vLLM containers (Ubuntu/Debian-based) so that other mods can rely on `git apply`, the launcher can use `--earlyoom`, and vLLM can use `--load-format instanttensor` and SciPy-based functionality. The Python install preserves the image's existing Torch build. The mod also redirects the pip-installed NCCL library to the system `libnccl2` library to avoid DGX Spark multi-node NCCL hangs. Apply this mod first when using official vLLM images (e.g. `vllm-openai`).

Each mod directory typically contains:
- Patch files (`.patch`) for code modifications and/or other assets.
- `run.sh` script to apply the patch.

Patch can also be represented as a `.zip` file with the same structure.

### Using Mods

To apply mods when launching the cluster, use the `--apply-mod` flag:

```bash
./launch-cluster.sh --apply-mod ./mods/fix-Salyut1-GLM-4.7-NVFP4
```

You can apply multiple mods by specifying additional `--apply-mod` flags:

```bash
./launch-cluster.sh --apply-mod ./mods/fix-Salyut1-GLM-4.7-NVFP4 --apply-mod ./mods/other-mod
```

When using recipes, any mods listed in the recipe are applied first, followed by mods supplied on the command line:

```bash
./run-recipe.sh glm-4.7-flash-awq --solo --apply-mod ./mods/other-mod
```

### Applying an Upstream PR at Launch Time

For Python/package-only vLLM changes, `launch-cluster.sh` can create a temporary
mod from an upstream PR without rebuilding the image or adding a permanent
directory under `mods/`:

```bash
./launch-cluster.sh --solo \
  --apply-vllm-pr 12345 \
  exec vllm serve MODEL_NAME --port 8000 --host 0.0.0.0

./run-recipe.sh glm-4.7-flash-awq --solo \
  --apply-vllm-pr 12345
```

The PR diff is downloaded once on the head node, checksum-logged, and copied to
every container through the normal mod distribution path. Multiple PRs are
applied in the order specified. With direct launcher use, `--apply-mod` and
`--apply-vllm-pr` share command-line ordering. With recipes, recipe-declared
mods remain first, followed by command-line layers in their specified order.

The runtime path only applies files installed under `vllm/`; tests, docs, and
examples are ignored. A PR that changes CUDA/C++, build configuration,
dependencies, packaging, or other source-tree files is rejected. Apply such a
PR while building the image instead:

```bash
./build-and-copy.sh --apply-vllm-pr 12345
```

Runtime PR application requires `git` in the container. For an official vLLM
image that omits it, put `mods/use-official-vllm` before the PR layer:

```bash
./launch-cluster.sh --solo \
  --apply-mod mods/use-official-vllm \
  --apply-vllm-pr 12345 \
  exec vllm serve MODEL_NAME
```

The change is ephemeral and is reapplied whenever new containers are created.
The launcher refuses `--apply-vllm-pr` when containers with the selected name
are already running, because it cannot verify that those containers contain the
requested patch.

### Creating Custom Mods

To create your own mod:

1. Create a new directory in the `mods/` folder
2. Add your patch files (`.patch`) or other assets as necessary (optional).
3. Create a `run.sh` script to apply the patch. It shouldn't accept any parameters. This script is required.
4. Reference your mod using the `--apply-mod path/to/your/mod` flag

Mods can be used for:
- Applying specific model compatibility fixes
- Testing experimental features
- Customizing vLLM behavior for specific workloads
- Rapid iteration on development without rebuilding the entire image

## 6\. Launch Scripts

Launch scripts provide a simple way to define reusable model configurations. Instead of passing long command lines, you can create a bash script that is copied into the container and executed directly.

### Basic Usage

```bash
# Use a launch script by name (looks in examples/ directory)
./launch-cluster.sh --launch-script example-vllm-minimax

# Manual fallback when autodiscovery cannot support the topology
./launch-cluster.sh -n 192.168.1.1,192.168.1.2 --launch-script vllm-openai-gpt-oss-120b.sh

# Combine with mods for models requiring patches
./launch-cluster.sh --launch-script vllm-glm-4.7-nvfp4.sh --apply-mod mods/fix-Salyut1-GLM-4.7-NVFP4
```

### Script Format

Launch scripts are simple bash files that run directly inside the container:

```bash
#!/bin/bash
# PROFILE: OpenAI GPT-OSS 120B
# DESCRIPTION: vLLM serving openai/gpt-oss-120b with FlashInfer MOE optimization

# Set environment variables if needed
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

# Run your command
vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --enable-auto-tool-choice
```

### Available Launch Scripts

The `examples/` directory contains ready-to-use launch scripts:

- **example-vllm-minimax.sh** - MiniMax-M2-AWQ cluster launch example
- **vllm-openai-gpt-oss-120b.sh** - OpenAI GPT-OSS 120B with FlashInfer MOE
- **vllm-glm-4.7-nvfp4.sh** - GLM-4.7-NVFP4 (requires the glm4_moe patch mod)

See [examples/README.md](examples/README.md) for detailed documentation and more examples.

## 7\. Using cluster mode for inference

The preferred path is to let `launch-cluster.sh` start containers and run the command in one step:

```bash
./launch-cluster.sh exec vllm serve RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4 \
  --port 8888 --host 0.0.0.0 \
  --gpu-memory-utilization 0.7 \
  -tp 2 \
  --max-model-len 32768
```

No-Ray mode is the default. The launcher starts worker commands first, then runs
rank 0 on the head. To opt into Ray, add `--ray` before `exec`; do not add the
executor backend to the vLLM arguments:

```bash
./launch-cluster.sh --ray exec vllm serve RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4 \
  --port 8888 --host 0.0.0.0 \
  --gpu-memory-utilization 0.7 \
  -tp 2 \
  --max-model-len 32768
```

When parallelism flags are present, the launcher automatically trims the active node list or errors before startup if more nodes are required than configured.

## 8\. Model Loading

This build includes support for fastsafetensors and InstantTensor loading.

[fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors/) significantly improves loading speeds, especially on DGX Spark where MMAP performance is currently poor. It uses more efficient multi-threaded loading while avoiding mmap.

To use it, include `--load-format fastsafetensors` when running vLLM:

```bash
./launch-cluster.sh -e HF_HUB_OFFLINE=1 exec vllm serve openai/gpt-oss-120b --port 8888 --host 0.0.0.0 --trust_remote_code --swap-space 16 --gpu-memory-utilization 0.7 -tp 2 --load-format fastsafetensors
```

InstantTensor is available with `--load-format instanttensor`. Several large-model recipes use it to reduce load-time memory pressure.

## 9\. Benchmarking

I recommend using [llama-benchy](https://github.com/eugr/llama-benchy) - a new benchmarking tool that delivers results in the same format as llama-bench from llama.cpp suite.

## 10\. Downloading Models

The `hf-download.sh` script provides a convenient way to download models from HuggingFace and distribute them across your cluster nodes. It uses Huggingface CLI via `uvx` for fast downloads and `rsync` for distribution across the cluster.

### Prerequisites

- `uvx` must be installed (the script will prompt you to install it if missing).
- Passwordless SSH access to other nodes (if copying).

### Usage

**Download a model (local only):**

```bash
./hf-download.sh QuantTrio/MiniMax-M2-AWQ
```

**Download and copy using autodiscovery (recommended):**

```bash
./hf-download.sh -c QuantTrio/MiniMax-M2-AWQ
```

**Download and copy in parallel:**

```bash
./hf-download.sh -c --copy-parallel QuantTrio/MiniMax-M2-AWQ
```

**Use nodes from `.env` (respects `COPY_HOSTS`):**

```bash
./hf-download.sh -c QuantTrio/MiniMax-M2-AWQ
```

When `-c` is given without explicit hosts, the script checks `COPY_HOSTS` in `.env` first, then falls back to autodiscovery. In mesh mode this means transfers go over the direct IB-attached interfaces automatically.

If distribution fails with permission errors under `~/.cache`, verify ownership
on every node. When root-owned cache files are the cause and the cache should
belong to the login user, run this locally on each node:

```bash
sudo chown -R "$USER" "$HOME/.cache"
```

This is a privileged recursive change; confirm the target and ownership problem
before running it. If `HF_HOME` points elsewhere, repair that cache path instead.

**Manual host fallback:**

Pass specific nodes only when instructed or when autodiscovery cannot support
the topology:

```bash
./hf-download.sh -c 192.168.177.12,192.168.177.13 QuantTrio/MiniMax-M2-AWQ
```

**Use a custom config file:**

```bash
./hf-download.sh --config /path/to/cluster.env -c QuantTrio/MiniMax-M2-AWQ
```

**Available options:**

| Flag | Description |
| :--- | :--- |
| `<model-name>` | HuggingFace model ID (e.g. `QuantTrio/MiniMax-M2-AWQ`). Required. |
| `-c, --copy-to <hosts>` | Host(s) to copy the model to after download (space- or comma-separated). Omit hosts to use `COPY_HOSTS` from `.env` or autodiscovery. |
| `--copy-to-host` | Alias for `--copy-to` (backwards compatibility). |
| `--copy-parallel` | Copy to all hosts concurrently instead of serially. |
| `-u, --user <user>` | SSH username for remote copies (default: current user). |
| `--config <file>` | Path to `.env` configuration file (default: `.env` in script directory). |
| `-h, --help` | Show help message. |

### Hardware Architecture

**Note:** This project targets `12.1a` architecture (NVIDIA GB10 / DGX Spark). If you are using different hardware, you can use `--gpu-arch` flag in `./build-and-copy.sh`.
