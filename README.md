
# vLLM Docker Optimized for DGX Spark (single or multi-node)

This repository contains the Docker configuration and startup scripts to run a multi-node vLLM inference cluster using Ray. It supports InfiniBand/RDMA (NCCL) and custom environment configuration for high-performance setups.
Cluster setup supports direct connect between dual Sparks, connecting via QSFP/RoCE switch and 3-node mesh configuration.

While it was primarily developed to support multi-node inference, it works just as well on a single node setups.

## Table of Contents

- [DISCLAIMER](#disclaimer)
- [QUICK START](#quick-start)
- [CHANGELOG](#changelog)
- [1. Building the Docker Image](#1-building-the-docker-image)
- [2. Launching the Cluster (Recommended)](#2-launching-the-cluster-recommended)
- [3. Running the Container (Manual)](#3-running-the-container-manual)
- [4. Using `run-cluster-node.sh` (Internal)](#4-using-run-cluster-nodesh-internal)
- [5. Configuration Details](#5-configuration-details)
- [6. Mods and Patches](#6-mods-and-patches)
- [7. Launch Scripts](#7-launch-scripts)
- [8. Using cluster mode for inference](#8-using-cluster-mode-for-inference)
- [9. Fastsafetensors](#9-fastsafetensors)
- [10. Benchmarking](#10-benchmarking)
- [11. Downloading Models](#11-downloading-models)

## DISCLAIMER

This repository is not affiliated with NVIDIA or their subsidiaries. This is a community effort aimed to help DGX Spark users to set up and run the most recent versions of vLLM on Spark cluster or single nodes. 

Unless `--rebuild-vllm` or `--vllm-ref` or `--apply-vllm-pr` is specified, the builder will fetch the latest precompiled vLLM wheels from the repository. They are built nightly and tested on multiple models in both cluster and solo configuration before publishing.
We will expand the selection of models we test in the pipeline, but since vLLM is a rapidly developing platform, some things may break.

If you want to build the latest from main branch, you can specify `--rebuild-vllm` flag. Or you can target a specific vLLM release by setting `--vllm-ref` parameter.

Similarly, `--rebuild-flashinfer`, `--flashinfer-ref`, and `--apply-flashinfer-pr` control the FlashInfer build in the same way.

## QUICK START

### Build

Check out locally. If using DGX Spark cluster, do it on the head node.

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

Build the container.

**If you have only one DGX Spark:**

```bash
./build-and-copy.sh
```

**On DGX Spark cluster:**

Make sure you connect your Sparks together and enable passwordless SSH as described in our [Networking Guide](docs/NETWORKING.md). You can also check out NVidia's [Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks), but using our guide is the best way to get started.
**NEW**: the guide now includes instructions on setting up 3-node Spark mesh!

Then run the following command that will build and distribute image across the cluster.

```bash
./build-and-copy.sh -c
```

An initial build speed depends on your Internet connection speed and whether the base image is already present on your machine. After base image pull, the build should take only 2-3 minutes. If `--rebuild-vllm` and/or `--rebuild-flashinfer` is used to trigger a build from the sourcew, it will take between 20-40 minutes, but subsequent builds will be faster. Prebuilt FlashInfer and vLLM wheels are downloaded automatically from GitHub releases, so compilation from source is usually not required.

### Run

**On a single node**:

`launch-cluster.sh` supports solo mode, which is now a recommended way to run the container on a single Spark:

```bash
./launch-cluster.sh --solo exec \
  vllm serve \
    QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization 0.7 \
    --load-format fastsafetensors
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
  --gpu-memory-utilization 0.7 \
  -tp 2 \
  --distributed-executor-backend ray \
  --max-model-len 128000 \
  --load-format fastsafetensors \
  --enable-auto-tool-choice --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2_append_think
```

This will run the model on all available cluster nodes.

**NOTE:** do not use `--load-format fastsafetensors` if you are loading models that would take >0.85 of available RAM (without KV cache) as it may result in out of memory situation.

**Also:** You can use any vLLM container that has "bash" as its default entrypoint with the launch script. It was tested with NGC vLLM, but can work with others too. To use such container in the cluster, you need to specify `--apply-mod use-ngc-vllm` argument to `./launch-cluster.sh`. However, it's recommended to build the container using this repository for best compatibility and most up-to-date features. 

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

- `--tf5` / `--pre-tf` - tag defaults to `vllm-node-tf5`
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
By default it will try to allocate 112 GB for vLLM on each node. You can change this by changing `--gpu-memory-utilization` (e.g. `--gpu-memory-utilization 113`), but please be aware that it uses GB instead of percentage **for this recipe**. 

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
./run-recipe.sh qwen3.5-122b-int4-autoround --solo -e HF_TOKEN=$HF_TOKEN
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

`build-and-copy.sh` now automatically downloads prebuilt vLLM wheels from the [GitHub releases](https://github.com/eugr/spark-vllm-docker/releases/tag/prebuilt-vllm-current) before falling back to a local build — identical to the existing FlashInfer download mechanism. This eliminates the need to compile vLLM from source on first use.

The download logic mirrors the FlashInfer behaviour:
- If prebuilt wheels are available and newer than any locally cached version, they are downloaded automatically.
- If the download fails (e.g. no network, release not found, GPU arch not supported), the script falls back to building locally, or reuses existing local wheels if present.
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

`build-and-copy.sh` now automatically downloads prebuilt FlashInfer wheels from the [GitHub releases](https://github.com/eugr/spark-vllm-docker/releases/tag/prebuilt-flashinfer-current) before falling back to a local build. This eliminates the need to compile FlashInfer from source on first use, which typically takes around 20 minutes.

The download logic:
- If prebuilt wheels are available and newer than any locally cached version, they are downloaded automatically.
- If the download fails (e.g. no network, release not found, gpu arch is not compatible), the script falls back to building locally, or reuses existing local wheels if present.
- `--rebuild-flashinfer` skips the download entirely and forces a fresh local build.

No new flags are required - the download happens transparently unless `--rebuild-flashinfer` is specified.

All wheels (downloaded or built locally) are cached in the `./wheels` directory for subsequent reuse.

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

`./build-and-copy.sh` now supports ability to apply vLLM PRs to builds. PR is applied to the most recent vLLM commit (or specific vllm-ref if set). This does NOT apply to wheels build and MXFP4 special build!

To use, just specify `--apply-vllm-pr <pr_num>` in the arguments. Please note that it may fail depending on whether the PR needs a rebase for the specified vLLM reference/main branch. Use with caution!

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

To use the mod, first build the container with Transformers 5 support (`--pre-tf`) flag, e.g.:

```bash
# Image tag defaults to vllm-node-tf5 when --tf5/--pre-tf is used
./build-and-copy.sh --pre-tf -c
```

Then, to run on a single node:

```bash
./launch-cluster.sh -t vllm-node-tf5 --solo \
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
./launch-cluster.sh -t vllm-node-tf5 \
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

- Added `--pre-tf` / `--pre-transformers` flag to `build-and-copy.sh` to install pre-release transformers (5.0.0rc or higher). Use it if you need to run GLM 4.6V or any other model that requires transformers 5.0. It may cause issues with other models, so you may want to stick to the release version for everything else.
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

The `build-and-copy.sh` script automates the build process and optionally copies the image to one or more nodes. This is the officially supported method for building and deploying to multiple Spark nodes.

**Basic usage (build only):**

```bash
./build-and-copy.sh
```

**Build with a custom tag:**

```bash
./build-and-copy.sh -t my-vllm-node
```

**Build and copy to Spark node(s):**

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

**Build and copy using autodiscovery:**

If you omit the host list after `--copy-to`, the script will attempt to auto-discover other nodes in the cluster (excluding the current node) and copy the image to them.

```bash
./build-and-copy.sh --copy-to
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
./build-and-copy.sh --rebuild-vllm -c 192.168.177.12
```

**Build for specific GPU architecture:**

```bash
./build-and-copy.sh --gpu-arch 12.0f
```

**Copy existing image without rebuilding:**

```bash
./build-and-copy.sh --no-build --copy-to 192.168.177.12
```

**Available options:**

| Flag | Description |
| :--- | :--- |
| `-t, --tag <tag>` | Image tag (default: `vllm-node`; auto-set to `vllm-node-tf5` with `--tf5`, `vllm-node-mxfp4` with `--exp-mxfp4`) |
| `--gpu-arch <arch>` | Target GPU architecture (default: `12.1a`) |
| `--rebuild-flashinfer` | Skip prebuilt wheel download; force a fresh local FlashInfer build |
| `--rebuild-vllm` | Force rebuild vLLM from source |
| `--vllm-ref <ref>` | vLLM commit SHA, branch or tag (default: `main`) |
| `--flashinfer-ref <ref>` | FlashInfer commit SHA, branch or tag (default: `main`) |
| `--apply-vllm-pr <pr-num>` | Apply a vLLM PR patch during build. Can be specified multiple times. |
| `--apply-flashinfer-pr <pr-num>` | Apply a FlashInfer PR patch during build. Can be specified multiple times. |
| `--tf5` | Install transformers v5 (5.0.0 or higher). Aliases: `--pre-tf, --pre-transformers`. |
| `--exp-mxfp4` | Build with experimental native MXFP4 support. Alias: `--experimental-mxfp4`. |
| `-c, --copy-to <hosts>` | Host(s) to copy the image to after building (space- or comma-separated). |
| `--copy-to-host` | Alias for `--copy-to` (backwards compatibility). |
| `--copy-parallel` | Copy to all specified hosts concurrently. |
| `-j, --build-jobs <jobs>` | Number of parallel build jobs (default: 16) |
| `-u, --user <user>` | Username for SSH connection (default: current user) |
| `--full-log` | Enable full Docker build output (`--progress=plain`) |
| `--no-build` | Skip building, only copy existing image (requires `--copy-to`) |
| `--network <name>` | Docker network to use during build (e.g. `host`). |
| `--cleanup` | Remove all cached `.whl` and `*-commit` files from the `wheels/` directory. |
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

The `launch-cluster.sh` script simplifies the process of starting the cluster nodes. It handles Docker parameters, network interface detection, and node configuration automatically.

### Basic Usage

**Start the container (auto-detects everything):**

```bash
./launch-cluster.sh
```

This will:
1.  Auto-detect the active InfiniBand and Ethernet interfaces.
2.  Auto-detect the node IP.
3.  Launch the container in interactive mode.
4.  Start the Ray cluster node (head or worker depending on the IP).

Assumptions and limitations:

- It assumes that you've already set up passwordless SSH access on all nodes. If not, follow NVidia's [Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks). I recommend setting up static IPs in the configuration instead of automatically assigning them every time, but this script should work with automatically assigned addresses too.
- By default, it assumes that the container image name is `vllm-node`. If it differs, you need to specify it with `-t <name>` parameter.
- If both ConnectX **physical** ports are utilized, and both have IP addresses, it will use whatever interface it finds first. Use `--eth-if` to override.
- It will ignore IPs associated with the 2nd "clone" of the physical interface. For instance, the outermost port on Spark has two logical Ethernet interfaces: `enp1s0f1np1` and `enP2p1s0f1np1`. Only `enp1s0f1np1` will be used. To override, use `--eth-if` parameter.
- It assumes that the same physical interfaces are named the same on all nodes (IOW, enp1s0f1np1 refers to the same physical port on all nodes). If it's not the case, you will have to launch cluster nodes manually or modify the script.
- It will mount only `~/.cache/huggingface` to the container by default. If you want to mount other caches, you'll have to pass set `VLLM_SPARK_EXTRA_DOCKER_ARGS` environment variable, e.g.: `VLLM_SPARK_EXTRA_DOCKER_ARGS="-v $HOME/.cache/vllm:/root/.cache/vllm" ./launch-cluster.sh ...`. Please note that you must use `$HOME` instead of `~` here as the latter won't be expanded if passed through the variable to docker arguments.


**Start in daemon mode (background):**

```bash
./launch-cluster.sh -d
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

### Auto-Detection

The script attempts to automatically detect:
*   **Ethernet Interface (`ETH_IF`):** Determined by the number of active CX7 interfaces:
    - **2 active** (standard): the `enp*` interface (no capital P) that has an IP address.
    - **4 active** (mesh topology): `enP7s7` (preferred) or `wlP9s9` (wireless, shown with a warning) — the cluster coordination interface is separate from the CX7 ports in this configuration.
*   **InfiniBand Interface (`IB_IF`):** All active RoCE devices. In mesh mode this is always `rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1`.
*   **Cluster peers:** Discovered by scanning the `ETH_IF` subnet for hosts with SSH access **and** a GB10 GPU (`nvidia-smi --query-gpu=name` must return `NVIDIA GB10`).
*   **Copy hosts (`COPY_HOSTS`):** In standard mode, same as cluster peers. In mesh mode, scanned separately on `enp1s0f0np0` and `enp1s0f1np1` subnets so that image/model transfers use the direct InfiniBand path.

### Manual Overrides

You can override the auto-detected values if needed:

```bash
./launch-cluster.sh --nodes "10.0.0.1,10.0.0.2" --eth-if enp1s0f1np1 --ib-if rocep1s0f1 -e MY_ENV=123
```

| Flag | Description |
| :--- | :--- |
| `-n, --nodes` | Comma-separated list of node IPs (Head node first). |
| `-t` | Docker image name (default: `vllm-node`). |
| `--name` | Container name (default: `vllm_node`). |
| `--eth-if` | Ethernet interface name. |
| `--ib-if` | InfiniBand interface name. |
| `-e, --env` | Environment variable to pass to container (e.g. `-e VAR=val`). Can be used multiple times. |
| `-j` | Number of parallel jobs for build environment variables (optional). |
| `--apply-mod` | Apply mods/patches from specified directory. Can be used multiple times to apply multiple mods. |
| `--nccl-debug` | NCCL debug level (e.g., INFO, WARN). Defaults to INFO if flag is present but value is omitted. |
| `--check-config` | Check configuration and auto-detection without launching. |
| `--solo` | Solo mode: skip autodetection, launch only on current node, do not launch Ray cluster |
| `--no-ray` | No-Ray mode: run multi-node vLLM without Ray (uses PyTorch distributed backend). |
| `--master-port` / `--head-port` | Port for cluster coordination: Ray head port or PyTorch distributed master port (default: 29501). |
| `--no-cache-dirs` | Do not mount default cache directories (~/.cache/vllm, ~/.cache/flashinfer, ~/.triton). |
| `--launch-script` | Path to bash script to execute in the container (from examples/ directory or absolute path). If launch script is specified, action should be omitted. |
| `-d` | Run in daemon mode (detached). |
| `--non-privileged` | Run in non-privileged mode (removes `--privileged` and `--ipc=host`). |
| `--mem-limit-gb` | Memory limit in GB (default: 110, only with `--non-privileged`). |
| `--mem-swap-limit-gb` | Memory+swap limit in GB (default: mem-limit + 10, only with `--non-privileged`). |
| `--pids-limit` | Process limit (default: 4096, only with `--non-privileged`). |
| `--shm-size-gb` | Shared memory size in GB (default: 64, only with `--non-privileged`). |
| `--config <file>` | Path to `.env` configuration file (default: `.env` in script directory). |
| `--setup` | Force autodiscovery and save configuration to `.env` (even if `.env` already exists). |
| `start \| stop \| status \| exec` | Action to perform (default: `start`). Not compatible with `--launch-script`. |
| `command` | Command to execute inside the container (only for `exec` action). |

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

These resource limits can be customized:
```bash
./launch-cluster.sh --non-privileged \
  --mem-limit-gb 120 \
  --mem-swap-limit-gb 130 \
  --shm-size-gb 64 \
  exec vllm serve ...
```

## 3\. Running the Container (Manual)

Ray and NCCL require specific Docker flags to function correctly across multiple nodes (Shared memory, Network namespace, and Hardware access).

```bash
docker run -it --rm \
  --gpus all \
  --net=host \
  --ipc=host \
  --privileged \
  --name vllm_node \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node bash
```

Or if you want to start the cluster node (head or regular), you can launch with the run-cluster.sh script (see details below):

**On head node:**

```bash
docker run --privileged --gpus all -it --rm \
  --ipc=host \
  --network host \
  --name vllm_node \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node ./run-cluster-node.sh \
    --role head \
    --host-ip 192.168.177.11 \
    --eth-if enp1s0f1np1 \
    --ib-if rocep1s0f1,roceP2p1s0f1 
```

**On worker node**

```bash
docker run --privileged --gpus all -it --rm \
  --ipc=host \
  --network host \
  --name vllm_node \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node ./run-cluster-node.sh \
    --role node \
    --host-ip 192.168.177.12 \
    --eth-if enp1s0f1np1 \
    --ib-if rocep1s0f1,roceP2p1s0f1 \
    --head-ip 192.168.177.11
```

**IMPORTANT**: use the IP addresses associated with ConnectX 7 interface, not with 10G or wireless one!


**Flags Explained:**

  * `--net=host`: **Required.** Ray and NCCL need full access to host network interfaces.
  * `--ipc=host`: **Recommended.** Allows shared memory access for PyTorch/NCCL. As an alternative, you can set it via `--shm-size=16g`.
  * `--privileged`: **Recommended for InfiniBand.** Grants the container access to RDMA devices (`/dev/infiniband`). As an alternative, you can pass `--ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/infiniband`.

-----

## 4\. Using `run-cluster-node.sh` (Internal)

The script is used to configure the environment and launch Ray either in head or node mode.

Normally you would start it with the container like in the example above, but you can launch it inside the Docker session manually if needed (but make sure it's not already running).

### Syntax

```bash
./run-cluster-node.sh [OPTIONS]
```

| Flag | Long Flag | Description | Required? |
| :--- | :--- | :--- | :--- |
| `-r` | `--role` | Role of the machine: `head` or `node`. | **Yes** |
| `-h` | `--host-ip` | The IP address of **this** specific machine (for ConnectX port, e.g. `enp1s0f1np1`). | **Yes** |
| `-e` | `--eth-if` | ConnectX 7 Ethernet interface name (e.g., `enp1s0f1np1`). | **Yes** |
| `-i` | `--ib-if` | ConnectX 7 InfiniBand interface name (e.g., `rocep1s0f1` - on Spark specifically you want to use both "twins": `rocep1s0f1,roceP2p1s0f1`). | **Yes** |
| `-m` | `--head-ip` | The IP address of the **Head Node**. | Only if role is `node` |


**Hint**: to decide which interfaces to use, you can run `ibdev2netdev`. You will see an output like this:

```
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
```

Each physical port on Spark has two pairs of logical interfaces in Linux. 
Current NVIDIA guidance recommends using only one of them, in this case it would be `enp1s0f1np1` for Ethernet, but use **both** `rocep1s0f1,roceP2p1s0f1` for IB.

You need to make sure you allocate IP addresses to them (no need to allocate IP to their "twins").

### Example: Starting inside the Head Node

```bash
./run-cluster-node.sh \
  --role head \
  --host-ip 192.168.177.11 \
  --eth-if enp1s0f1np1 \
  --ib-if rocep1s0f1,roceP2p1s0f1
```

### Example: Starting inside a Worker Node

```bash
./run-cluster-node.sh \
  --role node \
  --host-ip 192.168.177.12 \
  --eth-if enp1s0f1np1 \
  --ib-if rocep1s0f1,roceP2p1s0f1 \
  --head-ip 192.168.177.11
```

-----

## 5\. Configuration Details

### Cluster Configuration (`.env` file)

The scripts share a `.env` file (default: `.env` in the repo directory) for persistent cluster configuration. It is created automatically by autodiscovery — run `--discover` (via `run-recipe.sh`) or `--setup` (via `launch-cluster.sh` / `build-and-copy.sh`) on first use.

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

The script automatically appends exported variables to `~/.bashrc`. If you need to open a second terminal into the running container for debugging, simply run:

```bash
docker exec -it vllm_node bash
```

All environment variables (NCCL, Ray, vLLM config) set by the startup script will be loaded automatically in this new session.

## 6\. Mods and Patches

The vLLM Docker setup supports applying custom mods and patches to address specific model compatibility issues or apply experimental features. This functionality is primarily managed through the `--apply-mod` option in the cluster launch script.

### Available Mods

The repository includes several pre-configured mods in the `mods/` directory:

- **fix-Salyut1-GLM-4.7-NVFP4/**: Contains patches glm4moe parser to work with fused QKV quantization scheme for Salyut1/GLM-4.7-NVFP4 quant of the newly released GLM 4.7 model.

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

## 7\. Launch Scripts

Launch scripts provide a simple way to define reusable model configurations. Instead of passing long command lines, you can create a bash script that is copied into the container and executed directly.

### Basic Usage

```bash
# Use a launch script by name (looks in profiles/ directory)
./launch-cluster.sh --launch-script example-vllm-minimax

# Use with explicit nodes
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
    --distributed-executor-backend ray \
    --enable-auto-tool-choice
```

### Available Launch Scripts

The `examples/` directory contains ready-to-use launch scripts:

- **example-vllm-minimax.sh** - MiniMax-M2-AWQ with Ray distributed backend
- **vllm-openai-gpt-oss-120b.sh** - OpenAI GPT-OSS 120B with FlashInfer MOE
- **vllm-glm-4.7-nvfp4.sh** - GLM-4.7-NVFP4 (requires the glm4_moe patch mod)

See [examples/README.md](examples/README.md) for detailed documentation and more examples.

## 8\. Using cluster mode for inference

First, start follow the instructions above to start the head container on your first Spark, and node container on the second Spark.
Then, on the first Spark, run vllm like this:

```bash
docker exec -it vllm_node bash -i -c "vllm serve RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4 --port 8888 --host 0.0.0.0 --gpu-memory-utilization 0.7 -tp 2 --distributed-executor-backend ray --max-model-len 32768"
```

Alternatively, run an interactive shell first:

```bash
docker exec -it vllm_node
```

And execute vllm command inside.

## 9\. Fastsafetensors

This build includes support for fastsafetensors loading which significantly improves loading speeds, especially on DGX Spark where MMAP performance is very poor currently.
[Fasttensors](https://github.com/foundation-model-stack/fastsafetensors/) solve this issue by using more efficient multi-threaded loading while avoiding mmap.

This build also implements an EXPERIMENTAL patch to allow use of fastsafetensors in a cluster configuration (it won't work without it!).
Please refer to [this issue](https://github.com/foundation-model-stack/fastsafetensors/issues/36) for the details.

To use this method, simply include `--load-format fastsafetensors` when running VLLM, for example:

```bash
HF_HUB_OFFLINE=1 vllm serve openai/gpt-oss-120b --port 8888 --host 0.0.0.0 --trust_remote_code --swap-space 16 --gpu-memory-utilization 0.7 -tp 2 --distributed-executor-backend ray --load-format fastsafetensors
```

## 10\. Benchmarking

I recommend using [llama-benchy](https://github.com/eugr/llama-benchy) - a new benchmarking tool that delivers results in the same format as llama-bench from llama.cpp suite.

## 11\. Downloading Models

The `hf-download.sh` script provides a convenient way to download models from HuggingFace and distribute them across your cluster nodes. It uses Huggingface CLI via `uvx` for fast downloads and `rsync` for distribution across the cluster.

### Prerequisites

- `uvx` must be installed (the script will prompt you to install it if missing).
- Passwordless SSH access to other nodes (if copying).

### Usage

**Download a model (local only):**

```bash
./hf-download.sh QuantTrio/MiniMax-M2-AWQ
```

**Download and copy to specific nodes:**

```bash
./hf-download.sh -c 192.168.177.12,192.168.177.13 QuantTrio/MiniMax-M2-AWQ
```

**Download and copy using autodiscovery:**

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
