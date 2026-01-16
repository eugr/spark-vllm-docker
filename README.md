
# vLLM Docker Optimized for DGX Spark (single or multi-node)

This repository contains the Docker configuration and startup scripts to run a multi-node vLLM inference cluster using Ray. It supports InfiniBand/RDMA (NCCL) and custom environment configuration for high-performance setups.

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
- [7. Using cluster mode for inference](#7-using-cluster-mode-for-inference)
- [8. Fastsafetensors](#8-fastsafetensors)
- [9. Benchmarking](#9-benchmarking)
- [10. Downloading Models](#10-downloading-models)

## DISCLAIMER

This repository is not affiliated with NVIDIA or their subsidiaries. This is a community effort aimed to help DGX Spark users to set up and run the most recent versions of vLLM on Spark cluster or single nodes. 

The Dockerfile builds from the main branch of VLLM, so depending on when you run the build process, it may not be in fully functioning state. You can target a specific vLLM release by setting `--vllm-ref` parameter or use `--use-wheels release` to install pre-built release wheels.

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
./build-and-copy.sh --use-wheels
```

**On DGX Spark cluster:**

Make sure you connect your Sparks together and enable passwordless SSH as described in NVidia's [Connect Two Sparks Playbook](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks). 

Then run the following command that will build and distribute image across the cluster.

```bash
./build-and-copy.sh --use-wheels -c
```

### Run

**On a single node**:

```bash
 docker run \
  --privileged \
  --gpus all \
  -it --rm \
  --network host --ipc=host \
  -v  ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node \
  bash -c -i "vllm serve \
  QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
  --port 8000 --host 0.0.0.0 \
  --gpu-memory-utilization 0.7 \
  --load-format fastsafetensors"
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

**NOTE:** do not use `--load-format fastsafetensors` if you are loading models that would take >0.8 of available RAM (without KV cache) as it may result in out of memory situation.

## CHANGELOG

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

### 2026-01-15

- Added `--download-mod` flag to `launch-cluster.sh` to download GitHub PRs as mods.
- Added `--list-mods` flag to `launch-cluster.sh` to list available mods.
- Added `--revert-mod` flag to `launch-cluster.sh` to revert a previously applied mod.
- Updated `--apply-mod` to explicitly call the `apply` method in the mod's `run.sh` script.

Example: Downloading a PR as a mod

```bash
./launch-cluster.sh --download-mod https://github.com/vllm-project/vllm/pull/31386
```

Example: Reverting a mod

```bash
./launch-cluster.sh --revert-mod ./mods/fix-Salyut1-GLM-4.7-NVFP4
```

Example: Listing available mods

```bash
./launch-cluster.sh --list-mods
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

The Dockerfile includes specific **Build Arguments** to allow you to selectively rebuild layers (e.g., update the vLLM source code without re-downloading PyTorch).
Using a provided build script is recommended, but if you want to build using `docker build` command, here are the supported build arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `CACHEBUST_DEPS` | `1` | Change this to force a re-download of PyTorch, FlashInfer, and system dependencies. |
| `CACHEBUST_VLLM` | `1` | Change this to force a fresh git clone and rebuild of vLLM source code. |
| `TRITON_REF` | `v3.5.1` | Triton commit SHA, branch, or tag to build. |
| `VLLM_REF` | `main` | vLLM commit SHA, branch, or tag to build. |
| `BUILD_JOBS` | `16` | Number of parallel build jobs (default: 16). |
| `FLASHINFER_PRE` | `""` | Set to `--pre` to use pre-release versions of FlashInfer. |
| `PRE_TRANSFORMERS` | `0` | Set to `1` to install pre-release transformers (5.0.0rc or higher). |

### Building Manually using Wheels

If you prefer to use pre-built wheels (faster build, smaller image), you can use `Dockerfile.wheels`.

```bash
docker build -f Dockerfile.wheels -t vllm-node .
```

Supported build arguments for `Dockerfile.wheels`:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `BUILD_JOBS` | `16` | Number of parallel build jobs (default: 16). |
| `CACHEBUST_VLLM` | `1` | Change this to force a re-download of vLLM wheels. |
| `WHEELS_FROM_GITHUB_RELEASE` | `0` | Set to `1` to use GitHub release wheels instead of nightly wheels. |
| `FLASHINFER_PRE` | `""` | Set to `--pre` to use pre-release versions of FlashInfer. |
| `PRE_TRANSFORMERS` | `0` | Set to `1` to install pre-release transformers (5.0.0rc or higher). |

### Using the Build Script (Recommended)

The `build-and-copy.sh` script automates the build process and optionally copies the image to one or more nodes. This is the recommended method for building and deploying to multiple Spark nodes.

**Basic usage (build only):**

```bash
./build-and-copy.sh
```

**Build with a custom tag:**

```bash
./build-and-copy.sh --tag my-vllm-node
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

**Force rebuild vLLM source only:**

```bash
./build-and-copy.sh --rebuild-vllm
```

**Force rebuild all dependencies:**

```bash
./build-and-copy.sh --rebuild-deps
```

**Combined example (rebuild vLLM and copy to another node):**

```bash
./build-and-copy.sh --rebuild-vllm --copy-to 192.168.177.12
```

**Build with specific Triton commit:**

```bash
./build-and-copy.sh --triton-ref abc123def456
```

**Copy existing image without rebuilding:**

```bash
./build-and-copy.sh --no-build --copy-to 192.168.177.12
```

**Available options:**

| Flag | Description |
| :--- | :--- |
| `-t, --tag <tag>` | Image tag (default: 'vllm-node') |
| `--rebuild-deps` | Force rebuild all dependencies (sets CACHEBUST_DEPS) |
| `--rebuild-vllm` | Force rebuild vLLM source only (sets CACHEBUST_VLLM) |
| `--triton-ref <ref>` | Triton commit SHA, branch or tag (default: 'v3.5.1') |
| `--vllm-ref <ref>` | vLLM commit SHA, branch or tag (default: 'main') |
| `--pre-tf` | Install pre-release transformers (5.0.0rc or higher). Alias: `--pre-transformers`. |
| `--use-wheels [mode]` | Use pre-built vLLM wheels. Mode: `nightly` (default) or `release`. |
| `--pre-flashinfer` | Use pre-release versions of FlashInfer. |
| `-c, --copy-to <host[,host...] or host host...>` | Host(s) to copy the image to after building (space- or comma-separated list after the flag). |
| `--copy-to-host` | Alias for `--copy-to` (backwards compatibility). |
| `--copy-parallel` | Copy to all specified hosts concurrently. |
| `-j, --build-jobs <jobs>` | Number of parallel build jobs (default: Dockerfile default) |
| `-u, --user <user>` | Username for SSH connection (default: current user) |
| `--no-build` | Skip building, only copy existing image (requires `--copy-to`) |
| `-h, --help` | Show help message |

**IMPORTANT**: When copying to another node, make sure you use the Spark IP assigned to its ConnectX 7 interface (enp1s0f1np1), and not the 10G interface (enP7s7)!

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
*   **Ethernet Interface:** The interface associated with the active InfiniBand device that has an IP address.
*   **InfiniBand Interface:** The active InfiniBand devices. By default both active RoCE interfaces that correspond to active IB port(s) will be utilized.
*   **Node Role:** Based on the detected IP address and the list of nodes (defaults to `192.168.177.11` as head and `192.168.177.12` as worker).

### Manual Overrides

You can override the auto-detected values if needed:

```bash
./launch-cluster.sh --nodes "10.0.0.1,10.0.0.2" --eth-if enp1s0f1np1 --ib-if rocep1s0f1
```

| Flag | Description |
| :--- | :--- |
| `-n, --nodes` | Comma-separated list of node IPs (Head node first). |
| `-t` | Docker image name (default: `vllm-node`). |
| `--name` | Container name (default: `vllm_node`). |
| `--eth-if` | Ethernet interface name. |
| `--ib-if` | InfiniBand interface name. |
| `--apply-mod` | Apply mods/patches from specified directory. Can be used multiple times to apply multiple mods. |
| `--nccl-debug` | NCCL debug level (e.g., INFO, WARN). Defaults to INFO if flag is present but value is omitted. |
| `--check-config` | Check configuration and auto-detection without launching. |
| `-d` | Run in daemon mode (detached). |

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
3. Create a `run.sh` script to apply the patch. This script should handle `apply`, `revert` and `description` actions.
4. Reference your mod using the `--apply-mod path/to/your/mod` flag

### Downloading Mods from GitHub PRs

You can automatically download a pull request from the vLLM repository and convert it into a mod.

```bash
./launch-cluster.sh --download-mod https://github.com/vllm-project/vllm/pull/31386
```

This will:
1.  Download the PR patch.
2.  Create a mod directory (e.g., `mods/pr-31386`).
3.  Generate a `run.sh` script to apply/revert the patch.
4.  Fetch the PR title for the description.

You can then apply it:

```bash
./launch-cluster.sh --apply-mod pr-31386
```

Mods can be used for:
- Applying specific model compatibility fixes
- Testing experimental features
- Customizing vLLM behavior for specific workloads
- Rapid iteration on development without rebuilding the entire image

## 7\. Using cluster mode for inference

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

## 8\. Fastsafetensors

This build includes support for fastsafetensors loading which significantly improves loading speeds, especially on DGX Spark where MMAP performance is very poor currently.
[Fasttensors](https://github.com/foundation-model-stack/fastsafetensors/) solve this issue by using more efficient multi-threaded loading while avoiding mmap.

This build also implements an EXPERIMENTAL patch to allow use of fastsafetensors in a cluster configuration (it won't work without it!).
Please refer to [this issue](https://github.com/foundation-model-stack/fastsafetensors/issues/36) for the details.

To use this method, simply include `--load-format fastsafetensors` when running VLLM, for example:

```bash
HF_HUB_OFFLINE=1 vllm serve openai/gpt-oss-120b --port 8888 --host 0.0.0.0 --trust_remote_code --swap-space 16 --gpu-memory-utilization 0.7 -tp 2 --distributed-executor-backend ray --load-format fastsafetensors
```

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

### Hardware Architecture

**Note:** The Dockerfile defaults to `TORCH_CUDA_ARCH_LIST=12.1a` (NVIDIA GB10). If you are using different hardware, update the `ENV` variable in the Dockerfile before building.
