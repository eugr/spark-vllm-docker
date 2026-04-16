# vLLM Docker Optimized for NVIDIA Jetson AGX Thor

This repository builds a **vLLM** Docker image tuned for **NVIDIA Jetson AGX Thor**
(T5000 / T4000) running **JetPack 7.x** (Jetson Linux r38.x, Ubuntu 24.04,
CUDA 13.0). Thor is a Blackwell-Jetson SoC with compute capability
**sm_110a**; its tensor cores natively accelerate **NVFP4**, **MXFP8**, and
**FP8 (E4M3 / E5M2)**.

The project was originally written for **NVIDIA DGX Spark** (GB10, sm_121a)
and was retargeted to Thor on the `claude/llm-optimization-jetson-thor`
branch. The Spark→Thor port covers:

| Aspect                  | DGX Spark                          | Jetson AGX Thor                  |
|-------------------------|------------------------------------|----------------------------------|
| GPU arch                | sm_121a (`12.1a`)                  | **sm_110a (`11.0a`)**            |
| Base CUDA image         | `nvidia/cuda:13.2.0-...`           | `nvidia/cuda:13.0.1-...` (sbsa)  |
| CPU                     | Grace Neoverse-V2 (20-core)        | Neoverse-V3AE (14-core)          |
| Interconnect            | ConnectX-7 200 GbE, RoCE mesh      | 4× 25 GbE MGBE (no ConnectX)     |
| NCCL fork               | `zyang-dev/nccl dgxspark-3node-ring` | **stock `libnccl2`**            |
| PyTorch wheels          | `whl/cu130` upstream               | `pypi.jetson-ai-lab.io/sbsa/cu130` (fallback to upstream) |
| Default image tag       | `vllm-node`                        | `vllm-thor`                      |
| Default BUILD_JOBS      | 16                                 | 8 (memory-budget-aware)          |
| Multi-node              | 2×, 3×-mesh, 4× clusters           | Solo preferred; multi-Thor over 25 GbE is supported but slow |

See `docs/NETWORKING.md` for the full networking-assumption diff.

> ⚠️ This image will **not** work on DGX Spark. Use
> [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) for Spark.

## Disclaimer

This repository is not affiliated with NVIDIA. It is a community effort to
help Jetson Thor owners run recent vLLM builds on-device. Until a public
prebuilt-wheel release for sm_110a exists, every build compiles vLLM and
FlashInfer from source on-device — expect 30–60 min for a first build,
much faster on subsequent rebuilds thanks to the shared ccache layer.

## Quick start

```bash
git clone https://github.com/facadedevil/thor-vllm-docker.git
cd thor-vllm-docker
./build-and-copy.sh                # builds the `vllm-thor` image
./launch-cluster.sh --solo exec \
    vllm serve QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ \
        --port 8000 --host 0.0.0.0 \
        --gpu-memory-utilization 0.75 \
        --load-format fastsafetensors
```

Or use the curated Thor recipe:

```bash
./run-recipe.sh qwen3-vl-30b-awq-thor --solo
```

### Requirements

* NVIDIA Jetson AGX Thor with **JetPack 7.0 or newer** flashed.
* NVIDIA Container Runtime configured as the default Docker runtime
  (`sudo nvidia-ctk runtime configure --runtime=docker --set-as-default`).
* At least 32 GB of free disk for build artifacts; more if you intend to
  rebuild vLLM/FlashInfer from source.
* Hugging Face token exported as `HF_TOKEN` (add it to `.env`).

## Table of Contents

- [Architecture at a glance](#architecture-at-a-glance)
- [Building the image](#building-the-image)
- [Running vLLM](#running-vllm)
- [LMCache (KV-cache reuse)](#lmcache-kv-cache-reuse)
- [NVFP4 and FP8 on Thor](#nvfp4-and-fp8-on-thor)
- [Multi-Thor clustering (advanced)](#multi-thor-clustering-advanced)
- [TensorRT Edge-LLM (alternative stack)](#tensorrt-edge-llm-alternative-stack)
- [Known issues on sm_110a](#known-issues-on-sm_110a)

## Architecture at a glance

Jetson Thor T5000 ships with:

* **Blackwell-Jetson GPU** — 2560 CUDA cores, 96 fifth-gen Tensor Cores
  with `tcgen05`/`tmem`, NVFP4 pipe. ~2070 FP4 TFLOPS sparse.
* **128 GB LPDDR5X** at ~273 GB/s — **unified** between CPU and GPU.
  Weight loading and KV-cache offload are near-zero-copy.
* **14× Arm Neoverse-V3AE** (Armv9.2-A), SBSA-compliant.
* **4× 25 GbE MGBE** on the devkit — no on-board ConnectX. RDMA requires
  an external SmartNIC over PCIe.
* 40 W – 130 W configurable TDP.

The Dockerfile pins:

```
TORCH_CUDA_ARCH_LIST=11.0a
FLASHINFER_CUDA_ARCH_LIST=11.0a
```

`11.0a` (with the trailing `a`) keeps architecture-specific features
(`tcgen05`, `tmem`) enabled. `ptxas` from CUDA 12.x does **not**
recognize `sm_110a` — the image strictly requires CUDA 13.

## Building the image

```bash
./build-and-copy.sh                       # default: tag=vllm-thor, arch=11.0a
./build-and-copy.sh --rebuild-vllm        # force source rebuild of vLLM
./build-and-copy.sh --rebuild-flashinfer  # force rebuild of FlashInfer
./build-and-copy.sh --vllm-ref v0.11.0    # build a specific vLLM tag/SHA
./build-and-copy.sh --apply-vllm-pr 12345 # apply an upstream PR
```

Prebuilt-wheel downloads are **disabled by default** on Thor (no public
sm_110a wheels exist yet). All builds compile from source against the
PyTorch wheel installed in the builder image.

### Cross-building for DGX Spark

The repo can still cross-build the original Spark image:

```bash
./build-and-copy.sh --gpu-arch 12.1a -t vllm-node
```

## Running vLLM

```bash
./launch-cluster.sh --solo exec \
    vllm serve <model-id> \
        --port 8000 --host 0.0.0.0 \
        --gpu-memory-utilization 0.75 \
        -tp 1
```

`--solo` skips peer discovery and runs on the current Thor only. The
launch script auto-mounts `~/.cache/huggingface` and exposes the standard
vLLM OpenAI-compatible server on `$PORT`.

Relevant environment variables (all set in `.env`):

* `HF_TOKEN` — HuggingFace access token for gated models.
* `CONTAINER_*` — anything prefixed becomes a `-e` flag in the container.
* `VLLM_THOR_EXTRA_DOCKER_ARGS` — extra `docker run` flags (use for
  volume mounts, extra env vars, etc.).
* `DOCKER_GPU_FLAG` — override the Jetson `--runtime=nvidia` default if
  you're running on a non-Jetson host.

## LMCache (KV-cache reuse)

[LMCache](https://github.com/LMCache/LMCache) is a KV-cache reuse layer
that offloads and shares KV blocks across requests, sessions, and
processes. On Thor it's particularly effective because LPDDR5X is
**unified** between CPU and GPU — "CPU offload" is effectively
zero-copy.

LMCache is installed in the runner image (set `--build-arg
INSTALL_LMCACHE=0` to skip). Enable it per-request by adding the
connector to your `vllm serve` line:

```bash
vllm serve <model> \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    ...
```

Tune via env vars (put them in `.env` prefixed with `CONTAINER_`, or
set `ENABLE_LMCACHE=1` for the defaults):

| Env var                      | Meaning                                   | Default |
|------------------------------|-------------------------------------------|---------|
| `LMCACHE_CHUNK_SIZE`         | KV tokens per chunk                       | 256     |
| `LMCACHE_LOCAL_CPU`          | Enable CPU DRAM backend                   | True    |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | CPU KV budget (GB)                        | 16      |

For NIXL/GPUDirect, Redis, or remote KV sharing backends, see the
[LMCache docs](https://docs.lmcache.ai/).

## NVFP4 and FP8 on Thor

Thor's fifth-gen Tensor Cores natively execute **NVFP4**, **MXFP8**,
**FP8-E4M3**, and **FP8-E5M2**. For maximum throughput on models
≥ 70 B, prefer NVFP4 (`nvidia/*-FP4` repos on HF) over AWQ or GGUF.

Known caveat: trtllm-gen FMHA cubins upstream are not compiled for
sm_110 yet — tracking
[flashinfer#2913](https://github.com/flashinfer-ai/flashinfer/pull/2913)
and
[TensorRT-LLM#11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799).
Pass `--apply-flashinfer-pr 2913` to the builder when the PR is not
yet merged; otherwise FlashInfer falls back to PTX-JIT at the first
invocation (small one-time cost).

## Multi-Thor clustering (advanced)

Stock Thor has 4× 25 GbE MGBE ports — far lower bandwidth than Spark's
200 GbE ConnectX-7 fabric. You can still run multi-node vLLM over plain
TCP; autodiscovery will pick the default-route interface and NCCL will
use `NCCL_SOCKET_IFNAME` + `NCCL_IB_DISABLE=1`. Expect noticeable
slowdown vs a single board.

```bash
./launch-cluster.sh exec vllm serve <big-model> \
    -tp 2 --distributed-executor-backend ray \
    --max-model-len 64000 ...
```

`CLUSTER_NODES`, `ETH_IF`, `LOCAL_IP` go in `.env`. Run `./build-and-copy.sh --setup`
once to auto-populate them via peer scanning.

If you attached a ConnectX SmartNIC to the Thor devkit over PCIe, set
`THOR_ETHERNET_ONLY=0` in the environment and the legacy RoCE-mesh
auto-detection path in `autodiscover.sh` will re-enable.

> **Not ported from Spark**: the `zyang-dev/nccl dgxspark-3node-ring`
> NCCL fork, ConnectX dual-port RoCE detection, and 3-node mesh
> recipes. Those files have been moved under `recipes/legacy-spark-*`
> for reference only.

## TensorRT Edge-LLM (alternative stack)

For NVFP4-optimized, C++ deployments where latency is critical, NVIDIA
ships [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)
(v0.6.0, Mar 2026) — a purpose-built edge serving runtime for JP7.1.
It doesn't provide an OpenAI-compatible server out of the box but has
better NVFP4 kernels than vLLM/FlashInfer on Thor today. The two
runtimes can coexist; pick per-model based on model-support matrix.

## Known issues on sm_110a

| Upstream issue                                                        | Status                      |
|-----------------------------------------------------------------------|-----------------------------|
| [vllm#26791](https://github.com/vllm-project/vllm/issues/26791) sm_110 arch not listed                   | Fixed by rebuilding from source with `TORCH_CUDA_ARCH_LIST=11.0a`       |
| [vllm#38411](https://github.com/vllm-project/vllm/issues/38411) `_vllm_fa2_C` compiled SM80-only          | Use FlashInfer backend (`VLLM_ATTENTION_BACKEND=FLASHINFER`)            |
| [vllm#27364](https://github.com/vllm-project/vllm/issues/27364) Qwen3-VL FP8 degenerate output            | Use NVFP4 or AWQ variant on Thor                                        |
| [vllm#32093](https://github.com/vllm-project/vllm/issues/32093) Nemotron Nano V3 FP16 bugs                | `mods/nemotron-nano` contains the workaround                            |
| [flashinfer#2913](https://github.com/flashinfer-ai/flashinfer/pull/2913) trtllm-gen cubins missing sm_110 | Apply the PR or rely on JIT fallback                                    |
| NIM `ptxas-blackwell` doesn't accept `sm_110a`                        | Only CUDA 13.x toolchain works; JP7.0+ ships it                         |

## Credits

Original DGX Spark implementation by [@eugr](https://github.com/eugr)
and contributors to `spark-vllm-docker`. Thor retargeting on the
`claude/llm-optimization-jetson-thor` branch.
