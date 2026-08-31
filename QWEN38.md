# Qwen3.8 Flash Next retain-12 on one DGX Spark

**Release candidate: bounded sequential-image qualification passed.** The
earlier image-memory failure was not reproduced after early native resizing,
one-image GPU batches, bounded admission, and pressure cleanup. All 52 HTTP
test cases behaved as expected (41 completed generations, 11 deliberate
rejections). The unchanged ~800K KV configuration still has narrow headroom:
the largest image request reached 2.09 GiB available. Keep earlyoom. Startup
driver-allocation warnings remain open. Subsequent C1 prefill checks reached
237,568 total prompt tokens; this is not a guarantee for arbitrary long-context
C16 workloads.

## Why this fork exists

The checkpoint keeps 12 accuracy-sensitive routed-expert layers in FP8 and uses
MXFP4 weights with dynamic MXFP8 activations for the remaining routed experts.
Single-Spark fit additionally requires NVMe-backed lookup of the approximately
51 GB PLE n-gram table. Host-memory offload alone does not solve this on a Spark:
CPU and GPU share the same physical memory.

Required additions are the mixed-precision loader, index-filtered fast loading,
NVMe PLE lookup and native Linux AIO helper, FP8 QSA KV support, packed MXFP4
draft-only head, GDN projection overlap, and bounded image admission. Target
attention weights and the target BF16 output head remain unchanged.

## Base-image compatibility

The requested base is:

```
docker.io/eugr/spark-vllm@sha256:1342a788154051e7d526d096b992ca88ae21d3da1ecc7326a9328172478da843
```

This ARM64 image was published August 26, 2026. Its layers total 11.48 GiB
compressed; installation needs additional space for unpacking and the derived
build. Published wheel metadata identifies vLLM source `7a9993878` and
FlashInfer 0.6.18, while the previous working preview runtime used vLLM
`0.1.dev20073+g8e685d198` and FlashInfer 0.6.17. Inspect installed versions after
pulling; wheel release metadata is not a complete installed-package inventory.

The published Spark runtime predates the August 31 Flash-Next model merge.
The merged model uses `vllm/models/qwen4_exp`, whereas these existing patches
target `vllm/models/qwen3_8_flash_next`. Async PLE integration is separate from
the model-support merge. Changing only the base-image name is insufficient;
neither bypassing version checks nor renaming paths establishes compatibility.

The compatibility route replaces only the vLLM distribution with the binary
distribution from the official Qwen preview image. Its Python files, native
extensions and entry point stay together; the Spark image's Torch, CUDA, NCCL,
RDMA and system libraries remain. This is **binary-artifact reproducibility**,
not a claim that the preview wheel was independently rebuilt from public source.

`fetch_preview_vllm.py` downloads only the 526,442,424-byte installation layer
from `vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`.
It verifies the immutable manifest, layer SHA256 and installed-file RECORD
hashes, and extracts only vLLM-owned paths. It does not pull the entire donor
image or copy its OS/dependencies. The old vLLM distribution is uninstalled
before replacement, preventing a mixture of files from two versions. This
adds approximately 526 MB of network transfer beyond the Spark base, plus our
explicit Python dependency adjustments and runtime patches.

- [Spark runtime source](https://github.com/eugr/spark-vllm-docker)
- [Flash-Next model integration](https://github.com/vllm-project/vllm/pull/53896)
- [Async PLE integration](https://github.com/vllm-project/vllm/pull/53899)

`vllm/vllm-openai`, the former base, is the vLLM project's OpenAI-compatible API
image, not an OpenAI-produced runtime. This port intentionally uses eugr's Spark
image instead and must requalify its native dependencies and model integration.

Installed candidate versions are Torch 2.13.0+cu130, Triton 3.8.0, FlashInfer
0.6.18 (Python/cubin/JIT-cache packages together), CUTLASS DSL 4.7.0 and TVM FFI
0.1.11. The preview vLLM metadata declares older FlashInfer 0.6.17 and CUTLASS
DSL 4.6.2. Keeping the Spark versions is an explicit compatibility deviation
being tested, not a claim that dependency metadata agrees without exceptions.

## Installation

Prerequisites: one otherwise available DGX Spark, NVIDIA-enabled Docker, local
NVMe, Python 3.10+ with PyYAML, `uv`/`uvx`, and adequate disk space. The pinned
checkpoint contains approximately 146.55 GB of files. Authenticate with Hugging
Face if the model is private; never include a token in an image or Git commit.

Clone the release branch and run on the Spark (the Hugging Face account must
have model access while the checkpoint remains private):

```bash
git clone --branch qwen38-single-spark-release https://github.com/marco-jeffrey/spark-vllm-docker.git
cd spark-vllm-docker
./run-recipe.sh qwen3.8-flash-next-mxfp4-fp8-r12 --setup --solo
```

Setup builds the derived image when absent, downloads the pinned revision into
the standard Hugging Face cache when absent, mounts the cache, and starts the
server. No source FP8 checkpoint, calibration corpus or research checkout is
required. Model bytes are not embedded in Docker.

For this recipe, setup checks the parsed config/index, every referenced shard
(including PLE), and required tokenizer/processor/draft assets before skipping
the download. Incomplete snapshots trigger the resumable downloader. This is a
presence check, not a full checksum verification of all weights. `HF_HOME` may
select another cache; conflicting `HF_HUB_CACHE` / `HUGGINGFACE_HUB_CACHE`
overrides are rejected because the launcher mounts the HF_HOME-based cache.

The model is `MJPansa/Qwen3.8-Flash-Next-MXFP4-FP8-R12`, pinned to
`92e1187ab292e2320edc5d9bccc477450b83e5ff`. On the host its expected location is:

```
~/.cache/huggingface/hub/models--MJPansa--Qwen3.8-Flash-Next-MXFP4-FP8-R12/
  blobs/
  snapshots/92e1187ab292e2320edc5d9bccc477450b83e5ff/
```

To test the derived build without reusing Docker build layers:

```bash
./build-and-copy.sh --qwen38-r12 --qwen38-no-cache --full-log
```

This retains downloaded base layers and model files. Use fresh isolated runtime
cache mounts for cold-start qualification; rebuilding Docker alone does not
clear JIT kernels. The Dockerfile-specific ignore file permits only the runtime
bundle into the build context.

## Selected configuration

| Setting | Candidate value |
|---|---|
| Parallelism | One Spark, TP1, up to 16 active sequences |
| Context / maximum output | 262,144 / 32,768 tokens |
| KV cache | FP8 E4M3, 13,774,094,336 bytes; 799,539 slots confirmed |
| MTP | 2 speculative tokens, packed draft-only head |
| Full decode graph capture sizes | 3, 6, 12, 18, 24, 30, 36, 42, 48 token positions |
| Intended sequence counts | 1, 2, 4, 6, 8, 10, 12, 14, 16 |
| Chunked prefill | 8,192 tokens; prefix caching enabled |
| Images | Up to 32 per complete request and 32,768 combined visual tokens |
| Per-image processing | Resized to approximately 2,048 visual tokens maximum |
| Source images | Inline PNG/JPEG/WebP; up to 16 MP; remote URLs disabled |
| Request size | At most 8 MiB decoded image-file bytes per image and 64 MiB JSON body |
| Video | Disabled |
| Multimodal processor cache | 0.125 GiB |
| GPU image batch | One image, split before device transfer |
| Aggregate body reservation | 128 MiB logical serialized-body budget |
| Emergency earlyoom thresholds | TERM at 2 GiB available, KILL at 1 GiB |
| Container limit | 116 GiB, no additional container swap allowance |

Inference is exposed only through `POST /v1/chat/completions`, including
text-only requests and tools. Health, models, version and metrics remain
available. Other inference endpoints, video/audio, file/remote image URLs and
client image-processor overrides return explicit errors. A caller must send
inline images and stay within **both** the image count and visual-token limits.

Image upload/validation is serialized. A separate image request gate remains
held through downstream completion, including streaming generation; response
headers do not release it. Additional image requests receive HTTP 429 with
`Retry-After: 1` rather than waiting with decoded images in memory. Text
generation bypasses this second gate. The upload/validation queue allows
16 waiters with a 120-second deadline, subject to a 128 MiB logical body
reservation budget. Unknown content lengths conservatively reserve that
whole budget. Neither that reservation nor these gates bound all transport,
Python, native-processor, or GPU memory.

After native EXIF orientation and white-background RGB conversion, source
images are resized on CPU with the pinned processor's uint8 transform before
loader results accumulate. Full-resolution decoded sources are closed. The
GPU then processes one image at a time, preserving final embedding order.
At image boundaries only, unused allocator blocks can be released when host
memory is below 8 GiB and estimated reclaimable storage exceeds 256 MiB.
This cleanup trigger is not a kill threshold and does not free live KV or
model tensors. The 2/1 GiB emergency thresholds remain unchanged.

These image-pipeline changes passed the bounded full-stack checks below. The
earlier failure observations refer to the previous batching/admission pipeline.

Graph sizes count scheduled token positions, not simply user requests. All
nine sizes were captured; the backend selects `FULL_DECODE_ONLY` from the
requested `FULL` mode. Prefill is not fully graph-captured. Emergency
thresholds are not a reserved memory pool and cannot guarantee against GPU
driver hangs. There is no 5 GiB test-client cancellation policy in this recipe.

## Qualification observations

### Final-recipe prefill at increasing context depths

The [README performance tables](README.md#single-spark-performance-observations)
include the earlier coding decode results separately from final-recipe prefill.
The decode table uses a different historical image/cache configuration and must
not be treated as a benchmark of this release image.

With the final image, C1 prefill of 32,768 new tokens after warmed prefixes of
32,768 / 131,072 / 204,800 tokens achieved 2,119 / 1,857 / 1,723 computed
input tokens/s. Final prompts were 65,536 / 163,840 / 237,568 tokens. These
rates include the 7,296 / 6,896 / 7,392 prefix tokens that cache boundaries
required recomputing. Native prefill times were 18.91 / 21.36 / 23.30 seconds;
client TTFT was 19.84 / 22.69 / 23.91 seconds. Warm-up is excluded.

All six requests (three warm-ups and three measured requests) completed with
no preemptions, OOMs or guard actions. Minimum sampled host available memory
was 4.98 GiB. Startup for this later run took 161.23 seconds; it was not a
fresh-install cold-cache measurement. Tests used deterministic synthetic
engineering text, thinking off, temperature zero and one output token.
The test server was subsequently shut down; no serving configuration changed.

### Current sequential-image pipeline

Tested on the rebuilt pinned Spark-base image with the selected configuration
unchanged except for image-pipeline controls and the smaller processor cache.
The full runtime passed 45 component tests. Startup to health readiness took
309.4 seconds, including 50.7 seconds for the target checkpoint loader; this
is not a universal first-install/startup estimate.

All generation cases below used a 64-token output cap with thinking enabled.
These are memory/lifecycle tests, not representative decode speed or image
quality benchmarks. TTFT includes CPU processing and LLM prefill, not just
vision encoding. The first image also triggered initialization/JIT.

| Request | TTFT | Minimum sampled host available memory |
|---|---:|---:|
| Text, 8,300 input tokens | 6.02 s | 4.68 GiB |
| Text, 32,877 input tokens | 14.70 s | 4.72 GiB |
| One 4096×4096 image, first use | 19.04 s | 3.80 GiB |
| Same one-image request again | 1.68 s | 5.48 GiB |
| Two new 4096×4096 images | 3.61 s | 4.39 GiB |
| Four new 4096×4096 images | 6.93 s | 3.52 GiB |
| Four-image repeat | 4.82 s | 3.99 GiB |
| Sixteen 4096×4096 images / 32,400 visual tokens | 25.05 s | 2.09 GiB |
| Thirty-two 256×256 images / 2,048 visual tokens | 2.03 s | 5.12 GiB |

Warm full-size GPU encoder intervals were **286–298 ms**, median **291 ms**
across 26 forwards. A 4096×4096 source is reduced to 1440×1440 / 2025 tokens.
The cold first encoder interval was 5.30 s; it includes JIT/launch idle time,
not 5.30 s of pure GPU compute. Before that first image, cleanup released
2.18 GiB of unused allocator reservation without releasing live tensors.

Thirty-three images, excessive visual tokens, and video all returned the
expected HTTP 400. Under four mixed submissions, two text and one image
request completed and one image received 429. Under sixteen mixed submissions,
eight text and one image request completed and seven images received 429.
Both later retry probes succeeded. Separately, all sixteen simultaneously
submitted text-only requests completed without rejection. This verifies
admission behavior, not sixteen concurrent image executions or GPU occupancy.

No engine failure, emergency termination, or cgroup OOM was observed during
these tests. Twenty-nine `NV_ERR_NO_MEMORY` driver warnings occurred during
startup, before health readiness; none were observed during this inference
matrix. Their cause remains unresolved. Half-second memory sampling may miss
shorter peaks. The 2.09 GiB minimum leaves only about 90 MiB above the TERM
threshold, so passing this bounded run does not justify removing the guard.

After testing, the qualification instance was intentionally stopped. Docker's
30-second stop grace period expired and the launcher returned137; the host
recovered its memory. That manual teardown was not an OOM, but graceful engine
signal forwarding through the upstream sleep/exec lifecycle is not qualified.

### Previous pipeline / historical failure

The first clean derived build took **32 seconds after the base download** on
our test Spark. That is not total first-install time: the 11.48 GiB compressed
base pull, model download and first-use CUDA compilation are separate. Triton
3.8.0 installed from an ARM64 wheel; it was not compiled from source.

The real packed draft head produced bit-exact outputs against the previous
runtime for 1, 3, 16 and 48 rows, including bit-exact CUDA-graph replay. This
checks that component, not whole-model accuracy. The documented recipe finds
the pinned checkpoint in the standard HF cache without redownloading it.

With empty runtime cache directories, cold health readiness took **348 seconds**.
Target weight reading took 56 seconds; total target/draft initialization took
174 seconds and reported 86.95 GiB. The remaining cold-start time includes JIT,
KV allocation and graph/warmup work. These durations overlap and must not be
added as independent totals. PLE skipped the resident table load. Nine decode
graphs captured in approximately 10 seconds. Warm restart was not tested.

Text, a 4096×4096 image, its repeat, two/three-image requests and a three-request
image batch completed. The square image became 2,025 visual tokens. A later
repeat crossed the **2 GiB** threshold: earlyoom sent SIGTERM to the engine,
the client received an error inside its SSE stream, and the launcher cleaned
up the container. The host recovered without a reboot. Initial HTTP 200 alone
did not mean inference succeeded.

| Phase | Minimum sampled host MemAvailable |
|---|---:|
| Cold startup | 3.05 GiB |
| Text request after readiness | 6.75 GiB |
| First full-sized image | 5.40 GiB |
| Two-image request, about 6.5K total input tokens | 3.73 GiB |
| Three-image request, about 8.5K total input tokens | 3.20 GiB |
| Three simultaneous submissions, each two/three images | 2.14 GiB |
| Subsequent repeat; earlyoom termination | 1.84 GiB |

This was a bounded lifecycle test with 64-token outputs, not a throughput or
image-quality benchmark. It does not establish C16 image stability, 32K visual
budget stability, tool-call correctness, or long-context safety. Larger tests
were not run after the failure. Startup also produced driver allocation
warnings. The precise source of retained/peak memory still needs attribution;
this is not sufficient evidence to call it a memory leak.

Keep the guard. The container's cgroup peak was only 30.68 GiB despite the much
larger host/GPU footprint, so its 116 GiB limit alone did not enforce total
unified-memory safety in this run. No cgroup OOM kill occurred: earlyoom's
SIGTERM was the intervention. The revised pipeline above subsequently passed
the bounded lifecycle tests without lowering the selected KV allocation or
emergency guard.

## Release gate

Before advertising this as broadly production-ready: resolve the startup
driver-warning question; qualify repeat startup, tools, prefix resume, long
outputs and worst-case long-context concurrent memory. The clean build,
snapshot lookup, PLE, MTP, bounded full-image-budget tests and C16 text/mixed
submissions have passed. Do not
publish throughput or safety claims from a different runtime as results of
this port. Short smoke requests reported zero KV-prefix hits; processor-cache
reuse is not proof of recurrent/KV prefix-cache reuse.

This fork adds the runtime, recipe integration, regression tests, this guide
and a README quick start with selected benchmark observations. Upstream
documentation and licenses remain intact. Private experiment reports, model
outputs, raw benchmark artifacts, credentials and research Git history are not
part of the release.

## Notices

This community work builds on eugr/spark-vllm-docker, vLLM and FlashInfer.
vLLM/FlashInfer use Apache-2.0; the vendored Triton compatibility frontend uses
MIT (see `docker/qwen38-r12/LICENSE.triton`). Model weights retain their separate
Qwen license and are downloaded independently. Preserve upstream notices.
