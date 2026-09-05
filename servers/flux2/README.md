# FLUX.2 image server — torchao NVFP4 on Blackwell

An OpenAI-compatible image server for [FLUX.2](https://huggingface.co/black-forest-labs)
built on 🤗 diffusers, with **on-the-fly NVFP4 quantization** for a large speedup
on NVIDIA Blackwell (sm_120a / sm_121a, e.g. DGX Spark GB10, RTX PRO 6000).

The transformer is quantized to NVFP4 (W4A4, Triton kernels) at load time via
[torchao](https://github.com/pytorch/ao) — no pre-quantized checkpoint, no
TensorRT. Based on the PyTorch blog
[*Faster Diffusion on Blackwell with MXFP8 and NVFP4*](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/).

## Measured on a DGX Spark GB10 (sm_121a), FLUX.2-dev, 28 steps @ 1024²

| | BF16 | NVFP4 (this) | speedup |
|---|---|---|---|
| text-to-image | ~2.3 min | **~45 s** (warm) | **~3×** |
| single-ref edit | ~4 min 20 s | **~1 min 51 s** | **~2.3×** |
| VRAM | ~112 GB | **~66 GB** | ~40% less |

Warm steady rate ≈ **1.6 s/it**. Output quality is visually on par with BF16
(mixed quant keeps the accuracy-sensitive layers in BF16 — see below).

## Requirements

- **NVIDIA Blackwell** (sm_120a / sm_121a). The NVFP4 Triton kernels need it;
  this will not run on Ada/Hopper.
- **CUDA 13** (the image is built on `nvidia/cuda:13.2.0`).
- Enough VRAM/unified memory for the model — on-the-fly quant loads the BF16
  weights resident first, so peak is BF16-sized during startup. A 128 GB GB10
  is comfortable; a 16 GB card is not.
- An HF token in your environment if the model is gated (FLUX.2-dev is).

## Build

Self-contained — build with this folder as the context:

```bash
docker build -f servers/flux2/Dockerfile -t flux2-torchao:latest servers/flux2
```

> The `torch`/`torchao`/`mslk` installs are Blackwell/CUDA-13 nightly builds;
> the first build is slow. See the "Pinning" note in the Dockerfile.

## Run

Via the repo's recipe runner:

```bash
./run-recipe.sh flux2-dev-torchao-nvfp4.yaml --name flux2 --solo -d
docker logs -f flux2      # watch: quantize (first boot) → save → serve on :8000
```

First boot of a given quant config: BF16 load + quantize + save (a few minutes).
Later boots **fast-load the saved quant** from the HF cache mount (~seconds of
weight load; the only remaining cost is the one-time `torch.compile` pass).

## How it works

Nothing exotic — it's stock diffusers + torchao, wired together so the quantize
step happens once and the result is reused. The whole pipeline:

### 1. Load & quantize (startup)

```
DiffusionPipeline.from_pretrained(MODEL_PATH, quantization_config=PipelineQuantizationConfig(...))
```

- The **BF16 checkpoint loads normally** (HF cache or local path). diffusers
  picks the concrete pipeline class from `model_index.json`, so the same image
  serves FLUX.2-dev, Klein, etc.
- torchao then **quantizes the transformer's linear layers in place** to NVFP4
  using `NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True)` (or
  MXFP8). This is a **real low-precision compute path**, not weight-only packing:
  weights are FP4 and activations are dynamically quantized to FP4 per-tensor,
  so the matmuls run on Blackwell's FP4 tensor cores via torchao's Triton
  kernels. That's where the ~2–3× wall-clock speedup comes from — it is *not* a
  memory-only trick (weight-only FP4 with BF16 math would save VRAM but not
  time). The approach is exactly the one documented in the
  [PyTorch Blackwell diffusion blog](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/).
- The text encoder and VAE stay BF16 (they aren't the bottleneck), so peak
  memory during startup is BF16-sized; the steady state is ~40% smaller.

### 2. Mixed precision (quality safeguard)

Fully quantizing every layer can hurt quality on the accuracy-sensitive ones
(patch/context embedders, timestep embed, final projection). `TorchAoConfig`'s
`modules_to_not_convert` keeps those in BF16 while NVFP4-ing the heavy
attention/MLP stack — the bulk of the FLOPs, so you keep almost all the speed
and lose almost none of the quality. The list is the `TORCHAO_SKIP_MODULES` env
(substring match; empty = full quant).

### 3. Save once, fast-boot after

Quantizing on every boot is wasteful, so with `QUANT_CACHE_DIR` set the
quantized transformer is written out with `save_pretrained(...)` after the first
quantize and reloaded on subsequent boots:

- The cache subdir is **content-addressed**: `md5(MODEL_PATH + quant + skip)[:12]`.
  Different quant/skip settings get different subdirs automatically, so they
  coexist and switching back to a prior config still fast-loads — no stale
  cache, no manual cleanup.
- The quant is saved as **pickled** (`safe_serialization=False`) because torchao
  tensor subclasses don't round-trip through safetensors. Reload therefore
  passes `use_safetensors=False` so `from_pretrained` reads the `.bin` shard
  index instead of probing for a non-existent `.safetensors`.
- Every step is wrapped so a cache miss/corruption **degrades gracefully** back
  to a fresh on-the-fly quantize rather than failing to start.
- First boot of a config: BF16 load + quantize + save (a few minutes). Later
  boots: the weights load in seconds; the only remaining cost is the one-time
  `torch.compile` pass (`compile_repeated_blocks`).

### 4. Serving

- Endpoints mirror the **OpenAI Images API**, so any OpenAI-style client works;
  `steps`/`guidance`/`seed` are accepted as extensions.
- A diffusers pipeline isn't thread-safe and one GPU can't run two denoise loops
  at once, so concurrent requests are **serialized behind a lock** (they queue,
  they don't crash).
- Long generations **stream whitespace keepalive** while the pipeline runs, so a
  fronting reverse-proxy / CDN idle timeout doesn't kill a multi-minute request.

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `MODEL_PATH` | — | base BF16 repo: an HF id (downloaded to the HF cache) or a local path |
| `SERVED_MODEL_NAME` | `MODEL_PATH` | id reported by `/v1/models` |
| `TORCHAO_QUANT` | *(unset = BF16)* | `nvfp4` (W4A4, fastest) · `mxfp8` (W8A8, higher accuracy) |
| `TORCHAO_SKIP_MODULES` | `proj_out,x_embedder,context_embedder,time_guidance_embed` | comma-separated linear-name substrings kept in BF16 (mixed quant). `""` = full quant |
| `QUANT_CACHE_DIR` | *(unset = no cache)* | where to save/load the quantized transformer. Content-addressed by `md5(model+quant+skip)` — see below |
| `DEFAULT_STEPS` | `4` | inference steps when the request omits `steps` (dev: use 28) |
| `DEFAULT_GUIDANCE` | `4.0` | guidance scale when the request omits `guidance` |
| `KEEPALIVE_SECS` | `15` | whitespace keepalive interval for long generations (proxy 5xx avoidance) |

### Mixed (selective) quant

Some layers are accuracy-sensitive; `TORCHAO_SKIP_MODULES` keeps them in BF16 and
NVFP4s the rest. **Verify the names against your model** — the defaults are a
FLUX.2 best-guess; substrings that match nothing are silently ignored (→ full
quant). List module names with:

```python
from diffusers import Flux2Transformer2DModel
m = Flux2Transformer2DModel.from_pretrained(MODEL_PATH, subfolder="transformer")
print({n.split('.')[0] for n,_ in m.named_modules()})
```

### Save-once / content-addressed cache

With `QUANT_CACHE_DIR` set, the quantized transformer is saved after the first
quantize and reloaded on later boots. The subdir is keyed by
`md5(MODEL_PATH + quant + skip)`, so different configs coexist and switching back
to a prior config still fast-loads — no stale cache, no manual deletion. Point
`QUANT_CACHE_DIR` at a **mounted** path (e.g. under the auto-mounted HF cache) so
it persists across container rebuilds.

## API (OpenAI Images-compatible)

| method | path | body |
|---|---|---|
| `GET` | `/v1/models` | — |
| `GET` | `/health` | — |
| `POST` | `/v1/images/generations` | JSON |
| `POST` | `/v1/images/edits` | multipart form (reference image upload) |

`steps`, `guidance`, and `seed` are non-standard extensions (pass via `extra_body`
with the OpenAI SDK, or plain JSON / `-F` form fields). Responses return
`b64_json` (no hosted URLs).

**Generate:**

```bash
curl http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red fox in snow at golden hour","size":"1024x1024","steps":28}'
```

**Edit / reference** (accepts repeated `image` fields or `image[]`):

```bash
curl http://localhost:8000/v1/images/edits \
  -F prompt="make it a snowy night under a full moon, keep the fox" \
  -F image=@fox.png \
  -F size="1024x1024" -F steps=28
```

## Notes & gotchas

- **First request of each new *shape* pays a one-time Triton JIT** (frozen at
  `0/N`, GPU idle for a minute or two). NVFP4 tensors have no stable
  `torch.compile` cache key, so it recompiles per resolution/aspect and per
  gen-vs-edit, and does **not** persist across restart. Warm the shapes you use.
- **Reference edits are slower** than plain generation (the reference adds
  conditioning tokens → longer sequence). Multi-reference is slower still;
  downscale references to cut tokens.
- **HF downloads are silent in `docker logs`** (no TTY) — a first run sits at
  "loading …" while tens of GB stream. Track with `du -sh` on the HF cache, not
  the log.
- One GPU + a non-thread-safe pipeline: concurrent requests are **serialized**
  behind a lock (they queue, they don't race).

## Credit

Quantization approach from the PyTorch team's
[Faster Diffusion on Blackwell (MXFP8 / NVFP4 + torchao)](https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/).
