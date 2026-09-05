"""OpenAI-compatible image server for FLUX.2 (diffusers).

Serves any FLUX.2 checkpoint — the concrete pipeline class (Flux2Pipeline,
Flux2KleinPipeline, …) is auto-detected from the model's model_index.json, so
one image serves FLUX.2-dev, Klein, etc. The transformer can be quantized to
NVFP4/MXFP8 on the fly via torchao for a large Blackwell speedup (see README).

Endpoints mirror the OpenAI Images API so standard OpenAI clients work:
  POST /v1/images/generations  (JSON)             text-to-image
  POST /v1/images/edits        (multipart form)   image edit / multi-reference
  GET  /v1/models  ·  GET /health

OpenAI has no steps/guidance/seed fields; we accept them as optional extensions
(clients pass via extra_body / extra form fields), defaulting per model via
DEFAULT_STEPS/DEFAULT_GUIDANCE. response_format defaults to b64_json (no hosted
URLs). Long generations stream whitespace keepalive so a fronting proxy's idle
timeout doesn't fire; auth, if any, belongs at that proxy hop, not here.
"""
import base64
import io
import json
import os
import threading
import time

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

KEEPALIVE_SECS = int(os.environ.get("KEEPALIVE_SECS", "15"))


def _streaming_json(work):
    """Run work() (which returns a JSON-able dict) in a thread; emit a space
    every KEEPALIVE_SECS while it runs so Cloudflare's ~100s idle timeout never
    fires on long (minutes) non-streaming generation, then emit the JSON body.
    Leading whitespace is valid JSON — clients ignore it. NOTE: status is 200 as
    soon as streaming starts, so errors are reported in the body as {"error":..}.
    """
    box = {}

    def run():
        try:
            box["result"] = work()
        except HTTPException as e:
            box["error"] = {"message": str(e.detail), "code": e.status_code}
        except Exception as e:  # noqa: BLE001
            box["error"] = {"message": f"{type(e).__name__}: {e}", "code": 500}

    t = threading.Thread(target=run, daemon=True)
    t.start()

    def gen():
        while True:
            t.join(timeout=KEEPALIVE_SECS)
            if not t.is_alive():
                break
            yield b" "
        yield json.dumps(box.get("result") or {"error": box["error"]}).encode()

    return StreamingResponse(gen(), media_type="application/json")

# MODEL_PATH is normally set by the recipe (an HF id or a local path). The
# fallback is the flagship FLUX.2-dev repo so a bare run still does something
# sensible (it downloads to the HF cache).
MODEL_PATH = os.environ.get("MODEL_PATH", "black-forest-labs/FLUX.2-dev")
MODEL_ID = os.environ.get("SERVED_MODEL_NAME", MODEL_PATH)
# Per-model defaults via env: FLUX.2-dev is not distilled (28-50 steps); the
# distilled Klein needs only ~4. Both use guidance ~4. The recipe sets these.
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "28"))
DEFAULT_GUIDANCE = float(os.environ.get("DEFAULT_GUIDANCE", "4.0"))
_CREATED = int(time.time())

app = FastAPI(title="FLUX.2 (OpenAI-compatible)")

print(f"loading {MODEL_PATH} (default steps={DEFAULT_STEPS}) ...", flush=True)
# DiffusionPipeline auto-detects the concrete class (Flux2KleinPipeline /
# Flux2Pipeline / …) from the checkpoint's model_index.json — so one server
# image serves both klein and FLUX.2-dev.
from diffusers import DiffusionPipeline
# TORCHAO_QUANT selects on-the-fly torchao quantization of the transformer:
#   nvfp4 → W4A4 fp4, fastest, Triton kernels (needs mslk)
#   mxfp8 → W8A8 fp8, higher accuracy, a bit less speed
#   (unset/off/bf16) → no quant, plain BF16
# The Triton kernels JIT for the running arch (sm_120a/121a on Blackwell). The
# BF16 weights load first and are quantized in place, so MODEL_PATH is always
# the normal BF16 repo.
TORCHAO_QUANT = os.environ.get("TORCHAO_QUANT", "").lower()
if TORCHAO_QUANT in ("nvfp4", "mxfp8"):
    from diffusers import TorchAoConfig, PipelineQuantizationConfig
    if TORCHAO_QUANT == "nvfp4":
        from torchao.prototype.mx_formats.inference_workflow import (
            NVFP4DynamicActivationNVFP4WeightConfig,
        )
        ao_cfg = NVFP4DynamicActivationNVFP4WeightConfig(
            use_dynamic_per_tensor_scale=True, use_triton_kernel=True)
    else:  # mxfp8
        from torchao.prototype.mx_formats.inference_workflow import (
            MXDynamicActivationMXWeightConfig,
        )
        from torchao.prototype.mx_formats.constants import KernelPreference
        ao_cfg = MXDynamicActivationMXWeightConfig(
            activation_dtype=torch.float8_e4m3fn, weight_dtype=torch.float8_e4m3fn,
            kernel_preference=KernelPreference.AUTO)
    # MIXED / selective quant: keep accuracy-critical linears (embeddings, final
    # projection) in BF16; NVFP4 only the heavy attention/MLP layers. Substring
    # match; override per model via TORCHAO_SKIP_MODULES (comma-sep, "" = full).
    # Verify names against transformer.named_modules(). Built here because it
    # keys the cache.
    skip = [s.strip() for s in os.environ.get(
        "TORCHAO_SKIP_MODULES",
        "proj_out,x_embedder,context_embedder,time_guidance_embed").split(",") if s.strip()]

    # CONTENT-ADDRESSED cache: key by (model, quant, skip) so each distinct
    # config gets its own subdir automatically — no stale cache, no manual
    # deletion. Change quant/skip → new key → quantizes fresh & caches there;
    # switch back → the old key's cache is still there and fast-loads. First
    # boot of a key: ~2min BF16 load + quant + save; later boots: fast load.
    QUANT_CACHE = os.environ.get("QUANT_CACHE_DIR")
    cache_path = None
    if QUANT_CACHE:
        import hashlib
        key = hashlib.md5(
            f"{MODEL_PATH}|{TORCHAO_QUANT}|{','.join(skip)}".encode()).hexdigest()[:12]
        cache_path = os.path.join(QUANT_CACHE, key)
        print(f"  quant cache key {key} ({TORCHAO_QUANT}, skip={skip or 'none'})", flush=True)

    loaded = False
    if cache_path and os.path.isdir(cache_path) and os.listdir(cache_path):
        try:
            from diffusers import Flux2Transformer2DModel
            print(f"  loading cached transformer from {cache_path}", flush=True)
            # use_safetensors=False: the quant is saved as pickled (sharded)
            # .bin — without this, from_pretrained probes only the safetensors
            # names and the single-file .bin, never the .bin.index.json, and
            # falls through to "no file found" → needless re-quantize.
            tf = Flux2Transformer2DModel.from_pretrained(
                cache_path, torch_dtype=torch.bfloat16, use_safetensors=False)
            pipe = DiffusionPipeline.from_pretrained(
                MODEL_PATH, transformer=tf, torch_dtype=torch.bfloat16).to("cuda")
            loaded = True
        except Exception as e:  # noqa: BLE001
            print(f"  cache load failed ({e}); quantizing fresh", flush=True)
    if not loaded:
        try:
            tao = TorchAoConfig(ao_cfg, modules_to_not_convert=skip) if skip else TorchAoConfig(ao_cfg)
        except TypeError:  # config doesn't accept skip list on this version → full
            print("  (modules_to_not_convert unsupported here; full quant)", flush=True)
            tao = TorchAoConfig(ao_cfg)
        print(f"  quantizing on-the-fly with torchao {TORCHAO_QUANT.upper()} (triton); "
              f"BF16-kept: {skip or 'none (full)'}", flush=True)
        pqc = PipelineQuantizationConfig(quant_mapping={"transformer": tao})
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, quantization_config=pqc).to("cuda")
        if cache_path:
            try:
                os.makedirs(cache_path, exist_ok=True)
                pipe.transformer.save_pretrained(cache_path, safe_serialization=False)
                print(f"  saved quantized transformer → {cache_path} (fast boot next time)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  cache save failed ({e}); staying on-the-fly", flush=True)
    try:
        pipe.transformer.compile_repeated_blocks(fullgraph=True)
        print("  compiled repeated blocks", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  compile_repeated_blocks skipped: {e}", flush=True)
else:
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
print(f"pipeline loaded: {type(pipe).__name__}", flush=True)

# A diffusers pipeline is NOT thread-safe and one GPU can't run concurrent
# denoise loops. FastAPI runs sync endpoints in a threadpool, so N simultaneous
# requests would call pipe() at once → CUDA races / OOM / crash. Serialize:
# concurrent callers queue and run one at a time.
_gpu_lock = threading.Lock()


def _parse_size(size: str):
    try:
        w, h = size.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise HTTPException(400, f"bad size '{size}', want WIDTHxHEIGHT e.g. 1024x1024")


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _run(prompt, steps, guidance, w, h, seed, n, refs=None):
    """Run the pipeline n times (seed+i each), return list of PIL images."""
    out_imgs = []
    for i in range(max(1, n)):
        gen = torch.Generator(device="cuda").manual_seed(seed + i)
        kwargs = dict(prompt=prompt, num_inference_steps=steps,
                      guidance_scale=guidance, width=w, height=h, generator=gen)
        if refs:
            # Pass references UNMODIFIED (never resize — that would distort a
            # reference you want to keep). width/height still request the output
            # size; whether FLUX.2 honors them independently of the reference
            # depends on the pipeline's param names — see note below.
            kwargs["image"] = refs[0] if len(refs) == 1 else refs
        try:
            with _gpu_lock, torch.inference_mode():
                out_imgs.append(pipe(**kwargs).images[0])
        except Exception as e:
            raise HTTPException(500, f"{type(e).__name__}: {e}")
    return out_imgs


def _payload(imgs, t0):
    return {
        "created": int(time.time()),
        "data": [{"b64_json": _b64_png(im)} for im in imgs],
        "_elapsed_s": round(time.time() - t0, 2),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


def _model_obj():
    return {"id": MODEL_ID, "object": "model", "created": _CREATED,
            "owned_by": "black-forest-labs"}


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [_model_obj()]}


@app.get("/v1/models/{model_id:path}")
def get_model(model_id: str):
    return _model_obj()


class GenerationsRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    # non-standard extensions; default per-model via DEFAULT_STEPS/_GUIDANCE env
    seed: int = 0
    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE


@app.post("/v1/images/generations")
def images_generations(req: GenerationsRequest):
    t0 = time.time()
    w, h = _parse_size(req.size)
    # stream keepalive whitespace during the (possibly minutes-long) gen so
    # Cloudflare's 100s idle timeout never fires on this non-streaming endpoint
    return _streaming_json(
        lambda: _payload(_run(req.prompt, req.steps, req.guidance,
                              w, h, req.seed, req.n), t0))


@app.post("/v1/images/edits")
async def images_edits(
    prompt: str = Form(...),
    # Reference images arrive under TWO spellings in the wild: repeated `image` fields (curl /
    # FastAPI convention) and `image[]` (how the official OpenAI SDKs serialize an array).
    # Liberal server: accept both and merge, so any standard client works.
    image: list[UploadFile] = File(default=[]),
    image_arr: list[UploadFile] = File(default=[], alias="image[]"),
    model: str | None = Form(None),
    n: int = Form(1),
    size: str = Form("1024x1024"),
    response_format: str = Form("b64_json"),
    seed: int = Form(0),
    steps: int = Form(DEFAULT_STEPS),
    guidance: float = Form(DEFAULT_GUIDANCE),
):
    t0 = time.time()
    files = [*image, *image_arr]
    if not files:
        raise HTTPException(422, "at least one reference image is required (multipart field `image`, or `image[]`)")
    # decode uploads + parse size up front (async/fast); the slow pipeline call
    # runs inside the keepalive stream so Cloudflare doesn't 524 on long edits
    try:
        refs = [Image.open(io.BytesIO(await f.read())).convert("RGB") for f in files]
    except Exception as e:
        raise HTTPException(400, f"cannot decode image: {e}")
    w, h = _parse_size(size)
    return _streaming_json(
        lambda: _payload(_run(prompt, steps, guidance, w, h, seed, n, refs=refs), t0))
