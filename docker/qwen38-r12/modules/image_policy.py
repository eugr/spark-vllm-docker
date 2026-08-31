"""Initial, inline-only Qwen3.8 image admission candidate (not GPU qualification).

Integration, installed as site-packages/qwen38_image_policy.py::

    vllm serve MODEL --middleware qwen38_image_policy.ImagePolicyMiddleware \
        --limit-mm-per-prompt '{"image":32,"video":0}' \
        --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":2097152}'

Also set VLLM_MAX_IMAGE_PIXELS=16777216 and VLLM_MEDIA_LOADING_THREAD_COUNT=1.
Use one API process: the admission gate is per middleware instance/process.
Only POST /v1/chat/completions is enabled for inference, including text-only
generation. Health, models, version and metrics GET/HEAD remain available.
Other endpoints, remote/file URLs, media UUIDs, embeddings, video/audio and
client processor/media overrides are deliberately unsupported. Public remote
URL support needs a separately bounded fetch implementation before release.

One gate admits one upload/validation at a time with 16 *unread* waiters.
A separate image gate admits one downstream image request through completion;
busy image submissions return 429 instead of queuing full request bodies.
Text bypasses that second gate, so text decodes are not serialized. A 128 MiB
global logical-body reservation bounds buffering across active requests (not
total Python heap or upstream transport buffering). The unread queue has a
120-second deadline; uploads have
a 30-second total deadline. The two pipeline stages can overlap.
Response bodies/SSE and disconnects are passed through, never buffered here.
This bounds admission, not resident engine requests, encoder/GPU caches, or
total host memory. Native vLLM still validates combined text/image context.

Inspected runtime: vLLM 0.1.dev20073+g8e685d198, Qwen3VLProcessingInfo and
transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize. Use the
installed native function, not an area approximation or a second resize kernel.
The checkpoint geometry is patch_size=16, merge_size=2, temporal_patch_size=2.
The checkpoint's min_pixels=65536 is preserved. Pinned Transformers source,
Qwen2VLImageProcessor._standardize_kwargs (read from stopped container):

    if min_pixels is not None and max_pixels is not None:
        size = SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels)
    return super()._standardize_kwargs(size=size, **kwargs)

Pinned vLLM MultiModalConfig.merge_mm_processor_kwargs returns
``kwargs | dict(inference_kwargs)``. Reject client overrides before injecting
both min/max and size: this fixes native counting AND actual preprocessing.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import json
import logging
from typing import Callable

from PIL import Image

MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_BUFFERED_BODY_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_PIXELS = 16777216
MAX_PIXELS = 2097152
MIN_PIXELS = 65536
MAX_IMAGES = 32
MAX_VISUAL_TOKENS = 32768
MAX_GENERATION_TOKENS = 32768
MAX_WAITERS = 16
QUEUE_TIMEOUT = 120.0
UPLOAD_TIMEOUT = 30.0
FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
READ_PATHS = {"/health", "/v1/models", "/version", "/metrics"}
FORBIDDEN_FIELDS = {
    "mm_processor_kwargs", "media_io_kwargs", "multi_modal_data",
    "multi_modal_uuids", "image_embeds", "audio_embeds", "prompt_embeds",
    "inputs_embeds", "input_embeds", "pixel_values", "image_grid_thw",
    "ec_transfer_params", "kv_transfer_params", "vllm_xargs", "extra_body",
    "chat_template", "documents", "audio", "modalities",
}
logger = logging.getLogger(__name__)


def fixed_processor_kwargs() -> dict:
    # Both min/max are necessary: the pinned HF _standardize_kwargs only
    # replaces size when BOTH are supplied. Fix geometry and resizing too.
    return {
        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
        "size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
        "do_resize": True, "patch_size": 16, "merge_size": 2,
        "temporal_patch_size": 2,
    }


class PolicyError(Exception):
    def __init__(self, code: str, message: str, param: str | None = None,
                 status: int = 400):
        super().__init__(message)
        self.code, self.param, self.status = code, param, status


def native_smart_resize(**kwargs):
    # Lazy import keeps CPU unit tests independent of torch/vLLM. There is
    # deliberately no fallback if the installed processor is incompatible.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    return smart_resize(**kwargs)


@dataclass(frozen=True)
class ImageBudget:
    width: int
    height: int
    resized_width: int
    resized_height: int
    visual_tokens: int


def image_budget(url: str, param: str,
                 resize: Callable = native_smart_resize) -> ImageBudget:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise PolicyError("unsupported_image_url",
                          "Initial candidate accepts inline base64 images only; "
                          "remote and file URLs are disabled.", param)
    header, separator, encoded = url.partition(",")
    mime = header[5:].removesuffix(";base64")
    if not separator or header != f"data:{mime};base64" or mime not in FORMATS:
        raise PolicyError("unsupported_image_format",
                          "Use base64 data URLs containing JPEG, PNG or WebP still images.", param)
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise PolicyError("image_bytes_exceeded", "Inline image exceeds 8 MiB.", param, 413)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise PolicyError("invalid_image", "Invalid base64 image.", param) from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise PolicyError("image_bytes_exceeded", "Inline image exceeds 8 MiB.", param, 413)
    try:
        # Check header dimensions BEFORE frame checks, verify or raster load.
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_SOURCE_PIXELS:
                raise PolicyError("source_pixels_exceeded",
                                  "Source image exceeds 16777216 pixels.", param)
            if image.format != FORMATS[mime]:
                raise PolicyError("unsupported_image_format",
                                  "Image bytes must match the declared JPEG/PNG/WebP type.", param)
            if getattr(image, "n_frames", 1) != 1:
                raise PolicyError("animated_image_disabled", "Only single-frame still images are supported.", param)
            image.verify()
        # Catch corrupt raster data here, one source at a time. No decoded
        # source is retained across images or requests by this component.
        with Image.open(BytesIO(raw)) as image:
            image.load()
            # Read EXIF only after verification on a fresh stream: PNG EXIF
            # inspection may load/close its stream, invalidating verify().
            # Native vLLM normalize_image transposes these same orientations.
            if image.getexif().get(274) in (5, 6, 7, 8):
                width, height = height, width
    except PolicyError:
        raise
    except Image.DecompressionBombError as exc:
        raise PolicyError("source_pixels_exceeded", "Source image exceeds 16777216 pixels.", param) from exc
    except (OSError, ValueError, SyntaxError, EOFError) as exc:
        raise PolicyError("invalid_image", "Image header or raster is invalid.", param) from exc
    try:
        resized_height, resized_width = resize(
            height=height, width=width, factor=32,
            min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
        )
    except ValueError as exc:
        raise PolicyError("invalid_image_dimensions",
                          "Image aspect ratio must not exceed 200:1.", param) from exc
    area = resized_height * resized_width
    if (resized_height <= 0 or resized_width <= 0 or resized_height % 32
            or resized_width % 32 or area > MAX_PIXELS):
        raise PolicyError("invalid_processor_geometry",
                          "Native processor dimensions violate the fixed image policy.", param)
    return ImageBudget(width, height, resized_width, resized_height, area // 1024)


def _reject_fields(value: dict, param: str) -> None:
    if any(key in FORBIDDEN_FIELDS for key in value):
        raise PolicyError("unsupported_override",
                          "Processor/media overrides, embeddings and alternate inputs are disabled.", param)


def _images(payload: dict) -> list[tuple[str, str]]:
    _reject_fields(payload, "request")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise PolicyError("invalid_request", "messages must be a nonempty array.", "messages")
    images = []
    # Complete structural pre-scan rejects videos/embeddings/count excess
    # before ANY image decoder or native multimodal parser is called.
    for index, message in enumerate(messages):
        param = f"messages[{index}].content"
        if not isinstance(message, dict):
            raise PolicyError("invalid_request", "Each message must be an object.", param)
        _reject_fields(message, param)
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise PolicyError("invalid_request", "Content must be text or a parts array.", param)
        for part_index, part in enumerate(content):
            part_param = f"{param}[{part_index}]"
            if isinstance(part, str):
                continue
            if not isinstance(part, dict):
                raise PolicyError("invalid_request", "Content part must be an object.", part_param)
            kind = part.get("type")
            if kind in ("text", "refusal"):
                if set(part) - {"type", kind} or not isinstance(part.get(kind), str):
                    raise PolicyError("invalid_request", "Invalid text/refusal part.", part_param)
            elif kind == "image_url":
                if set(part) - {"type", "image_url"}:
                    raise PolicyError("unsupported_override", "Image UUIDs and extra image fields are disabled.", part_param)
                image = part.get("image_url")
                if not isinstance(image, dict) or set(image) - {"url", "detail"}:
                    raise PolicyError("invalid_request", "image_url must contain url and optional detail only.", part_param)
                if image.get("detail", "auto") not in ("auto", "low", "high"):
                    raise PolicyError("invalid_request", "Invalid image detail.", part_param)
                # Qwen ignores OpenAI detail; it never changes the fixed budget.
                images.append((image.get("url"), part_param + ".image_url.url"))
            else:
                raise PolicyError("unsupported_modality",
                                  "Only explicit text, refusal and image_url parts are supported; "
                                  "video, audio and embeddings are disabled.", part_param)
    if len(images) > MAX_IMAGES:
        raise PolicyError("too_many_images", f"Received {len(images)} images; maximum 32.", "messages")
    return images


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Non-finite JSON number")


def admit(body: bytes, resize: Callable = native_smart_resize, *, reject_images: bool = False) -> tuple[bytes, list[ImageBudget]]:
    try:
        payload = json.loads(body, object_pairs_hook=_unique_object,
                             parse_constant=_invalid_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise PolicyError("invalid_json", "Expected a valid JSON object without duplicate keys.") from exc
    if not isinstance(payload, dict):
        raise PolicyError("invalid_request", "Expected a JSON object.")
    images = _images(payload)
    if images and reject_images:
        raise PolicyError("server_busy", "Another image request is active; retry after it finishes.", status=429)
    for field in ("max_tokens", "max_completion_tokens", "min_tokens"):
        value = payload.get(field)
        minimum = 0 if field == "min_tokens" else 1
        if value is not None and (type(value) is not int or not minimum <= value <= MAX_GENERATION_TOKENS):
            raise PolicyError("generation_limit_exceeded",
                              f"{field} must be an integer between {minimum} and 32768.", field)
    # Pin omitted/null output defaults as well as explicit requests. vLLM's
    # tokenizer then checks total context against this effective output limit.
    output_limit = payload.get("max_completion_tokens") or payload.get("max_tokens") or MAX_GENERATION_TOKENS
    if payload.get("min_tokens", 0) is not None and payload.get("min_tokens", 0) > output_limit:
        raise PolicyError("invalid_request", "min_tokens exceeds the output limit.", "min_tokens")
    payload["max_completion_tokens"] = output_limit
    budgets = []
    aggregate = 0
    for url, param in images:
        budget = image_budget(url, param, resize)
        aggregate += budget.visual_tokens
        if aggregate > MAX_VISUAL_TOKENS:
            raise PolicyError("visual_token_limit_exceeded",
                              f"Images require more than 32768 visual tokens after native resizing ({aggregate} so far).",
                              "messages")
        budgets.append(budget)
    if images:
        payload["mm_processor_kwargs"] = fixed_processor_kwargs()
        # Prevent server media configuration from changing the admitted raster.
        payload["media_io_kwargs"] = {"image": {"image_mode": "RGB", "qwen38_early_resize": True}}
    try:
        rewritten = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise PolicyError("invalid_json", "JSON values are outside the supported range.") from exc
    if len(rewritten) > MAX_BODY_BYTES:
        raise PolicyError("payload_too_large", "Normalized request exceeds 64 MiB.", status=413)
    return rewritten, budgets


async def _error(send, error: PolicyError):
    body = json.dumps({"error": {"message": str(error), "type": "invalid_request_error"
                               if error.status != 429 else "rate_limit_error",
                               "param": error.param, "code": error.code}}).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if error.status == 429:
        headers.append((b"retry-after", b"1"))
    await send({"type": "http.response.start", "status": error.status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive, reserve=None) -> bytes | None:
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        if message["type"] != "http.request":
            raise PolicyError("invalid_request", "Unexpected request event.")
        chunk = message.get("body", b"")
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise PolicyError("payload_too_large", "Request body exceeds 64 MiB.", status=413)
        if reserve is not None:
            reserve(len(body) + len(chunk))
        body.extend(chunk)
        if not message.get("more_body", False):
            return bytes(body)


class ImagePolicyMiddleware:
    """Pure ASGI class suitable for vLLM --middleware (no BaseHTTPMiddleware)."""

    def __init__(self, app):
        self.app = app
        self._gate = asyncio.Lock()
        self._waiting = 0
        self._image_gate = asyncio.Lock()
        self._image_waiting = 0
        self._buffered_body_bytes = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        root = scope.get("root_path", "")
        if root and path.startswith(root + "/"):
            path = path[len(root):]
        path = path.rstrip("/")
        method = scope.get("method")
        if ((path in READ_PATHS and method in ("GET", "HEAD", "OPTIONS"))
                or (path == "/v1/chat/completions" and method == "OPTIONS")):
            await self.app(scope, receive, send)
            return
        if path != "/v1/chat/completions" or method != "POST":
            await _error(send, PolicyError("unsupported_endpoint",
                                          "Initial candidate supports POST /v1/chat/completions only for inference.", status=404))
            return
        acquired = False
        image_acquired = False
        ready = False
        worker = None
        body_reservation = 0

        def reserve_body(amount):
            nonlocal body_reservation
            change = amount - body_reservation
            if self._buffered_body_bytes + change > MAX_BUFFERED_BODY_BYTES:
                raise PolicyError("server_busy", "Request buffering budget is full; retry later.", status=429)
            self._buffered_body_bytes += change
            body_reservation = amount

        def release_body(_future=None):
            nonlocal body_reservation
            self._buffered_body_bytes -= body_reservation
            body_reservation = 0

        def release(_future=None):
            nonlocal acquired
            if acquired:
                acquired = False
                self._gate.release()

        def release_image():
            nonlocal image_acquired
            if image_acquired:
                image_acquired = False
                self._image_gate.release()

        try:
            headers = {}
            for key, value in scope.get("headers", []):
                key = key.lower()
                if key in (b"content-length", b"content-type", b"content-encoding") and key in headers:
                    raise PolicyError("invalid_request", "Duplicate request content headers.")
                headers[key] = value
            if headers.get(b"content-encoding", b"identity").lower() != b"identity":
                raise PolicyError("unsupported_encoding", "Compressed request bodies are disabled.", status=415)
            if headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower() != b"application/json":
                raise PolicyError("unsupported_content_type", "Use application/json.", status=415)
            if b"content-length" in headers:
                length = headers[b"content-length"]
                if not length.isdigit() or len(length) > 20:
                    raise PolicyError("invalid_request", "Invalid Content-Length.")
                if int(length) > MAX_BODY_BYTES:
                    raise PolicyError("payload_too_large", "Request body exceeds 64 MiB.", status=413)
            # Reserve before reading, including chunked uploads. No await
            # between checking/updating this per-process event-loop counter.
            # Two logical copies cover input and normalized replay storage.
            reserve_body(2 * max(1, int(headers[b"content-length"]))
                         if b"content-length" in headers else 2 * MAX_BODY_BYTES)
            if self._gate.locked() and self._waiting >= MAX_WAITERS:
                raise PolicyError("server_busy", "Image admission queue is full; retry later.", status=429)
            self._waiting += 1
            try:
                await asyncio.wait_for(self._gate.acquire(), QUEUE_TIMEOUT)
                acquired = True
            except asyncio.TimeoutError as exc:
                raise PolicyError("server_busy", "Image admission queue deadline exceeded.", status=429) from exc
            finally:
                self._waiting -= 1
            try:
                body = await asyncio.wait_for(
                    _read_body(receive, lambda size: reserve_body(max(body_reservation, 2 * size))),
                    UPLOAD_TIMEOUT,
                )
            except asyncio.TimeoutError as exc:
                raise PolicyError("upload_timeout", "Request upload deadline exceeded.", status=408) from exc
            if body is None:
                return
            if b"content-length" in headers and int(headers[b"content-length"]) != len(body):
                raise PolicyError("invalid_request", "Content-Length does not match the request body.")
            worker = asyncio.create_task(asyncio.to_thread(
                admit, body, reject_images=self._image_gate.locked() or bool(self._image_waiting)))
            # Cancellation cannot stop a decoder thread. Keep the gate until
            # that worker finishes, even if this ASGI task is cancelled.
            rewritten, budgets = await asyncio.shield(worker)
            worker = None  # A completed Task otherwise keeps its result alive.
            reserve_body(2 * max(len(body), len(rewritten)))
            del body
            release()
            if budgets:
                if self._image_gate.locked() or self._image_waiting:
                    raise PolicyError("server_busy", "Another image request is active; retry after it finishes.", status=429)
                self._image_waiting += 1
                try:
                    await asyncio.wait_for(self._image_gate.acquire(), QUEUE_TIMEOUT)
                    image_acquired = True
                except asyncio.TimeoutError as exc:
                    raise PolicyError("server_busy", "Image processing queue deadline exceeded.", status=429) from exc
                finally:
                    self._image_waiting -= 1
                logger.info("Qwen image admission: dimensions=%s aggregate_visual_tokens=%d",
                            [(b.width, b.height, b.resized_width, b.resized_height) for b in budgets],
                            sum(b.visual_tokens for b in budgets))
            downstream_scope = dict(scope)
            downstream_scope["headers"] = [
                (k, v) for k, v in scope.get("headers", [])
                if k.lower() not in (b"content-length", b"transfer-encoding")
            ] + [(b"content-length", str(len(rewritten)).encode())]
            downstream_scope["state"] = dict(scope.get("state", {}))
            downstream_scope["state"]["qwen38_visual_tokens"] = sum(b.visual_tokens for b in budgets)

            async def replay():
                nonlocal rewritten
                if rewritten is not None:
                    event = {"type": "http.request", "body": rewritten, "more_body": False}
                    rewritten = None
                    return event
                return await receive()

            async def passthrough(message):
                # Headers are NOT proof that encoder work has finished. Hold
                # the image slot until downstream completion/abort unwinds.
                await send(message)

            # Outside the PolicyError handler: don't turn a downstream failure
            # after response start into an invalid second HTTP response.
            ready = True
        except PolicyError as exc:
            release()
            await _error(send, exc)
            return
        finally:
            # A cancelled decoder retains the upload gate until its thread ends.
            if worker is not None and not worker.done():
                worker.add_done_callback(release)
                worker.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            else:
                release()
            if not ready:
                release_image()
                if worker is not None and not worker.done():
                    worker.add_done_callback(release_body)
                else:
                    release_body()
        try:
            await self.app(downstream_scope, replay, passthrough)
        finally:
            release_image()
            release_body()
