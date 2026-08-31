"""CPU-only admission tests; no vLLM, torch, model, Docker or network required.

    python3 tests/test_qwen38_image_policy.py

Requires Python >=3.10 and Pillow. For exact pinned-source qualification, pipe
the inspected Transformers image_processing_qwen2_vl.py to this command with
--native-smart-resize-stdin. Only its smart_resize AST is executed (with math);
the heavy model/torch imports are not executed. Never fetch source in tests.
"""

import ast
import asyncio
import base64
from io import BytesIO
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / "docker/qwen38-r12/modules/image_policy.py"
spec = importlib.util.spec_from_file_location("qwen38_image_policy", MODULE)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def cpu_resize(height, width, factor=32, min_pixels=65536, max_pixels=2097152):
    # Minimal CPU fixture transcribed from Apache-2.0 Qwen/Hugging Face
    # smart_resize. The optional stdin mode replaces it with the pinned AST.
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_url(size=(256, 256), fmt="PNG", mode="RGB", **kwargs):
    image = Image.new(mode, size, 0)
    output = BytesIO()
    image.save(output, format=fmt, **kwargs)
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "GIF": "gif"}[fmt]
    return f"data:image/{mime};base64," + base64.b64encode(output.getvalue()).decode()


def image_part(url=None, **extra):
    return {"type": "image_url", "image_url": {"url": url or image_url()}, **extra}


def payload(parts=None, **kwargs):
    return {"messages": [{"role": "user", "content": "Hello" if parts is None else parts}], **kwargs}


def admit(value):
    return policy.admit(json.dumps(value).encode(), resize=cpu_resize)


class AdmissionTests(unittest.TestCase):
    def assert_rejected(self, value, code):
        with self.assertRaises(policy.PolicyError) as caught:
            admit(value)
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_native_dimension_examples(self):
        for size, processed, tokens in [
            ((32, 32), (256, 256), 64),
            ((512, 512), (512, 512), 256),
            ((1024, 1024), (1024, 1024), 1024),
            ((2048, 2048), (1440, 1440), 2025),
            ((4096, 4096), (1440, 1440), 2025),
            ((2048, 1024), (2048, 1024), 2048),
            ((1920, 1080), (1920, 1088), 2040),
            ((1080, 1920), (1088, 1920), 2040),
        ]:
            with self.subTest(size=size):
                result = policy.image_budget(image_url(size), "image", cpu_resize)
                self.assertEqual((result.resized_width, result.resized_height), processed)
                self.assertEqual(result.visual_tokens, tokens)

    def test_still_formats_transparency_and_exif(self):
        for fmt in ("PNG", "JPEG", "WEBP"):
            with self.subTest(format=fmt):
                result = policy.image_budget(image_url((512, 256), fmt), "image", cpu_resize)
                self.assertEqual(result.visual_tokens, 128)
        result = policy.image_budget(image_url(mode="RGBA"), "image", cpu_resize)
        self.assertEqual(result.visual_tokens, 64)
        exif = Image.Exif()
        exif[274] = 6
        result = policy.image_budget(image_url((512, 256), "JPEG", exif=exif), "image", cpu_resize)
        self.assertEqual((result.width, result.height), (256, 512))
        self.assertEqual((result.resized_width, result.resized_height), (256, 512))

    def test_header_limit_before_load_verify_or_frames(self):
        class Header:
            size = (4097, 4096)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def __getattr__(self, name):
                raise AssertionError(f"Touched {name} before checking source pixels")
        with patch.object(policy.Image, "open", return_value=Header()):
            self.assert_rejected(payload([image_part()]), "source_pixels_exceeded")

    def test_exact_source_limit_accepted_and_one_over_rejected(self):
        self.assertEqual(policy.image_budget(image_url((4096, 4096)), "image", cpu_resize).visual_tokens, 2025)
        self.assert_rejected(payload([image_part(image_url((4097, 4096)))]), "source_pixels_exceeded")

    def test_remote_file_and_cache_urls_rejected(self):
        for url in ("https://example.invalid/a.png", "http://127.0.0.1/a", "file:///tmp/a", "/tmp/a"):
            with self.subTest(url=url), patch.object(policy.Image, "open", side_effect=AssertionError("decoder")):
                self.assert_rejected(payload([image_part(url)]), "unsupported_image_url")
        for item in ({"type": "image_url", "image_url": {"url": None}},
                     image_part(uuid="cache-key"),
                     {"type": "image_url", "image_url": {"url": None, "uuid": "cache-key"}}):
            with self.assertRaises(policy.PolicyError):
                admit(payload([item]))

    def test_invalid_format_base64_corruption_and_animation(self):
        cases = [
            ("data:image/png;base64,!!!", "invalid_image"),
            ("data:image/png,hello", "unsupported_image_format"),
            (image_url(fmt="GIF"), "unsupported_image_format"),
            (image_url().replace("image/png", "image/jpeg"), "unsupported_image_format"),
            ("data:image/png;base64," + base64.b64encode(b"broken").decode(), "invalid_image"),
        ]
        for url, code in cases:
            self.assert_rejected(payload([image_part(url)]), code)
        for fmt in ("PNG", "WEBP"):
            first = Image.new("RGB", (32, 32), "red")
            second = Image.new("RGB", (32, 32), "blue")
            buffer = BytesIO()
            first.save(buffer, format=fmt, save_all=True, append_images=[second], duration=50)
            url = f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()
            self.assert_rejected(payload([image_part(url)]), "animated_image_disabled")

    def test_image_bytes_predecode_and_exact_decoded_limit(self):
        with patch.object(policy, "MAX_IMAGE_BYTES", 8), patch.object(policy.Image, "open", side_effect=AssertionError("decoder")):
            self.assert_rejected(payload([image_part("data:image/png;base64," + "A" * 16)]), "image_bytes_exceeded")
            # 9 decoded bytes fit the encoded-length rounding bucket for 8.
            self.assert_rejected(payload([image_part("data:image/png;base64," + "A" * 12)]), "image_bytes_exceeded")
        url = image_url()
        size = len(base64.b64decode(url.split(",")[1]))
        with patch.object(policy, "MAX_IMAGE_BYTES", size):
            admit(payload([image_part(url)]))
        with patch.object(policy, "MAX_IMAGE_BYTES", size - 1):
            self.assert_rejected(payload([image_part(url)]), "image_bytes_exceeded")

    def test_count_history_and_aggregate_exact_boundary(self):
        part = image_part()
        admit(payload([part] * 32))
        history = payload([part] * 16)
        history["messages"].append({"role": "tool", "content": [part] * 17})
        with patch.object(policy.Image, "open", side_effect=AssertionError("decoder")):
            self.assert_rejected(history, "too_many_images")
        budget = policy.ImageBudget(2048, 1024, 2048, 1024, 2048)
        with patch.object(policy, "image_budget", return_value=budget):
            _, budgets = admit(payload([part] * 16))
            self.assertEqual(sum(b.visual_tokens for b in budgets), 32768)
            self.assert_rejected(payload([part] * 17), "visual_token_limit_exceeded")
        # Sixteen square 4096 sources yield 32400, not 32768.
        budget = policy.ImageBudget(4096, 4096, 1440, 1440, 2025)
        with patch.object(policy, "image_budget", return_value=budget):
            _, budgets = admit(payload([part] * 16))
            self.assertEqual(sum(b.visual_tokens for b in budgets), 32400)
            self.assert_rejected(payload([part] * 17), "visual_token_limit_exceeded")

    def test_modalities_rejected_before_any_image_decode(self):
        for kind in ("video_url", "input_audio", "audio_url", "image_embeds", "audio_embeds",
                     "prompt_embeds", "input_image", "image_pil", "vision_chunk", None):
            with self.subTest(kind=kind), patch.object(policy.Image, "open", side_effect=AssertionError("decoder")):
                self.assert_rejected(payload([image_part(), {"type": kind, "video_url": "http://example.invalid/v"}]),
                                     "unsupported_modality")

    def test_processor_media_and_geometry_overrides(self):
        for field in policy.FORBIDDEN_FIELDS:
            with self.subTest(field=field):
                self.assert_rejected(payload(**{field: {}}), "unsupported_override")
        for extra in ({"do_resize": False}, {"max_pixels": 99999999}, {"uuid": "cached"}):
            self.assert_rejected(payload([image_part(**extra)]), "unsupported_override")
        self.assert_rejected(payload([{"type": "text", "text": "safe", "image_url": {"url": "file:///x"}}]), "invalid_request")
        rewritten, _ = admit(payload([image_part()]))
        result = json.loads(rewritten)
        self.assertEqual(result["mm_processor_kwargs"], policy.fixed_processor_kwargs())
        self.assertEqual(result["mm_processor_kwargs"]["min_pixels"], 65536)
        self.assertEqual(result["mm_processor_kwargs"]["size"]["longest_edge"], 2097152)
        self.assertTrue(result["mm_processor_kwargs"]["do_resize"])

    def test_generation_limit_defaults_aliases_and_types(self):
        for value in ({}, {"max_tokens": None}, {"max_completion_tokens": None}):
            rewritten, _ = admit(payload(**value))
            self.assertEqual(json.loads(rewritten)["max_completion_tokens"], 32768)
        for field in ("max_tokens", "max_completion_tokens"):
            for value in (0, -1, 32769, True, 32.0, "32"):
                self.assert_rejected(payload(**{field: value}), "generation_limit_exceeded")
            rewritten, _ = admit(payload(**{field: 17}))
            self.assertEqual(json.loads(rewritten)["max_completion_tokens"], 17)
        self.assert_rejected(payload(max_tokens=32769, max_completion_tokens=1), "generation_limit_exceeded")
        self.assert_rejected(payload(min_tokens=32769), "generation_limit_exceeded")
        self.assert_rejected(payload(min_tokens=10, max_tokens=1), "invalid_request")

    def test_json_rejections(self):
        for raw in (b"[]", b"null", b"{", b'{"messages":[],"messages":[]}',
                    b'{"messages":[],"x":NaN}', b"\xff", b"[" * 2000):
            with self.subTest(raw=raw[:40]), self.assertRaises(policy.PolicyError):
                policy.admit(raw, resize=cpu_resize)
        for value in ({}, {"messages": {}}, {"messages": [1]}, payload([42]), payload({"text": "bad"})):
            with self.assertRaises(policy.PolicyError):
                admit(value)

    def test_aspect_and_native_provider_fail_closed(self):
        self.assert_rejected(payload([image_part(image_url((201, 1)))]), "invalid_image_dimensions")
        with self.assertRaises(policy.PolicyError) as caught:
            policy.image_budget(image_url(), "image", lambda **kw: (2048, 2048))
        self.assertEqual(caught.exception.code, "invalid_processor_geometry")
        calls = []
        def provider(**kwargs):
            calls.append(kwargs)
            return cpu_resize(**kwargs)
        policy.image_budget(image_url(), "image", provider)
        self.assertEqual(calls, [{"height": 256, "width": 256, "factor": 32,
                                 "min_pixels": 65536, "max_pixels": 2097152}])


class ASGITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        original = policy.admit
        self.admission_patch = patch.object(policy, "admit", side_effect=lambda body, **kwargs: original(body, resize=cpu_resize, **kwargs))
        self.admission_patch.start()
        self.addCleanup(self.admission_patch.stop)
        self.calls = []
        async def app(scope, receive, send):
            event = await receive()
            self.calls.append((scope, event))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})
        self.middleware = policy.ImagePolicyMiddleware(app)

    async def request(self, value=None, *, path="/v1/chat/completions", method="POST",
                      chunks=None, headers=None, receive=None, sent=None, root=""):
        events = ([{"type": "http.request", "body": json.dumps(value or payload()).encode()}]
                  if chunks is None else chunks)
        if receive is None:
            async def receive():
                return events.pop(0) if events else {"type": "http.disconnect"}
        if sent is None:
            sent = []
        async def send(event):
            sent.append(event)
        if headers is None:
            headers = [(b"content-type", b"application/json")]
            if chunks is None:
                headers.append((b"content-length", str(sum(len(e.get("body", b"")) for e in events)).encode()))
        scope = {"type": "http", "method": method, "path": path, "root_path": root,
                 "headers": headers}
        await self.middleware(scope, receive, send)
        return sent

    async def wait_until(self, predicate):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.001)
        await asyncio.wait_for(poll(), 2)

    async def test_chunked_replay_state_headers_and_stream_body(self):
        raw = json.dumps(payload([image_part()], stream=True)).encode()
        events = [{"type": "http.request", "body": raw[:13], "more_body": True},
                  {"type": "http.request", "body": raw[13:]}]
        sent = await self.request(chunks=events, headers=[(b"content-type", b"application/json"),
                                                        (b"transfer-encoding", b"chunked")])
        self.assertEqual(sent[0]["status"], 200)
        scope, event = self.calls[0]
        self.assertEqual(scope["state"]["qwen38_visual_tokens"], 64)
        self.assertEqual(int(dict(scope["headers"])[b"content-length"]), len(event["body"]))
        self.assertNotIn(b"transfer-encoding", dict(scope["headers"]))
        self.assertTrue(json.loads(event["body"])["stream"])

    async def test_body_limit_without_or_with_false_length_and_openai_errors(self):
        with patch.object(policy, "MAX_BODY_BYTES", 8):
            for headers in ([(b"content-type", b"application/json")],
                            [(b"content-type", b"application/json"), (b"content-length", b"1")]):
                sent = await self.request(chunks=[{"type": "http.request", "body": b"12345", "more_body": True},
                                                 {"type": "http.request", "body": b"6789"}], headers=headers)
                self.assertEqual(sent[0]["status"], 413)
                self.assertEqual(json.loads(sent[1]["body"])["error"]["code"], "payload_too_large")
        self.assertFalse(self.calls)
        raw = json.dumps(payload()).encode()
        with patch.object(policy, "MAX_BODY_BYTES", len(raw)):
            # Injected defaults must fit the normalized-body cap too.
            self.assertEqual((await self.request())[0]["status"], 413)
        normalized, _ = policy.admit(raw)
        with patch.object(policy, "MAX_BODY_BYTES", max(len(raw), len(normalized))):
            self.assertEqual((await self.request())[0]["status"], 200)

    async def test_content_headers_rejected_before_receive(self):
        async def no_receive():
            raise AssertionError("read rejected upload")
        for additions, status in [([(b"content-length", b"999999999")], 413),
                                  ([(b"content-length", b"-1")], 400),
                                  ([(b"content-encoding", b"gzip")], 415),
                                  ([(b"content-type", b"multipart/form-data")], 400)]:
            sent = await self.request(headers=[(b"content-type", b"application/json")] + additions, receive=no_receive)
            self.assertEqual(sent[0]["status"], status)
        sent = await self.request(headers=[(b"content-type", b"multipart/form-data")], receive=no_receive)
        self.assertEqual(sent[0]["status"], 415)

    async def test_alternate_routes_never_receive_or_forward(self):
        async def no_receive():
            raise AssertionError("alternate route read")
        for route in ("/tokenize", "/detokenize", "/v1/embeddings", "/v1/completions",
                      "/v1/responses", "/invocations", "/v1/chat/completions/batch",
                      "/pooling", "/score", "/v1/audio/transcriptions", "/render"):
            sent = await self.request(path=route, receive=no_receive)
            self.assertEqual(sent[0]["status"], 404)
        self.assertFalse(self.calls)
        self.assertEqual((await self.request(path="/prefix/v1/chat/completions/", root="/prefix"))[0]["status"], 200)

    async def test_health_bypasses_full_gate_and_unread_queue_bounded(self):
        await self.middleware._gate.acquire()
        queued = [asyncio.create_task(self.request()) for _ in range(16)]
        try:
            await self.wait_until(lambda: self.middleware._waiting == 16)
            self.assertEqual((await self.request())[0]["status"], 429)
            sent = await asyncio.wait_for(self.request(path="/health", method="GET"), 0.5)
            self.assertEqual(sent[0]["status"], 200)
        finally:
            for task in queued:
                task.cancel()
            await asyncio.gather(*queued, return_exceptions=True)
            self.middleware._gate.release()
        self.assertEqual(self.middleware._waiting, 0)

    async def test_queue_upload_timeouts_and_disconnect_cleanup(self):
        await self.middleware._gate.acquire()
        try:
            with patch.object(policy, "QUEUE_TIMEOUT", 0.01):
                sent = await self.request()
                self.assertEqual(sent[0]["status"], 429)
        finally:
            self.middleware._gate.release()
        async def slow_receive():
            await asyncio.Event().wait()
        with patch.object(policy, "UPLOAD_TIMEOUT", 0.01):
            self.assertEqual((await self.request(receive=slow_receive))[0]["status"], 408)
        self.assertFalse(self.middleware._gate.locked())
        sent = await self.request(chunks=[{"type": "http.disconnect"}])
        self.assertEqual(sent, [])
        self.assertFalse(self.middleware._gate.locked())

    async def test_text_decode_not_serialized_and_busy_image_rejected(self):
        finish_image = asyncio.Event()
        started_images = []
        async def app(scope, receive, send):
            await receive()
            if scope.get("state", {}).get("qwen38_visual_tokens"):
                started_images.append(scope)
                await finish_image.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
        self.middleware.app = app
        first = asyncio.create_task(self.request(payload([image_part()])))
        await self.wait_until(lambda: len(started_images) == 1)
        second = asyncio.create_task(self.request(payload([image_part()])))
        try:
            self.assertEqual((await second)[0]["status"], 429)
            texts = await asyncio.wait_for(asyncio.gather(*(self.request() for _ in range(16))), 2)
            self.assertTrue(all(result[0]["status"] == 200 for result in texts))
            self.assertEqual(len(started_images), 1)
        finally:
            finish_image.set()
            await asyncio.gather(first, second)
        self.assertEqual(len(started_images), 1)
        self.assertFalse(self.middleware._image_gate.locked())
        self.assertEqual(self.middleware._buffered_body_bytes, 0)

    async def test_stream_holds_gate_after_headers_preserves_sse_and_disconnect(self):
        finish_stream = asyncio.Event()
        first_started = asyncio.Event()
        response_piece = {"type": "http.response.body", "body": b"data: first\n\n", "more_body": True}
        count = 0
        async def app(scope, receive, send):
            nonlocal count
            await receive()
            count += 1
            index = count
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send(response_piece)
            if index == 1:
                first_started.set()
                await finish_stream.wait()
            self.assertEqual((await receive())["type"], "http.disconnect")
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n"})
        self.middleware.app = app
        sent = []
        first = asyncio.create_task(self.request(payload([image_part()], stream=True), sent=sent))
        await first_started.wait()
        try:
            self.assertIs(sent[1], response_piece)
            self.assertTrue(self.middleware._image_gate.locked())
            second = await asyncio.wait_for(self.request(payload([image_part()], stream=True)), 2)
            self.assertEqual(second[0]["status"], 429)
        finally:
            finish_stream.set()
            await first

    async def test_busy_image_rejection_releases_body_reservation(self):
        await self.middleware._image_gate.acquire()
        try:
            self.assertEqual((await self.request(payload([image_part()])))[0]["status"], 429)
            self.assertEqual(self.middleware._image_waiting, 0)
            self.assertFalse(self.middleware._gate.locked())
            self.assertEqual(self.middleware._buffered_body_bytes, 0)
        finally:
            self.middleware._image_waiting = 0
            self.middleware._image_gate.release()

    async def test_global_body_budget_rejects_without_reading(self):
        self.middleware._buffered_body_bytes = policy.MAX_BUFFERED_BODY_BYTES
        async def forbidden_receive():
            self.fail("body should not be read when global budget is exhausted")
        try:
            result = await self.request(receive=forbidden_receive,
                                        headers=[(b"content-type", b"application/json"),
                                                 (b"content-length", b"100")])
            self.assertEqual(result[0]["status"], 429)
            self.assertEqual(self.middleware._buffered_body_bytes, policy.MAX_BUFFERED_BODY_BYTES)
        finally:
            self.middleware._buffered_body_bytes = 0

    async def test_cancel_cpu_worker_keeps_gate_until_thread_finished(self):
        entered, finish = threading.Event(), threading.Event()
        def work(body, **kwargs):
            entered.set()
            finish.wait(2)
            return b"{}", []
        with patch.object(policy, "admit", side_effect=work):
            task = asyncio.create_task(self.request())
            await self.wait_until(entered.is_set)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(self.middleware._gate.locked())
            finish.set()
            await self.wait_until(lambda: not self.middleware._gate.locked())

    async def test_downstream_exception_releases_gate_without_second_response(self):
        async def app(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            raise RuntimeError("downstream")
        self.middleware.app = app
        sent = []
        with self.assertRaisesRegex(RuntimeError, "downstream"):
            await self.request(payload([image_part()]), sent=sent)
        self.assertEqual(len(sent), 1)
        self.assertFalse(self.middleware._gate.locked())
        self.assertFalse(self.middleware._image_gate.locked())


if __name__ == "__main__":
    if "--native-smart-resize-stdin" in sys.argv:
        sys.argv.remove("--native-smart-resize-stdin")
        tree = ast.parse(sys.stdin.read())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "smart_resize"]
        if len(functions) != 1:
            raise SystemExit("Expected exactly one pinned smart_resize function")
        namespace = {"math": math}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned-smart-resize", "exec"), namespace)
        cpu_resize = namespace["smart_resize"]
        print("Testing with exact pinned native smart_resize AST", flush=True)
    unittest.main()
