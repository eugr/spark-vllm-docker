"""No Torch/vLLM/GPU required: python3 -B tests/test_qwen38_image_encoder.py.

Optional read-only qualification: pipe the complete pristine encoder_runner.py
to this command with --native-source-stdin. Verifies its real SHA and executes
only its original/patched method AST with fake Torch and model dependencies.
No test downloads, installs, patches runtime files, or starts remote work.
"""

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch
import weakref


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


encoder = load("qwen38_image_encoder", ROOT / "docker/qwen38-r12/modules/image_encoder.py")
patcher = load("patch_sequential_image_encoder", ROOT / "docker/qwen38-r12/patches/patch_sequential_image_encoder.py")
NATIVE_SOURCE = None
MIB = 1024**2
GIB = 1024**3


class DeviceInput:
    def __init__(self, items):
        self.items = items


class Embedding:
    ndim = 2
    device = "cuda:0"

    def __init__(self, item):
        self.value = item["value"]
        self.metadata = item["metadata"]


class TensorLike:
    ndim = 3

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)


class FakeCUDA:
    def __init__(self, backend):
        self.backend = backend
        self.capture = False
        self.current = None
        self.sync_count = self.empty_count = 0
        self.event_sync_count = 0
        self.timeline = []
        self.cached = 512 * MIB
        self.cache_after_forward = None
        self.inactive_split = self.pending = 0
        self.missing_stats = False
        self.on_sync = lambda: None

    @contextmanager
    def device(self, device):
        previous = self.current
        self.current = device
        try:
            yield
        finally:
            self.current = previous

    def is_current_stream_capturing(self):
        assert self.current is self.backend.device
        return self.capture

    def memory_allocated(self, device):
        assert device is self.backend.device
        return GIB + len(self.backend.live_outputs) * 10 * MIB + len(self.backend.live_inputs) * 24 * MIB

    def memory_reserved(self, device):
        return self.memory_allocated(device) + self.cached

    def memory_stats(self, device):
        if self.missing_stats:
            return {}
        return {
            "reserved_bytes.all.current": self.memory_reserved(device),
            "active_bytes.all.current": self.memory_allocated(device) + self.pending,
            "inactive_split_bytes.all.current": self.inactive_split,
        }

    def synchronize(self, device):
        assert device is self.current
        assert not self.capture
        assert not self.backend.live_inputs
        self.sync_count += 1
        self.timeline.append("device_sync")
        self.on_sync()

    def empty_cache(self):
        assert self.current is self.backend.device
        assert not self.capture
        assert not self.backend.live_inputs
        # Every prior GPU output must still be live at EVERY pressure cleanup.
        assert all(ref() is not None and ref().device == "cuda:0"
                   for ref in self.backend.output_refs)
        self.empty_count += 1
        self.timeline.append("empty_cache")
        self.cached = 0

    def Event(self, *, enable_timing):
        assert enable_timing is True
        cuda = self

        class Event:
            def record(self):
                assert cuda.current is cuda.backend.device
                assert not cuda.capture
                cuda.timeline.append("event_record")

            def synchronize(self):
                assert not cuda.capture
                cuda.timeline.append("event_sync")
                cuda.event_sync_count += 1

            def elapsed_time(self, other):
                assert cuda.timeline[-1] == "event_sync"
                assert other is not self
                return 12.5

        return Event()


class Backend:
    def __init__(self, output_type=tuple):
        self.device = SimpleNamespace(type="cuda", index=0)
        self.live_inputs = weakref.WeakSet()
        self.live_outputs = weakref.WeakSet()
        self.output_refs = []
        self.transfers = []
        self.live_before_transfer = []
        self.max_live = self.closed = 0
        self.failure = None
        self.output_type = output_type
        self.inference = False
        self.cuda = FakeCUDA(self)
        self.torch = SimpleNamespace(cuda=self.cuda, inference_mode=self.inference_mode,
                                     Tensor=TensorLike)
        self.runner = SimpleNamespace(device=self.device, model=self,
                                      cudagraph_manager=None, is_realtime=False)

    def inference_mode(self):
        def decorate(function):
            def wrapped(*args, **kwargs):
                previous = self.inference
                self.inference = True
                try:
                    return function(*args, **kwargs)
                finally:
                    self.inference = previous
            return wrapped
        return decorate

    def batch(self, items, *, device, pin_memory):
        assert device is self.device and pin_memory is True
        self.live_before_transfer.append(len(self.live_inputs))
        self.cuda.timeline.append("transfer")
        self.transfers.append(items)
        # Nested generator frames mimic native group_and_batch_mm_items/kwargs.
        def inner():
            payload = DeviceInput(items)
            self.live_inputs.add(payload)
            self.max_live = max(self.max_live, len(self.live_inputs))
            if self.failure == ("transfer", items[0][1]["value"]):
                raise RuntimeError("transfer failed")
            yield {"pixels": payload, "position_metadata": [item[1]["metadata"] for item in items]}
        try:
            for kwargs in inner():
                yield items[0][0], len(items), kwargs
        finally:
            self.closed += 1

    def embed_multimodal(self, *, pixels, position_metadata):
        assert self.inference
        self.cuda.timeline.append("forward")
        # Kept in a real failed backend traceback to test debugging preservation.
        diagnostic = "backend locals preserved"
        if self.failure == ("model", pixels.items[0][1]["value"]):
            raise RuntimeError(diagnostic)
        outputs = []
        for (_, item), metadata in zip(pixels.items, position_metadata):
            assert item["metadata"] is metadata
            output = Embedding(item)
            self.live_outputs.add(output)
            self.output_refs.append(weakref.ref(output))
            outputs.append(output)
        if self.cuda.cache_after_forward is not None:
            self.cuda.cached = self.cuda.cache_after_forward
        if self.failure == ("sanity", pixels.items[0][1]["value"]):
            return []
        return self.output_type(outputs)

    def sanity(self, outputs, *, expected_num_items):
        assert isinstance(outputs, (list, tuple, TensorLike))
        assert len(outputs) == expected_num_items
        assert all(output.ndim == 2 for output in outputs)

    def run(self, items, *, enabled=True, source=None):
        # Exercise the native method's decorator and actual inserted branch,
        # not just the helper in isolation.
        if source is None:
            source = "class EncoderRunner:\n" + patcher.NEW_METHOD
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "EncoderRunner")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "execute_mm_encoder")
        cls.body = [method]
        cls.bases = cls.keywords = cls.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
        namespace = {"torch": self.torch, "MultiModalKwargsItem": dict,
                     "group_and_batch_mm_kwargs": self.batch, "PIN_MEMORY": True,
                     "sanity_check_mm_encoder_outputs": self.sanity}
        with patch.dict(sys.modules, {"torch": self.torch}), patch.dict(
            "os.environ", {"QWEN38_SEQUENTIAL_IMAGE_ENCODER": "1" if enabled else "0"}
        ):
            exec(compile(module, "native_encoder_method", "exec"), namespace)
            return namespace["EncoderRunner"].execute_mm_encoder(self.runner, items)


def items(count):
    return [("image", {"value": i, "metadata": object()}) for i in range(count)]


class EncoderTests(unittest.TestCase):
    def setUp(self):
        self.host = patch.object(encoder, "host_available_bytes", return_value=16 * GIB)
        self.host_mock = self.host.start()
        self.addCleanup(self.host.stop)
        self.env = patch.dict("os.environ", {"QWEN38_IMAGE_ENCODER_PRESSURE_CLEANUP": "1",
                                             "QWEN38_IMAGE_ENCODER_TIMING": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.logs = patch.object(encoder.logger, "info")
        self.log_mock = self.logs.start()
        self.addCleanup(self.logs.stop)

    def test_zero_one_many_output_order_identity_native_protocol_and_batch_lifetime(self):
        for count in (0, 1, 16):
            for kind in (list, tuple, TensorLike):
                with self.subTest(count=count, kind=kind):
                    data = items(count)
                    backend = Backend(kind)
                    outputs = backend.run(data)
                    baseline = Backend(kind).run(data, enabled=False) if data else []
                    self.assertEqual([o.value for o in outputs], [o.value for o in baseline])
                    hashes = [f"identifier-{i // 2}" for i in range(count)]
                    self.assertEqual({k: o.value for k, o in zip(hashes, outputs)},
                                     {k: o.value for k, o in zip(hashes, baseline)})
                    self.assertEqual(backend.live_before_transfer, [0] * count)
                    self.assertEqual(backend.max_live, min(1, count))
                    self.assertEqual(len(backend.live_inputs), 0)
                    self.assertEqual(backend.closed, count)
                    self.assertEqual(backend.cuda.sync_count, 0)
                    self.assertEqual(backend.cuda.empty_count, 0)
                    self.assertEqual(backend.cuda.event_sync_count, 0)
                    self.assertEqual(len(backend.live_outputs), count)
                    for i, output in enumerate(outputs):
                        self.assertIs(output, backend.output_refs[i]())
                        self.assertIs(output.metadata, data[i][1]["metadata"])
                        self.assertIs(backend.transfers[i][0], data[i])
                        self.assertEqual(len(backend.transfers[i]), 1)

    def test_disabled_path_keeps_native_batching(self):
        backend = Backend()
        data = items(4)
        outputs = backend.run(data, enabled=False)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(backend.transfers), 1)
        self.assertIs(backend.transfers[0], data)
        self.log_mock.assert_not_called()

    def test_opt_in_timing_completes_each_forward_after_transfer_before_next_image(self):
        backend = Backend()
        with patch.dict("os.environ", {"QWEN38_IMAGE_ENCODER_TIMING": "1"}):
            outputs = backend.run(items(4))
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(backend.live_outputs), 4)
        self.assertEqual(backend.cuda.event_sync_count, 4)
        self.assertEqual(backend.cuda.sync_count, 0)  # No device-wide timing sync.
        self.assertEqual(backend.cuda.timeline,
                         ["transfer", "event_record", "forward", "event_record", "event_sync"] * 4)
        self.assertEqual(backend.live_before_transfer, [0] * 4)
        logs = [json.loads(call.args[1]) for call in self.log_mock.call_args_list]
        self.assertEqual(len(logs), 4)
        for row in logs:
            self.assertEqual(row["gpu_forward_ms"], 12.5)
            self.assertGreaterEqual(row["completed_elapsed_s"], 0)
            self.assertEqual(row["completed_elapsed_s"], row["encoder_host_elapsed_s"])

    def test_graph_realtime_capture_and_nonimage_fail_before_transfer(self):
        for mode in ("manager", "realtime", "capture", "video", "audio", "cpu"):
            with self.subTest(mode=mode):
                backend = Backend()
                data = items(2)
                if mode == "manager":
                    backend.runner.cudagraph_manager = object()
                elif mode == "realtime":
                    backend.runner.is_realtime = True
                elif mode == "capture":
                    backend.cuda.capture = True
                elif mode == "cpu":
                    backend.device.type = "cpu"
                else:
                    data.append((mode, {}))
                with self.assertRaises((RuntimeError, ValueError)):
                    backend.run(data)
                self.assertEqual(backend.transfers, [])
                self.assertEqual(backend.cuda.sync_count, 0)
                self.assertEqual(backend.cuda.empty_count, 0)

    def test_pressure_cleanup_retains_all_gpu_outputs_and_logs_snapshots(self):
        self.host_mock.return_value = 7 * GIB
        backend = Backend()
        backend.cuda.cache_after_forward = 512 * MIB
        outputs = backend.run(items(4))
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(backend.live_outputs), 4)
        self.assertEqual(backend.cuda.sync_count, 5)
        self.assertEqual(backend.cuda.empty_count, 5)
        logs = [json.loads(call.args[1]) for call in self.log_mock.call_args_list]
        encoded = [row for row in logs if row["event"] == "image_encoded"]
        cleanups = [row for row in logs if row["event"] == "pressure_cleanup"]
        self.assertEqual([row["phase"] for row in cleanups],
                         ["before_encoder"] + ["after_image"] * 4)
        self.assertIsNone(cleanups[0]["image_index"])
        self.assertEqual([row["image_index"] for row in cleanups[1:]], list(range(4)))
        self.assertEqual(backend.cuda.timeline[:3], ["device_sync", "empty_cache", "transfer"])
        self.assertEqual(len(encoded), 4)
        for cleanup in cleanups:
            self.assertEqual(cleanup["before"]["allocated_bytes"], cleanup["after"]["allocated_bytes"])
            self.assertGreater(cleanup["before"]["reserved_bytes"], cleanup["after"]["reserved_bytes"])
        for row in encoded:
            self.assertGreaterEqual(row["encoder_host_elapsed_s"], 0)
            self.assertIsNone(row["gpu_forward_ms"])
            self.assertIsNone(row["completed_elapsed_s"])
            self.assertEqual(set(row["before"]), {"allocated_bytes", "reserved_bytes", "host_available_bytes"})
            self.assertNotIn("metadata", json.dumps(row))

    def test_pressure_thresholds_and_nonreleasable_storage(self):
        cases = [(8 * GIB, 512 * MIB, 0, 0, False, False),
                 (7 * GIB, 256 * MIB - 1, 0, 0, False, False),
                 (7 * GIB, 256 * MIB, 0, 0, False, True),
                 (7 * GIB, 512 * MIB, 300 * MIB, 0, False, False),
                 (7 * GIB, 512 * MIB, 0, 300 * MIB, False, False),
                 (None, 512 * MIB, 0, 0, False, False),
                 (7 * GIB, 512 * MIB, 0, 0, True, False)]
        for host, cached, split, pending, missing, expected in cases:
            with self.subTest(case=(host, cached, split, pending, missing)):
                self.host_mock.return_value = host
                backend = Backend()
                backend.cuda.cached = cached
                backend.cuda.inactive_split = split
                backend.cuda.pending = pending
                backend.cuda.missing_stats = missing
                backend.run(items(1))
                self.assertEqual(backend.cuda.empty_count, int(expected))
                self.assertEqual(backend.cuda.sync_count, int(expected))

    def test_pressure_rechecked_after_sync(self):
        self.host_mock.return_value = 7 * GIB
        backend = Backend()
        backend.cuda.on_sync = lambda: setattr(self.host_mock, "return_value", 9 * GIB)
        backend.run(items(1))
        self.assertEqual(backend.cuda.sync_count, 1)
        self.assertEqual(backend.cuda.empty_count, 0)

    def test_cleanup_disabled_and_empty_input_never_cleaned(self):
        self.host_mock.return_value = GIB
        backend = Backend()
        backend.run([])
        with patch.dict("os.environ", {"QWEN38_IMAGE_ENCODER_PRESSURE_CLEANUP": "0"}):
            backend.run(items(1))
        self.assertEqual(backend.cuda.sync_count, 0)
        self.assertEqual(backend.cuda.empty_count, 0)

    def test_failure_closes_generators_preserves_backend_traceback_and_skips_cleanup(self):
        for stage in ("transfer", "model", "sanity"):
            backend = Backend()
            backend.failure = (stage, 1)
            retained = None
            try:
                backend.run(items(4))
            except (RuntimeError, AssertionError) as exc:
                retained = exc
            self.assertIsNotNone(retained)
            self.assertEqual(len(backend.transfers), 2)
            self.assertEqual(backend.closed, 2)
            self.assertEqual(backend.cuda.sync_count, 0)
            self.assertEqual(backend.cuda.empty_count, 0)
            self.assertIsNone(backend.output_refs[0]())  # Partial result released.
            traceback = retained.__traceback__
            frames = []
            while traceback:
                # Do not materialize THIS active test's f_locals: that snapshot
                # would itself retain `retained` after deleting the variable.
                if traceback.tb_frame.f_code.co_name == "embed_multimodal":
                    frames.append(traceback.tb_frame.f_locals)
                traceback = traceback.tb_next
            if stage == "model":
                self.assertTrue(any(frame.get("diagnostic") == "backend locals preserved" for frame in frames))
                self.assertEqual(len(backend.live_inputs), 1)  # Deliberate debug retention.
            del frames, traceback, retained
            self.assertEqual(len(backend.live_inputs), 0)

    def test_bad_batcher_fails_clearly(self):
        for groups in ([], [("image", 2, {})], [("video", 1, {})]):
            backend = Backend()
            backend.batch = lambda *args, **kwargs: (group for group in groups)
            with self.assertRaisesRegex(RuntimeError, "Pinned batcher"):
                backend.run(items(1))


class PatchTests(unittest.TestCase):
    def test_exact_anchor_hash_idempotence_and_drift_rejection(self):
        fixture = "class EncoderRunner:\n" + patcher.OLD_METHOD
        digest = hashlib.sha256(fixture.encode()).hexdigest()
        with patch.object(patcher, "PINNED_SHA256", digest):
            result = patcher.patch_source(fixture)
            self.assertEqual(patcher.patch_source(result), result)
            self.assertEqual(result.replace(patcher.INSERTION, ""), fixture)
            for drift in (fixture + "\n# drift\n", result + "\n# drift\n",
                          result.replace("if sequential_enabled():", "if True:")):
                with self.assertRaises(ValueError):
                    patcher.patch_source(drift)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            patcher.patch_source(fixture)

    def test_host_available_parser(self):
        for content, expected in [("MemTotal: 100 kB\nMemAvailable: 42 kB\n", 42 * 1024),
                                  ("MemAvailable: broken kB\n", None),
                                  ("MemAvailable: -1 kB\n", None),
                                  ("MemAvailable: 42 bytes\n", None), ("", None)]:
            with patch("builtins.open", mock_open(read_data=content)):
                self.assertEqual(encoder.host_available_bytes(), expected)
        with patch("builtins.open", side_effect=OSError):
            self.assertIsNone(encoder.host_available_bytes())

    def test_real_pinned_source_when_supplied(self):
        if NATIVE_SOURCE is None:
            self.skipTest("pipe pristine encoder_runner.py with --native-source-stdin")
        result = patcher.patch_source(NATIVE_SOURCE)
        self.assertEqual(patcher.patch_source(result), result)
        self.assertEqual(result.replace(patcher.INSERTION, ""), NATIVE_SOURCE)
        data = items(4)
        with patch.object(encoder, "host_available_bytes", return_value=16 * GIB):
            backend = Backend()
            output = backend.run(data, source=result)
            native = Backend().run(data, enabled=False, source=NATIVE_SOURCE)
        self.assertEqual([o.value for o in output], [o.value for o in native])
        self.assertEqual(backend.live_before_transfer, [0] * 4)


if __name__ == "__main__":
    if "--native-source-stdin" in sys.argv:
        sys.argv.remove("--native-source-stdin")
        NATIVE_SOURCE = sys.stdin.read()
    unittest.main()
