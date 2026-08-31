"""Native, opt-in one-image EncoderRunner execution for the pinned runtime.

Install as qwen38_image_encoder.py and apply patch_sequential_image_encoder.py.
QWEN38_SEQUENTIAL_IMAGE_ENCODER=1 selects this path; otherwise native execution
is unchanged. QWEN38_IMAGE_ENCODER_PRESSURE_CLEANUP=0 disables its pressure-only
cleanup (enabled by default within the opt-in path): before the first transfer
and after each image's input release. No prefill/decode hook or KV-budget change.
QWEN38_IMAGE_ENCODER_TIMING=1 enables completed per-image measurements: CUDA
events enclose only embed_multimodal, after input transfer; synchronizing the end
event also completes that image's work before the next image can be transferred.

One original CPU MultiModalKwargsItem goes through native field reduction and
device transfer at a time. No tensor copies beyond native batching, output CPU
offload, or cache writes are introduced. All GPU embeddings remain referenced
until return, in original order, for the caller's zip(mm_hashes, outputs).
The existing per-image token bound, request admission, and 2 GiB emergency
reserve remain external and unchanged. This is not a whole-server memory cap.

With timing disabled (default), per-image elapsed time is host/enqueue time,
NOT completed GPU execution time. No synchronization on the normal path.
With timing enabled, gpu_forward_ms is CUDA-event time and completed_elapsed_s
is wall time including transfer, forward completion, checking and input release.
Pressure cleanup synchronizes the target
device outside capture, rechecks pressure, and only frees unused CUDA cache.
Its releasable estimate excludes active and inactive-split allocator blocks;
it is still an estimate, not a promise of OS-visible recovery. Missing allocator
statistics or host MemAvailable disable cleanup rather than guessing.

Backend exception tracebacks are preserved: a retained exception may retain the
failing input/output. Only this helper's batch references/partial output list
are released; no clear_frames, cancellation claim, or CUDA recovery on failure.
"""

from contextlib import closing
import json
import logging
import os
import time


HOST_PRESSURE_BYTES = 8 * 1024**3
MIN_RELEASABLE_BYTES = 256 * 1024**2
logger = logging.getLogger("vllm.qwen38_image_encoder")


def sequential_enabled():
    return os.environ.get("QWEN38_SEQUENTIAL_IMAGE_ENCODER", "0") == "1"


def host_available_bytes():
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                fields = line.split()
                if fields and fields[0] == "MemAvailable:":
                    if len(fields) != 3 or fields[2] != "kB":
                        return None
                    value = int(fields[1]) * 1024
                    return value if value >= 0 else None
    except (OSError, ValueError):
        pass
    return None


def _snapshot(cuda, device):
    return {
        "allocated_bytes": cuda.memory_allocated(device),
        "reserved_bytes": cuda.memory_reserved(device),
        "host_available_bytes": host_available_bytes(),
    }


def _releasable_estimate(cuda, device):
    stats = cuda.memory_stats(device)
    keys = ("reserved_bytes.all.current", "active_bytes.all.current",
            "inactive_split_bytes.all.current")
    values = [stats.get(key) for key in keys]
    if any(not isinstance(value, int) or value < 0 for value in values):
        return 0
    reserved, active, inactive_split = values
    return max(0, reserved - active - inactive_split)


def _under_pressure(snapshot):
    available = snapshot["host_available_bytes"]
    return available is not None and available < HOST_PRESSURE_BYTES


def _cleanup_at_boundary(cuda, device, before, *, phase, image_index=None):
    # Only before any transfer, or after successful encoding/input release.
    if os.environ.get("QWEN38_IMAGE_ENCODER_PRESSURE_CLEANUP", "1") != "1":
        return False
    if not _under_pressure(before) or cuda.is_current_stream_capturing():
        return False
    estimate = _releasable_estimate(cuda, device)
    if estimate < MIN_RELEASABLE_BYTES:
        return False
    start = time.perf_counter()
    cuda.synchronize(device)
    before_release = _snapshot(cuda, device)
    estimate = _releasable_estimate(cuda, device)
    released = False
    if (_under_pressure(before_release) and estimate >= MIN_RELEASABLE_BYTES
            and not cuda.is_current_stream_capturing()):
        cuda.empty_cache()  # Current device selected by execute_mm_encoder below.
        released = True
    after = _snapshot(cuda, device)
    logger.info("qwen38_image_encoder %s", json.dumps({
        "event": "pressure_cleanup", "phase": phase, "image_index": image_index,
        "before": before, "before_release": before_release, "after": after,
        "releasable_estimate_bytes": estimate, "empty_cache_called": released,
        "cleanup_elapsed_s": time.perf_counter() - start,
    }, sort_keys=True))
    return released


def execute_mm_encoder(runner, mm_kwargs, *, group_and_batch_mm_kwargs,
                       sanity_check_mm_encoder_outputs, pin_memory):
    """Called inside native @torch.inference_mode(); no model protocol changes."""
    if not mm_kwargs:
        return []
    if runner.cudagraph_manager is not None:
        raise RuntimeError("Sequential image encoder requires encoder CUDA graphs disabled")
    if runner.is_realtime:
        raise RuntimeError("Sequential image encoder does not support realtime models")
    for modality, _ in mm_kwargs:
        if modality != "image":
            raise ValueError(f"Sequential image encoder is image-only, got {modality!r}")

    import torch

    if runner.device.type != "cuda":
        raise RuntimeError("Sequential image encoder requires a CUDA device")
    outputs = []
    groups = mm_kwargs_batch = batch_outputs = None
    forward_start = forward_end = None
    timing_enabled = os.environ.get("QWEN38_IMAGE_ENCODER_TIMING", "0") == "1"
    with torch.cuda.device(runner.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Sequential image encoder cannot execute during CUDA capture")
        try:
            # Old LLM-prefill cache can otherwise crowd out the first vision
            # allocation. This boundary is reached only for image encoder work.
            _cleanup_at_boundary(
                torch.cuda, runner.device, _snapshot(torch.cuda, runner.device),
                phase="before_encoder",
            )
            for image_index, item in enumerate(mm_kwargs):
                before = _snapshot(torch.cuda, runner.device)
                start = time.perf_counter()
                gpu_forward_ms = None
                try:
                    # Split CPU items BEFORE native reduction/pinning/transfer.
                    groups = group_and_batch_mm_kwargs(
                        [item], device=runner.device, pin_memory=pin_memory
                    )
                    with closing(groups):
                        try:
                            modality, num_items, mm_kwargs_batch = next(groups)
                        except StopIteration:
                            raise RuntimeError("Pinned batcher produced no image batch") from None
                        if modality != "image" or num_items != 1:
                            raise RuntimeError("Pinned batcher must produce one image item")
                        if timing_enabled:
                            forward_start = torch.cuda.Event(enable_timing=True)
                            forward_end = torch.cuda.Event(enable_timing=True)
                            forward_start.record()
                        batch_outputs = runner.model.embed_multimodal(**mm_kwargs_batch)
                        if timing_enabled:
                            forward_end.record()
                            forward_end.synchronize()
                            gpu_forward_ms = forward_start.elapsed_time(forward_end)
                        sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=1)
                        outputs.extend(batch_outputs)
                finally:
                    # Do not leave a suspended generator or previous input alive
                    # while the next iteration's RHS performs a new transfer.
                    groups = mm_kwargs_batch = batch_outputs = None
                    forward_start = forward_end = None
                elapsed = time.perf_counter() - start
                after = _snapshot(torch.cuda, runner.device)
                logger.info("qwen38_image_encoder %s", json.dumps({
                    "event": "image_encoded", "image_index": image_index,
                    "encoder_host_elapsed_s": elapsed,
                    "gpu_forward_ms": gpu_forward_ms,
                    "completed_elapsed_s": elapsed if timing_enabled else None,
                    "before": before, "after": after,
                }, sort_keys=True))
                _cleanup_at_boundary(
                    torch.cuda, runner.device, after,
                    phase="after_image", image_index=image_index,
                )
            return outputs
        except BaseException:
            outputs.clear()
            raise
