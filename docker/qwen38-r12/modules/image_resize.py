"""Opt-in Qwen retain-12 native CPU early resize; no model or tokenizer load.

Install as qwen38_image_resize.py. The patched ImageMediaIO invokes this only
for server-owned media_io_kwargs={"image": {"qwen38_early_resize": True}}.
The middleware must reject client media/processor overrides and retain the
matching fixed 65536..2097152 / patch16 / merge2 policy. This module cannot
infer model identity: only that Qwen route may insert the flag.
"""

from functools import lru_cache

MIN_PIXELS = 65536
MAX_PIXELS = 2097152
MAX_SOURCE_PIXELS = 16777216
NAMESPACE = "qwen38_early_resize"


@lru_cache(maxsize=1)
def _processor():
    # No from_pretrained, package downloads, device probing, or global patches.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    return Qwen2VLImageProcessor(
        size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
        patch_size=16, temporal_patch_size=2, merge_size=2,
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
    )


def early_resize(image, io_config=None):
    """Return (new reduced PIL, new hash metadata); never mutate/close input.

    Input must already have passed native EXIF and white/RGB conversion.
    The caller owns and closes the source, including on exceptions. Only the
    reduced PIL survives this function; no full-source tensors are cached.
    """
    if image.mode != "RGB":
        raise ValueError("Qwen early resize requires native RGB conversion")
    width, height = image.size
    if width <= 0 or height <= 0 or width * height > MAX_SOURCE_PIXELS:
        raise ValueError("Qwen early resize source exceeds fixed pixel policy")

    import torch
    from torchvision.transforms.v2 import functional as tvF
    from transformers.image_utils import SizeDict

    processor = _processor()
    with torch.inference_mode():
        chw = processor.process_image(image, do_convert_rgb=True, device="cpu")
        if chw.device.type != "cpu" or chw.dtype != torch.uint8:
            raise ValueError("Qwen early resize expected CPU uint8 source")
        resized = processor.resize(
            images=chw.unsqueeze(0),
            size=SizeDict(shortest_edge=MIN_PIXELS, longest_edge=MAX_PIXELS),
            resample=processor.resample, factor=32,
        ).squeeze(0)
        if resized.device.type != "cpu" or resized.dtype != torch.uint8:
            raise ValueError("Qwen early resize expected CPU uint8 result")
        out_h, out_w = resized.shape[-2:]
        if out_h % 32 or out_w % 32 or not MIN_PIXELS <= out_h * out_w <= MAX_PIXELS:
            raise ValueError("Qwen early resize output violates fixed geometry")
        reduced = tvF.to_pil_image(resized)

    # Preserve original bytes in the caller's NEW MediaWithBytes wrapper. This
    # namespace records that pixels are no longer just the original decode.
    # Do not copy EXIF: orientation has already been applied by the native loader.
    config = dict(io_config or {})
    config[NAMESPACE] = {
        "version": 1, "backend": "hf-qwen2vl-torchvision-cpu-uint8",
        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
        "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
        "resample": int(processor.resample), "antialias": True,
        "image_mode": "RGB", "rgba_background_color": [255, 255, 255],
        "exif": "native-before-resize",
    }
    return reduced, config
