"""Packed MXFP4-weight x dynamic-MXFP8 MTP draft vocabulary head."""

from __future__ import annotations

import os
from pathlib import Path

import flashinfer
import flashinfer.gemm
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_ARTIFACT = "/models/qwen38-retain12/draft_lm_head_mxfp4.safetensors"


def draft_head_mxfp4_enabled() -> bool:
    value = os.environ.get("QWEN_DRAFT_HEAD_MXFP4_ENABLED", "0")
    if value not in {"0", "1"}:
        raise ValueError("QWEN_DRAFT_HEAD_MXFP4_ENABLED must be 0 or 1")
    return value == "1"


def draft_head_mxfp4_artifact() -> Path:
    return Path(os.environ.get("QWEN_DRAFT_HEAD_MXFP4_ARTIFACT", _DEFAULT_ARTIFACT))


def _tile_n() -> int:
    value = int(os.environ.get("QWEN_DRAFT_HEAD_MXFP4_TILE_N", "32"))
    if value not in {32, 64, 128}:
        raise ValueError("QWEN_DRAFT_HEAD_MXFP4_TILE_N must be 32, 64, or 128")
    return value


class PackedMxfp4DraftHead(nn.Module):
    """One-group adapter around FlashInfer's native SM120/121 mixed GEMM."""

    def __init__(
        self,
        artifact_path: Path | str,
        *,
        vocab_size: int,
        hidden_size: int,
        max_padded_rows: int = 64,
    ) -> None:
        super().__init__()
        artifact_path = Path(artifact_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"packed draft-head artifact missing: {artifact_path}")
        if torch.cuda.get_device_capability() not in {(12, 0), (12, 1)}:
            raise RuntimeError(
                "packed draft head currently requires NVIDIA SM120 or SM121"
            )

        with safe_open(artifact_path, framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            if metadata.get("format") != "qwen_draft_lm_head_mxfp4_v1":
                raise ValueError(f"unexpected draft-head metadata: {metadata}")
        device = f"cuda:{torch.cuda.current_device()}"
        tensors = load_file(artifact_path, device=device)
        packed = tensors["weight_packed"]
        scales = tensors["weight_scale"]
        expected_packed = (1, vocab_size, hidden_size // 2)
        expected_scales = (1, vocab_size, hidden_size // 32)
        if tuple(packed.shape) != expected_packed or packed.dtype != torch.uint8:
            raise ValueError(
                f"bad packed draft weight: {tuple(packed.shape)} {packed.dtype}; "
                f"expected {expected_packed} uint8"
            )
        if tuple(scales.shape) != expected_scales or scales.dtype != torch.uint8:
            raise ValueError(
                f"bad packed draft scales: {tuple(scales.shape)} {scales.dtype}; "
                f"expected {expected_scales} uint8"
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tile_n = _tile_n()
        self.max_padded_rows = max_padded_rows
        # vLLM's generic Eagle/MTP loader normally replaces a draft head with
        # the target head. The patched sharing utility recognizes this marker
        # and preserves the packed independent head.
        self._qwen_force_own_lm_head = True
        self.register_buffer("weight_packed", packed, persistent=False)
        self.register_buffer("weight_scale", scales, persistent=False)
        for padded_m in range(4, max_padded_rows + 1, 4):
            self.register_buffer(
                f"_indptr_{padded_m}",
                torch.tensor([0, padded_m], device=device, dtype=torch.int32),
                persistent=False,
            )

        packed_bytes = packed.numel() * packed.element_size()
        scale_bytes = scales.numel() * scales.element_size()
        logger.info(
            "QWEN_DRAFT_HEAD_MXFP4 loaded artifact=%s logical_shape=(%d,%d) "
            "packed_bytes=%d scale_bytes=%d tile_n=%d backend="
            "flashinfer_group_gemm_mxfp8_mxfp4_nt_groupwise",
            artifact_path,
            vocab_size,
            hidden_size,
            packed_bytes,
            scale_bytes,
            self.tile_n,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"unsupported draft hidden shape: {tuple(hidden_states.shape)}"
            )
        if hidden_states.dtype not in {torch.bfloat16, torch.float16}:
            raise ValueError(f"unsupported draft hidden dtype: {hidden_states.dtype}")
        m = hidden_states.shape[0]
        padded_m = ((m + 3) // 4) * 4
        if padded_m == 0 or padded_m > self.max_padded_rows:
            raise ValueError(
                f"draft-head rows {m} exceed supported range 1..{self.max_padded_rows}"
            )
        if padded_m != m:
            hidden_states = F.pad(hidden_states, (0, 0, 0, padded_m - m))
        hidden_states = hidden_states.contiguous()

        activation, activation_scale = flashinfer.mxfp8_quantize(
            hidden_states,
            is_sf_swizzled_layout=True,
            backend="cuda",
        )
        output = torch.empty(
            (padded_m, self.vocab_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        indptr = getattr(self, f"_indptr_{padded_m}")
        flashinfer.gemm.group_gemm_mxfp8_mxfp4_nt_groupwise(
            activation,
            self.weight_packed,
            activation_scale,
            self.weight_scale,
            indptr,
            tile_m=128,
            tile_n=self.tile_n,
            tile_k=128,
            swap_ab=True,
            out=output,
        )
        return output[:m]


__all__ = [
    "PackedMxfp4DraftHead",
    "draft_head_mxfp4_artifact",
    "draft_head_mxfp4_enabled",
]
