# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-static NVFP4 MLA cache-format configuration and ABI identity."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

NVFP4_MLA_SCALES_ENV = "VLLM_NVFP4_MLA_SCALES_FILE"
NVFP4_MLA_DYNAMIC_SCALE_ENV = "VLLM_NVFP4_MLA_DYNAMIC_SCALE"
KV_FP8_ROPE_ENV = "KV_FP8_ROPE"


@dataclass(frozen=True)
class Nvfp4MlaCacheFormat:
    """Immutable writer/reader configuration captured at process import."""

    dynamic_scale: bool
    fp8_rope: bool
    scales_file: str

    @classmethod
    def from_env(cls) -> Nvfp4MlaCacheFormat:
        return cls(
            dynamic_scale=os.getenv(NVFP4_MLA_DYNAMIC_SCALE_ENV, "0") == "1",
            fp8_rope=os.getenv(KV_FP8_ROPE_ENV, "0") == "1",
            scales_file=os.getenv(NVFP4_MLA_SCALES_ENV, "").strip(),
        )

    def validate(self) -> None:
        if self.dynamic_scale and self.scales_file:
            raise ValueError(
                f"{NVFP4_MLA_SCALES_ENV} and "
                f"{NVFP4_MLA_DYNAMIC_SCALE_ENV}=1 are mutually exclusive"
            )
        if self.dynamic_scale and not self.fp8_rope:
            raise ValueError(
                f"{NVFP4_MLA_DYNAMIC_SCALE_ENV}=1 requires {KV_FP8_ROPE_ENV}=1"
            )

    def record_abi(self, cache_dtype: str) -> str:
        """Return an identity suitable for persistent external-cache keys."""
        normalized_dtype = str(cache_dtype).replace("torch.", "")
        if normalized_dtype != "nvfp4_ds_mla":
            return "vllm-default-v1"

        self.validate()
        if not self.dynamic_scale and not self.scales_file:
            # Preserve the existing namespace for every unconfigured/default
            # deployment. Only modes that change the record's scale semantics
            # opt into a new external-cache identity.
            return "vllm-default-v1"

        layout = "fp8-rope-368" if self.fp8_rope else "bf16-rope-432"
        if self.dynamic_scale:
            scale_mode = "dynamic-token-v1"
        else:
            try:
                scale_digest = hashlib.sha256(
                    Path(self.scales_file).read_bytes()
                ).hexdigest()
            except OSError as exc:
                raise ValueError(
                    f"Cannot fingerprint {NVFP4_MLA_SCALES_ENV}={self.scales_file!r}"
                ) from exc
            scale_mode = f"static-calibrated-v1:{scale_digest}"
        return f"nvfp4_ds_mla:{layout}:{scale_mode}"


# All consumers import this one frozen value, so a process cannot configure
# the writer, readers, and external-cache namespace from different env reads.
NVFP4_MLA_CACHE_FORMAT = Nvfp4MlaCacheFormat.from_env()
