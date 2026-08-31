#!/usr/bin/env python3
"""Enable Qwen3.8 QSA's Triton path for FP8-E4M3 KV caches."""

from __future__ import annotations

from pathlib import Path


QSA_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/qsa.py"
)
OPS_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/ops/qsa.py"
)
MARKER = "QWEN38_QSA_FP8_KV_V1"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected {description} exactly once; found {count}")
    return source.replace(old, new)


def patch_qsa(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("QSA FP8-KV support is already patched")

    source = replace_once(
        source,
        "    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]\n"
        '    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]\n',
        "    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]\n"
        f"    # {MARKER}\n"
        "    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n"
        '        "auto",\n'
        '        "bfloat16",\n'
        '        "fp8",\n'
        '        "fp8_e4m3",\n'
        "    ]\n"
        "\n"
        "    @classmethod\n"
        "    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:\n"
        "        # QSA has its own Triton sparse-attention kernel, so the generic\n"
        "        # FlashAttention capability gate does not apply to it.\n"
        "        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes\n"
        "\n"
        "    @classmethod\n"
        "    def supports_combination(\n"
        "        cls,\n"
        "        head_size: int,\n"
        "        dtype: torch.dtype,\n"
        "        kv_cache_dtype: CacheDType | None,\n"
        "        block_size: int | None,\n"
        "        use_mla: bool,\n"
        "        has_sink: bool,\n"
        "        use_sparse: bool,\n"
        "        use_mm_prefix: bool,\n"
        "        device_capability,\n"
        "    ) -> str | None:\n"
        '        if kv_cache_dtype in ("fp8", "fp8_e4m3"):\n'
        "            if dtype != torch.bfloat16:\n"
        '                return "QSA FP8 KV cache requires BF16 queries"\n'
        "            if has_sink:\n"
        '                return "QSA does not support attention sinks"\n'
        "            return None\n"
        "        return super().supports_combination(\n"
        "            head_size,\n"
        "            dtype,\n"
        "            kv_cache_dtype,\n"
        "            block_size,\n"
        "            use_mla,\n"
        "            has_sink,\n"
        "            use_sparse,\n"
        "            use_mm_prefix,\n"
        "            device_capability,\n"
        "        )\n",
        "QSA backend dtype declaration",
    )

    source = replace_once(
        source,
        "    def __init__(self, *args, **kwargs) -> None:\n"
        "        super().__init__(*args, **kwargs)\n",
        "    def __init__(self, *args, **kwargs) -> None:\n"
        "        # FlashAttentionImpl validates FP8 against flash-attn. QSA does not\n"
        "        # invoke that library for attention, so initialize the shared metadata\n"
        "        # machinery as BF16 and then restore the requested cache dtype.\n"
        '        requested_kv_cache_dtype = kwargs.get("kv_cache_dtype")\n'
        "        mutable_args = list(args)\n"
        "        if requested_kv_cache_dtype is None and len(mutable_args) > 6:\n"
        "            requested_kv_cache_dtype = mutable_args[6]\n"
        '        if requested_kv_cache_dtype in ("fp8", "fp8_e4m3"):\n'
        '            if "kv_cache_dtype" in kwargs:\n'
        '                kwargs["kv_cache_dtype"] = "auto"\n'
        "            else:\n"
        '                mutable_args[6] = "auto"\n'
        "        super().__init__(*mutable_args, **kwargs)\n"
        "        if requested_kv_cache_dtype is not None:\n"
        "            self.kv_cache_dtype = requested_kv_cache_dtype\n",
        "QSA implementation initializer",
    )
    source = replace_once(
        source,
        '        if self.kv_cache_dtype not in ("auto", "bfloat16"):\n'
        "            raise NotImplementedError(\n"
        '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
        "            )\n",
        '        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):\n'
        "            raise NotImplementedError(\n"
        '                "Qwen3.8-Flash-Next QSA supports BF16 or FP8-E4M3 KV cache"\n'
        "            )\n",
        "QSA implementation cache validation",
    )
    source = replace_once(
        source,
        "        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:\n"
        '            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")\n',
        '        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):\n'
        "            fp8_dtype = current_platform.fp8_dtype()\n"
        "            key_cache = key_cache.view(fp8_dtype)\n"
        "            value_cache = value_cache.view(fp8_dtype)\n"
        "        if query.dtype != torch.bfloat16 or key_cache.dtype not in (\n"
        "            torch.bfloat16,\n"
        "            current_platform.fp8_dtype(),\n"
        "        ):\n"
        "            raise NotImplementedError(\n"
        '                "Qwen3.8-Flash-Next QSA requires BF16 Q and BF16/FP8-E4M3 K/V"\n'
        "            )\n",
        "QSA runtime cache dtype validation",
    )
    source = replace_once(
        source,
        "            token_to_req,\n"
        "            output[:num_tokens],\n"
        "        )\n",
        "            token_to_req,\n"
        "            output[:num_tokens],\n"
        "            k_scale=layer._k_scale,\n"
        "            v_scale=layer._v_scale,\n"
        "        )\n",
        "QSA sparse-attention call",
    )
    source = replace_once(
        source,
        '        if cache_config.cache_dtype not in ("auto", "bfloat16"):\n'
        "            raise NotImplementedError(\n"
        '                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"\n'
        "            )\n",
        "        if cache_config.cache_dtype not in (\n"
        '            "auto",\n'
        '            "bfloat16",\n'
        '            "fp8",\n'
        '            "fp8_e4m3",\n'
        "        ):\n"
        "            raise NotImplementedError(\n"
        '                "Qwen3.8-Flash-Next QSA supports BF16 or FP8-E4M3 KV cache"\n'
        "            )\n",
        "QSA layer cache validation",
    )
    source = replace_once(
        source,
        "        if self.kv_cache_torch_dtype != torch.bfloat16:\n"
        "            raise NotImplementedError(\n"
        '                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"\n'
        "            )\n",
        "        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8):\n"
        "            raise NotImplementedError(\n"
        '                "Qwen3.8-Flash-Next QSA requires BF16 or packed FP8 cache storage"\n'
        "            )\n",
        "QSA cache storage validation",
    )
    return source


def patch_ops(source: str) -> str:
    if MARKER in source:
        raise RuntimeError("QSA FP8-KV ops are already patched")
    source = replace_once(
        source,
        '"""Triton kernels for the Qwen3.8-Flash-Next weight-free QSA path."""',
        f'"""Triton kernels for Qwen3.8 QSA. {MARKER}"""',
        "QSA ops module marker",
    )
    source = replace_once(
        source,
        "    token_to_req_ptr,\n"
        "    partial_output_ptr,\n",
        "    token_to_req_ptr,\n"
        "    k_scale_ptr,\n"
        "    v_scale_ptr,\n"
        "    partial_output_ptr,\n",
        "QSA split-K kernel scale arguments",
    )
    source = replace_once(
        source,
        "    BLOCK_M: tl.constexpr,\n"
        "    BLOCK_N: tl.constexpr,\n"
        ") -> None:\n",
        "    BLOCK_M: tl.constexpr,\n"
        "    BLOCK_N: tl.constexpr,\n"
        "    FP8_KV: tl.constexpr,\n"
        ") -> None:\n",
        "QSA split-K kernel constexprs",
    )
    source = replace_once(
        source,
        "        scores = tl.dot(query, keys)\n"
        "        # Scaling scores avoids re-quantizing a scaled query to BF16.\n",
        "        if FP8_KV:\n"
        "            # Generic cache writes store fp8(K / k_scale) and fp8(V / v_scale).\n"
        "            keys = keys.to(tl.bfloat16)\n"
        "            values = (\n"
        "                values.to(tl.float32) * tl.load(v_scale_ptr)\n"
        "            ).to(tl.bfloat16)\n"
        "        scores = tl.dot(query, keys)\n"
        "        if FP8_KV:\n"
        "            scores *= tl.load(k_scale_ptr)\n"
        "        # Scaling scores avoids re-quantizing a scaled query to BF16.\n",
        "QSA FP8 cache dequantization",
    )
    source = replace_once(
        source,
        "    token_to_req: torch.Tensor,\n"
        "    out: torch.Tensor | None = None,\n"
        ") -> torch.Tensor:\n"
        '    """Run sparse GQA directly over paged BF16 K/V caches."""\n',
        "    token_to_req: torch.Tensor,\n"
        "    out: torch.Tensor | None = None,\n"
        "    k_scale: torch.Tensor | None = None,\n"
        "    v_scale: torch.Tensor | None = None,\n"
        ") -> torch.Tensor:\n"
        '    """Run sparse GQA directly over paged BF16 or FP8-E4M3 K/V caches."""\n',
        "QSA sparse-attention signature",
    )
    source = replace_once(
        source,
        "    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16\n"
        "    assert logical_indices.dtype == block_table.dtype == torch.int32\n",
        "    fp8_kv = k_cache.dtype == current_platform.fp8_dtype()\n"
        "    assert q.dtype == torch.bfloat16\n"
        "    assert k_cache.dtype == v_cache.dtype\n"
        "    assert k_cache.dtype in (torch.bfloat16, current_platform.fp8_dtype())\n"
        "    if k_scale is None:\n"
        "        k_scale = torch.ones((), dtype=torch.float32, device=q.device)\n"
        "    if v_scale is None:\n"
        "        v_scale = torch.ones((), dtype=torch.float32, device=q.device)\n"
        "    assert k_scale.numel() == v_scale.numel() == 1\n"
        "    assert k_scale.dtype == v_scale.dtype == torch.float32\n"
        "    assert k_scale.device == v_scale.device == q.device\n"
        "    assert logical_indices.dtype == block_table.dtype == torch.int32\n",
        "QSA sparse-attention dtype validation",
    )
    source = replace_once(
        source,
        "        token_to_req,\n"
        "        partial_output,\n",
        "        token_to_req,\n"
        "        k_scale,\n"
        "        v_scale,\n"
        "        partial_output,\n",
        "QSA kernel scale operands",
    )
    source = replace_once(
        source,
        "        BLOCK_M=block_m,\n"
        "        BLOCK_N=block_n,\n"
        "        num_warps=partial_warps,\n",
        "        BLOCK_M=block_m,\n"
        "        BLOCK_N=block_n,\n"
        "        FP8_KV=fp8_kv,\n"
        "        num_warps=partial_warps,\n",
        "QSA kernel FP8 specialization flag",
    )
    return source


def main() -> int:
    QSA_PATH.write_text(patch_qsa(QSA_PATH.read_text()))
    OPS_PATH.write_text(patch_ops(OPS_PATH.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
