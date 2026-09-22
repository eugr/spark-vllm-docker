#!/bin/bash
set -euo pipefail

# MiMo-V2 fp8 KV cache enablement mod.
#
# Two upstream gaps (verified against vllm main 2026-09-22):
# 1. mimo_v2.py never passes cache_config to Attention(), so Attention()
#    resolves kv_cache_dtype to "auto" and --kv-cache-dtype fp8 is silently
#    ignored on all 48 target layers.
# 2. triton_attn_diffkv.py — the backend sm_121 selects for the 192/128 K/V
#    head dims — rejects quantized KV outright and does not view the cache as
#    fp8 on read.
#
# Same fix as tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark patches 01+03, verified
# on GB10 (fp8 KV pool measured at 12.6 GiB, needle tests at 250K). Unlike his
# patch, non-e4m3 quantized caches stay rejected: the fp8 view would corrupt
# them. Each file is patched independently and skipped when already fixed
# (assumptions.md).

PREFIX="[mimo-diffkv-fp8-kv]"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MIMO="$PYTHON_ROOT/vllm/model_executor/models/mimo_v2.py"
DIFFKV="$PYTHON_ROOT/vllm/v1/attention/backends/triton_attn_diffkv.py"

echo "=== MiMo fp8 KV cache mod ==="

for f in "$MIMO" "$DIFFKV"; do
  if [ ! -f "$f" ]; then
    echo "$PREFIX Missing $f; a newer image is required." >&2
    exit 1
  fi
done

python3 - "$MIMO" "$DIFFKV" <<'PY'
from pathlib import Path
import sys

mimo_path, diffkv_path = (Path(p) for p in sys.argv[1:3])


def patch(path: Path, edits: list[tuple[str, str]], label: str) -> None:
    text = path.read_text()
    changed = False
    for old, new in edits:
        if new in text:
            continue
        if old not in text:
            raise SystemExit(
                f"[mimo-diffkv-fp8-kv] {label}: expected anchor not found; "
                "the installed vLLM differs from the layout this mod knows."
            )
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text)
        print(f"[mimo-diffkv-fp8-kv] Patched {label}.")
    else:
        print(f"[mimo-diffkv-fp8-kv] {label} already patched; skipping.")


patch(
    mimo_path,
    [
        (
            "        self.attn = Attention(\n",
            "        if cache_config is None:\n"
            "            # spark-vllm-docker/mods/mimo-diffkv-fp8-kv: no caller passes\n"
            "            # cache_config, and Attention() then resolves kv_cache_dtype\n"
            "            # to \"auto\", silently ignoring --kv-cache-dtype.\n"
            "            cache_config = get_current_vllm_config().cache_config\n"
            "        self.attn = Attention(\n",
        ),
    ],
    "mimo_v2.py",
)

patch(
    diffkv_path,
    [
        (
            "    # No FP8 / int8 KV cache for the DiffKV path yet; require fp16/bf16/fp32.\n"
            "    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n"
            '        "auto",\n'
            '        "bfloat16",\n'
            "    ]\n",
            "    # fp8 (e4m3) KV: the reshape kernel quantizes on write; the attention\n"
            "    # kernel upcasts K/V to bf16 on load. KV scales are not applied on read,\n"
            "    # which is exact for unit-scale checkpoints (MiMo-V2).\n"
            "    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [\n"
            '        "auto",\n'
            '        "bfloat16",\n'
            '        "fp8",\n'
            '        "fp8_e4m3",\n'
            "    ]\n",
        ),
        (
            "        if is_quantized_kv_cache(self.kv_cache_dtype):\n"
            "            raise NotImplementedError(\n"
            '                "TritonAttentionDiffKVBackend does not yet support quantized "\n'
            '                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."\n'
            "            )\n",
            "        if is_quantized_kv_cache(self.kv_cache_dtype) and (\n"
            '            self.kv_cache_dtype not in ("fp8", "fp8_e4m3")\n'
            "        ):\n"
            "            raise NotImplementedError(\n"
            '                "TritonAttentionDiffKVBackend only supports fp8_e4m3 KV "\n'
            '                f"among quantized caches (got {self.kv_cache_dtype!r})."\n'
            "            )\n",
        ),
        (
            "        # Triton DiffKV kernels consume (B, N, H, D) cache views.\n"
            "        kv_cache = kv_cache.transpose(1, 2)\n"
            "        key_cache = kv_cache[..., :head_size_qk]\n",
            "        # Triton DiffKV kernels consume (B, N, H, D) cache views.\n"
            "        kv_cache = kv_cache.transpose(1, 2)\n"
            "        if is_quantized_kv_cache(self.kv_cache_dtype) and (\n"
            "            kv_cache.dtype != self.fp8_dtype\n"
            "        ):\n"
            "            kv_cache = kv_cache.view(self.fp8_dtype)\n"
            "        key_cache = kv_cache[..., :head_size_qk]\n",
        ),
    ],
    "triton_attn_diffkv.py",
)
PY

echo "=====> MiMo attention layers honor --kv-cache-dtype fp8 on the DiffKV backend"