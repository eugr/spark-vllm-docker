#!/usr/bin/env python3
"""Add opt-in FlashInfer CUTLASS MoE tactic overrides for isolation tests."""

from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/core.py"
)


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"expected one patch anchor, found {count}: {old[:80]!r}")
    return source.replace(old, new, 1)


source = TARGET.read_text()
source = replace_once(
    source,
    "import functools\nimport math\n",
    "import functools\nimport math\nimport os\n",
)
source = replace_once(
    source,
    "from .utils import (\n",
    """_QWEN_GEMM1_TACTIC_RAW = os.getenv("QWEN_FLASHINFER_MOE_GEMM1_TACTIC")
_QWEN_GEMM2_TACTIC_RAW = os.getenv("QWEN_FLASHINFER_MOE_GEMM2_TACTIC")


def _qwen_parse_tactic(raw: str | None, name: str) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        tactic = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer tactic ID") from exc
    if tactic < -1:
        raise ValueError(f"{name} must be -1 (heuristic) or a non-negative tactic ID")
    return tactic


_QWEN_GEMM1_TACTIC = _qwen_parse_tactic(
    _QWEN_GEMM1_TACTIC_RAW, "QWEN_FLASHINFER_MOE_GEMM1_TACTIC"
)
_QWEN_GEMM2_TACTIC = _qwen_parse_tactic(
    _QWEN_GEMM2_TACTIC_RAW, "QWEN_FLASHINFER_MOE_GEMM2_TACTIC"
)


from .utils import (
""",
)
source = replace_once(
    source,
    """        run_moe = (
            moe_runner.fused_moe_runner.run_moe_min_latency
""",
    """        if _QWEN_GEMM1_TACTIC is not None:
            gemm_tactic_1 = _QWEN_GEMM1_TACTIC
        if _QWEN_GEMM2_TACTIC is not None:
            gemm_tactic_2 = _QWEN_GEMM2_TACTIC

        run_moe = (
            moe_runner.fused_moe_runner.run_moe_min_latency
""",
)
TARGET.write_text(source)
print(f"patched {TARGET}")
