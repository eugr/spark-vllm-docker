#!/usr/bin/env python3
"""Runtime patch: soft-fallback --linear-backend + B12x NVFP4 kernel fix.

Applied at container startup before vLLM serves.

Patch 1 — linear/__init__.py: Replace hard ValueError in filtered-backend
  selection with a one-time warning + auto-fallback. This lets
  --linear-backend=flashinfer_b12x work on mixed NVFP4+FP8 models without
  aborting on layers that lack a b12x kernel.

Patch 2 — linear/__init__.py: Uncomment FlashInferB12xNvFp4LinearKernel from
  _POSSIBLE_NVFP4_KERNELS so --linear-backend=flashinfer_b12x can select it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


LINEAR_INIT = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/kernels/linear/__init__.py"
)


# ── Patch 1: Soft-fallback for 5 hard ValueError sites ─────────────────


SOFT_FALLBACK_RE = re.compile(
    r"""(?P<indent>[ ]{8})if not filtered:\n"""
    r"""[ ]{12}raise ValueError\(\n"""
    r"""[ ]{16}f"--linear-backend=\{linear_backend\} was requested but no "\n"""
    r"""[ ]{16}f"'\{linear_backend\}' kernel exists for (?P<kind>[^"]+)"\n"""
    r"""[ ]{12}\)\n"""
    r"""[ ]{8}(?P<var>platform_kernels|possible) = filtered""",
    re.MULTILINE,
)

SOFT_FALLBACK_REPL = """\
\\g<indent>if filtered:
            \\g<var> = filtered
        else:
            from vllm.logger import init_logger
            _linear_logger = init_logger(__name__)
            _linear_logger.warning_once(
                "--linear-background=%s has no kernel for %s; "
                "falling back to auto selection for this layer type.",
                linear_backend,
                "\\g<kind>",
            )"""


# ── Patch 2: Uncomment B12x in _POSSIBLE_NVFP4_KERNELS ──────────────────


NVFP4_OLD = (
    '        FlashInferCuteDslNvFp4LinearKernel,\n'
    '        # FlashInferB12xNvFp4LinearKernel excluded from auto-selection until\n'
    '        # upstream CUTLASS SM121 MMA op guard is resolved; use\n'
    '        # --linear-backend flashinfer_b12x to opt in explicitly.\n'
    '        FlashInferCutlassNvFp4LinearKernel,'
)

NVFP4_NEW = (
    '        FlashInferCuteDslNvFp4LinearKernel,\n'
    '        FlashInferB12xNvFp4LinearKernel,\n'
    '        FlashInferCutlassNvFp4LinearKernel,'
)


# ── Apply ──────────────────────────────────────────────────────────────


def _apply_re(path: Path, pattern: re.Pattern, repl: str, label: str) -> int:
    if not path.is_file():
        print(f"SKIP {label}: {path} not found")
        return 0
    text = path.read_text()
    new_text, n = pattern.subn(repl, text)
    if n == 0:
        if new_text == text:
            print(f"FAIL {label}: no match in {path}")
            return 0
        print(f"SKIP {label}: already applied")
        return 0
    path.write_text(new_text)
    print(f"OK   {label}: {n} site(s)")
    return 1


def _apply_text(path: Path, old: str, new: str, label: str) -> int:
    if not path.is_file():
        print(f"SKIP {label}: {path} not found")
        return 0
    text = path.read_text()
    if old not in text:
        if new in text:
            print(f"SKIP {label}: already applied")
            return 0
        print(f"FAIL {label}: old text not found in {path}")
        return 0
    text = text.replace(old, new)
    path.write_text(text)
    print(f"OK   {label}")
    return 1


changes = 0
changes += _apply_re(LINEAR_INIT, SOFT_FALLBACK_RE, SOFT_FALLBACK_REPL,
                     "Patch 1: soft-fallback linear backend")
changes += _apply_text(LINEAR_INIT, NVFP4_OLD, NVFP4_NEW,
                       "Patch 2: B12x in _POSSIBLE_NVFP4_KERNELS")

if changes:
    print(f"\nApplied {changes}/2 patches.")
else:
    print("\nNothing to patch.")
