#!/usr/bin/env python3
"""Restore the unsupported-primary SWA block fallback changed by vLLM #53007."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


TARGET_REL = Path("vllm/model_executor/layers/attention/attention.py")
MARKER = "spark-vllm-docker: preserve unsupported-primary SWA fallback"
ANCHOR = """            page_budget = shared_page or sw_per_token * block_size
            sw_block_size = _largest_kernel_block_within(
                self.attn_backend, sw_per_token, page_budget, block_size
            )
            return SlidingWindowSpec(
"""
REPLACEMENT = f"""            page_budget = shared_page or sw_per_token * block_size
            sw_block_size = _largest_kernel_block_within(
                self.attn_backend, sw_per_token, page_budget, block_size
            )
            # {MARKER}.
            # Keep #53007's primary-size choice when the backend supports it.
            # Otherwise start small so page unification can scale the block
            # exactly, instead of padding a larger non-divisor (e.g. 64 vs
            # 1648 tokens for FlashInfer on Qwen3.8 + DFlash2).
            # Explicitly padded skip-quant pages still prefer the largest fit.
            if shared_page is None and sw_block_size != block_size:
                kernel_sizes = self.attn_backend.get_supported_kernel_block_sizes()
                sw_block_size = min(
                    (s if isinstance(s, int) else s.base for s in kernel_sizes),
                    default=block_size,
                )
            return SlidingWindowSpec(
"""
LEGACY_CALL = """            sw_block_size = _largest_kernel_block_within(
                self.attn_backend, sw_per_token, shared_page, block_size
            )
"""


class PatchError(RuntimeError):
    """The selection code has an unexpected or partially patched layout."""


def patch_source(source: str) -> tuple[str, str]:
    if MARKER in source:
        if source.count(REPLACEMENT) != 1 or ANCHOR in source:
            raise PatchError("SWA block fallback patch is incomplete or duplicated")
        return source, "SWA block fallback fix is already present; skipping"
    if "def _largest_kernel_block_within(" not in source:
        return source, "Affected SWA block selector is absent; skipping"
    if LEGACY_CALL in source and "page_budget = shared_page or" not in source:
        return source, "Pre-#53007 SWA block selection is present; skipping"
    if source.count(ANCHOR) != 1:
        raise PatchError(
            "expected exactly one #53007 SWA selection block; "
            "review the upstream implementation before updating this patch"
        )
    patched = source.replace(ANCHOR, REPLACEMENT, 1)
    ast.parse(patched)
    return patched, "Applied SWA block fallback fix for vLLM PR #53007"


def main() -> None:
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    target = source_root / TARGET_REL
    if not target.exists():
        print(f"{TARGET_REL} is absent; SWA block fallback patch is not applicable")
        return
    source = target.read_text()
    try:
        patched, message = patch_source(source)
    except (PatchError, SyntaxError) as exc:
        raise SystemExit(f"SWA block fallback patch failed: {exc}") from exc
    if patched != source:
        target.write_text(patched)
    print(message)


if __name__ == "__main__":
    main()
