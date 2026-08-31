#!/usr/bin/env python3
"""Make the expensive API multimodal processor warmup opt-out.

The model and multimodal processor remain enabled.  This only defers the
synthetic maximum-context processor call from API startup to the first real
multimodal request.  The experiment launcher uses it together with vLLM's
upstream ``--skip-mm-profiling`` flag and an explicit KV-cache allocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "QWEN_DEFER_MM_PROCESSOR_WARMUP_V1"


def patch_renderer(source: str) -> str:
    if MARKER in source:
        raise ValueError("renderer is already patched")

    old_imports = """import asyncio
import time
"""
    new_imports = """import asyncio
import os
import time
"""
    if source.count(old_imports) != 1:
        raise ValueError("expected renderer imports exactly once")
    source = source.replace(old_imports, new_imports)

    old_block = """            if self.mm_processor:
                try:
                    logger.debug("Warming up multi-modal processing...")
                    self._warmup_mm_processor(
                        self.mm_processor,
                        log_prefix="Multi-modal",
                    )
                except Exception:
                    logger.warning("Multi-modal warmup failed")
                finally:
                    self.clear_mm_cache()

            if self._readonly_mm_processor is not None:
                try:
                    logger.debug("Warming up readonly multi-modal processing...")
                    self._warmup_mm_processor(
                        self._readonly_mm_processor,
                        log_prefix="Readonly multi-modal",
                    )
                except Exception:
                    logger.warning("Readonly multi-modal warmup failed")
                finally:
                    self._clear_processor_cache(self._readonly_mm_processor)
"""
    new_block = f"""            # {MARKER}. Keep multimodal serving enabled, but permit a
            # bounded experiment profile to defer this synthetic maximum-size
            # processor call. The first real image request is the correctness
            # and memory gate for the opt-in path.
            defer_mm_warmup = (
                os.environ.get("QWEN_DEFER_MM_PROCESSOR_WARMUP", "0") == "1"
            )
            if defer_mm_warmup and (
                self.mm_processor is not None
                or self._readonly_mm_processor is not None
            ):
                logger.warning(
                    "Deferring multimodal processor warmup to the first real "
                    "multimodal request."
                )

            if self.mm_processor and not defer_mm_warmup:
                try:
                    logger.debug("Warming up multi-modal processing...")
                    self._warmup_mm_processor(
                        self.mm_processor,
                        log_prefix="Multi-modal",
                    )
                except Exception:
                    logger.warning("Multi-modal warmup failed")
                finally:
                    self.clear_mm_cache()

            if self._readonly_mm_processor is not None and not defer_mm_warmup:
                try:
                    logger.debug("Warming up readonly multi-modal processing...")
                    self._warmup_mm_processor(
                        self._readonly_mm_processor,
                        log_prefix="Readonly multi-modal",
                    )
                except Exception:
                    logger.warning("Readonly multi-modal warmup failed")
                finally:
                    self._clear_processor_cache(self._readonly_mm_processor)
"""
    if source.count(old_block) != 1:
        raise ValueError("expected multimodal warmup block exactly once")
    return source.replace(old_block, new_block)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renderer",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py"
        ),
    )
    args = parser.parse_args()
    args.renderer.write_text(patch_renderer(args.renderer.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
