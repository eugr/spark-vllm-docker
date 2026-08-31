#!/usr/bin/env python3
"""Seed resumed Mamba state with the Mamba page size, not attention size."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "QWEN38_MAMBA_RESUME_BLOCK_SIZE_V1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/"
            "model_states/mamba_hybrid.py"
        ),
    )
    args = parser.parse_args()
    source = args.target.read_text()
    if MARKER in source:
        raise ValueError("Mamba resume block-size patch is already installed")

    old = '''        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )
'''
    new = f'''        if self._align_mode:
            # {MARKER}
            # A hybrid model can use a small attention page and a much larger
            # Mamba/GDN page. Prefix-cache resume must seed the recurrent-state
            # column in Mamba units; using cache_config.block_size can turn a
            # valid 1,600-token resume into column 399 when attention pages are
            # four tokens, causing an out-of-bounds block-table read.
            mamba_block_size = self.cache_config.mamba_block_size
            assert mamba_block_size is not None
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_block_size
            )
'''
    count = source.count(old)
    if count != 1:
        raise ValueError(f"resume seed: expected one source match, found {{count}}")
    source = source.replace(old, new, 1)

    compile(source, str(args.target), "exec")
    args.target.write_text(source)
    print(f"patched {args.target}")


if __name__ == "__main__":
    main()
