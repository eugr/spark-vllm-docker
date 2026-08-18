#!/usr/bin/env python3
"""Restrict vLLM's cooperative sparse-attention top-k path to SM90."""

import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "vllm/model_executor/layers/sparse_attn_indexer.py"
old = '''        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and current_platform.has_device_capability(90)
        )'''
new = '''        device_capability = current_platform.get_device_capability()
        use_cooperative_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and num_rows <= 32
            and logits.stride(0) % 4 == 0  # TMA 16-byte alignment
            and device_capability is not None
            and device_capability.to_int() == 90
        )'''

if not target.exists():
    print(f"{target} not found; skipping SM120 cooperative_topk workaround")
else:
    text = target.read_text()
    if "device_capability.to_int() == 90" in text:
        print("SM120 cooperative_topk workaround already present; skipping")
    elif old in text:
        target.write_text(text.replace(old, new, 1))
        print("Applied SM120 cooperative_topk workaround")
    else:
        print("Known cooperative_topk selector pattern not found; skipping")
