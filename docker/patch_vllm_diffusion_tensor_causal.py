#!/usr/bin/env python3
"""Allow DiffusionGemma Tensor causal masks after vLLM PR #47914."""

import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "vllm/v1/worker/gpu/attn_utils.py"
bad_signature = "causal: bool | Mapping[int, bool] = True,"
fixed_signature = "causal: bool | Mapping[int, bool] | torch.Tensor = True,"
bad_group_causal = (
    "        group_causal = causal if isinstance(causal, bool) else "
    "causal.get(i, True)"
)
fixed_group_causal = """        if isinstance(causal, (bool, torch.Tensor)):
            group_causal = causal
        else:
            group_causal = causal.get(i, True)"""

if not target.exists():
    raise SystemExit(f"{target} not found; cannot apply DiffusionGemma causal patch")

text = target.read_text()
if fixed_signature in text and fixed_group_causal in text:
    print("DiffusionGemma Tensor causal workaround already present; skipping")
elif bad_signature in text and bad_group_causal in text:
    text = text.replace(bad_signature, fixed_signature, 1)
    text = text.replace(bad_group_causal, fixed_group_causal, 1)
    target.write_text(text)
    print("Applied DiffusionGemma Tensor causal workaround for vLLM PR #47914")
else:
    print("Known vLLM PR #47914 causal regression pattern not found; skipping")
