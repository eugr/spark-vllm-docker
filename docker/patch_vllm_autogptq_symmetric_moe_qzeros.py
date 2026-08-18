#!/usr/bin/env python3
"""Do not pass AutoGPTQ MoE zero-points for symmetric quantization."""

import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "vllm/model_executor/layers/quantization/auto_gptq.py"
bad = '''            w1_zp=getattr(layer, "w13_qzeros", None),
            w2_zp=getattr(layer, "w2_qzeros", None),'''
fixed = '''            w1_zp=getattr(layer, "w13_qzeros", None)
            if not self.quant_config.is_sym
            else None,
            w2_zp=getattr(layer, "w2_qzeros", None)
            if not self.quant_config.is_sym
            else None,'''

if not target.exists():
    print(f"{target} not found; skipping AutoGPTQ MoE qzeros workaround")
else:
    text = target.read_text()
    if fixed in text:
        print("AutoGPTQ MoE qzeros workaround already present; skipping")
    elif bad in text:
        target.write_text(text.replace(bad, fixed, 1))
        print("Applied AutoGPTQ symmetric MoE qzeros workaround")
    else:
        print("Known vulnerable AutoGPTQ MoE qzeros pattern not found; skipping")
