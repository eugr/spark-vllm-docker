#!/usr/bin/env python3
"""NVFP4 Stage C: padded envelope — auto-detect vLLM site-packages."""
import os, sys
from pathlib import Path

site_packages = os.environ.get("VLLM_SITE_PACKAGES")
if not site_packages:
    for p in ["/opt/env/lib/python3.12/site-packages", "/usr/local/lib/python3.12/dist-packages"]:
        if Path(p, "vllm").exists():
            site_packages = p
            break
if not site_packages:
    sys.exit(0)

root = Path(site_packages) / "vllm"

def replace(path, old, new):
    p = root / path
    if not p.exists():
        return
    text = p.read_text()
    if new in text:
        return
    if old not in text:
        print(f"  [SKIP] {path}: anchor not found")
        return
    p.write_text(text.replace(old, new, 1))
    print(f"  [OK] {path}: patched")

print("NVFP4 Stage C: padded envelope")
replace("models/deepseek_v4/attention.py",
    '        if (\n            cache_config is not None\n            and cache_config.cache_dtype in ("nvfp4", "nvfp4_ds_mla")\n        ):\n            # Probe layout from the GLM-5.2 NVFP4 sparse-MLA path.\n            head_bytes = 416\n',
    '        if (\n            cache_config is not None\n            and cache_config.cache_dtype in ("nvfp4", "nvfp4_ds_mla")\n        ):\n            # Stage C: keep DeepSeek V4 proven 584-byte cache envelope.\n            head_bytes = 584\n')
print("Stage C complete")
