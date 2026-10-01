#!/usr/bin/env python3
"""NVFP4 Stage A: dtype plumbing — auto-detect vLLM site-packages."""
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

print("NVFP4 Stage A: dtype plumbing")
replace("config/cache.py",
    '    "fp8_ds_mla",\n    "turboquant_k8v4",',
    '    "fp8_ds_mla",\n    "nvfp4_ds_mla",\n    "turboquant_k8v4",')
replace("utils/torch_utils.py",
    '    "fp8_ds_mla": torch.uint8,\n    "turboquant_k8v4": torch.uint8,',
    '    "fp8_ds_mla": torch.uint8,\n    "nvfp4_ds_mla": torch.uint8,\n    "turboquant_k8v4": torch.uint8,')
replace("utils/torch_utils.py",
    '        or kv_cache_dtype == "nvfp4"\n',
    '        or kv_cache_dtype == "nvfp4"\n        or kv_cache_dtype == "nvfp4_ds_mla"\n')
replace("v1/kv_cache_interface.py",
    '    if kv_cache_dtype == "nvfp4":\n        return KVQuantMode.NVFP4\n',
    '    if kv_cache_dtype == "nvfp4":\n        return KVQuantMode.NVFP4\n    if kv_cache_dtype == "nvfp4_ds_mla":\n        return KVQuantMode.NVFP4\n')
replace("v1/kv_cache_interface.py",
    '        if self.cache_dtype_str == "fp8_ds_mla":\n',
    '        if self.cache_dtype_str == "nvfp4_ds_mla":\n            return self.storage_block_size * 416\n        if self.cache_dtype_str == "fp8_ds_mla":\n')
print("Stage A complete")
