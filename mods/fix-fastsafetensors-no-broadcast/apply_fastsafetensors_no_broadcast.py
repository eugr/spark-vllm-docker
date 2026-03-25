#!/usr/bin/env python3
"""
Force fastsafetensors to use SingleGroup (independent per-rank loading) instead
of the TP process group (NCCL broadcast during loading).

On GB10 unified memory (128GB shared CPU/GPU), the NCCL broadcast path in
fastsafetensors causes OOM: each reading rank temporarily holds
ALL tensors from a shard (~5GB) before broadcasting half away. With 100GB of
accumulated weights + 5GB broadcast buffer + OS overhead > 119GB usable.

With SingleGroup, each rank reads ALL shard files independently and extracts
only its own TP partition. No NCCL during loading. Peak memory = accumulated
own tensors + single tensor being read (MB, not GB).

After loading, NCCL is used normally for tensor-parallel inference — this patch
only affects the weight loading phase.

Trade-off: each rank independently reads the full checkpoint from disk (local NVMe
or networked) instead of receiving its shards via NCCL broadcast. This eliminates
the massive network memory buffers that crash GB10 Unified Memory architectures.
"""

import site
import re
from pathlib import Path

site_pkgs = Path(site.getsitepackages()[0])
weight_utils = site_pkgs / "vllm" / "model_executor" / "model_loader" / "weight_utils.py"

if not weight_utils.exists():
    print(f"  WARN: {weight_utils} not found, skipping.")
    exit(0)

src = weight_utils.read_text()

# Target: the weight_files_sub_lists construction in fastsafetensors_weights_iterator.
# Original code (may vary slightly across versions):
#
#   weight_files_sub_lists = [
#       hf_weights_files[i : i + pg.size()]
#       for i in range(0, len(hf_weights_files), pg.size())
#   ]
#
# We replace pg.size() with 1 in both the slice and the range step, making
# every file its own sub-list. Each rank processes every sub-list (SingleGroup
# behavior) regardless of the actual process group.

pattern = re.compile(
    r"(weight_files_sub_lists\s*=\s*\[\s*\n"
    r"\s*hf_weights_files\[i\s*:\s*i\s*\+\s*)pg\.size\(\)"
    r"(\]\s*\n\s*for\s+i\s+in\s+range\(0,\s*len\(hf_weights_files\),\s*)"
    r"pg\.size\(\)"
    r"(\)\s*\n\s*\])",
    re.MULTILINE,
)

match = pattern.search(src)
if not match:
    # Try alternate formatting (single line or different whitespace)
    alt = re.compile(
        r"(hf_weights_files\[i\s*:\s*i\s*\+\s*)pg\.size\(\)"
        r"(.*?range\(0,\s*len\(hf_weights_files\),\s*)"
        r"pg\.size\(\)",
        re.DOTALL,
    )
    match = alt.search(src)
    if not match:
        print("  WARN: Could not find fastsafetensors weight_files_sub_lists pattern, skipping.")
        exit(0)

# Inject `pg = None` directly before the assignment, and replace pg.size() with 1
replaced_block = "pg = None\n    " + match.group(0).replace("pg.size()", "1")
new_src = src[:match.start()] + replaced_block + src[match.end():]

# Scrub remaining pg.size() usages which would now crash as pg is None.
# Enforce nogds=True (required for GB10) by substituting the boolean check.
new_src = new_src.replace("pg.size() > 1", "True")
new_src = new_src.replace("pg.size()", "1")

# Add a comment before the patched line
comment = "    # GB10: force per-rank independent loading (no NCCL broadcast) for unified memory\n"
insert_pos = new_src.rfind("\n", 0, new_src.find("pg = None")) + 1
if new_src[insert_pos:insert_pos + len(comment)] != comment:
    new_src = new_src[:insert_pos] + comment + new_src[insert_pos:]

if new_src != src:
    weight_utils.write_text(new_src)
    print("  Applied fastsafetensors no-broadcast patch (pg=None enforced).")
else:
    print("  Already applied or no change needed.")
