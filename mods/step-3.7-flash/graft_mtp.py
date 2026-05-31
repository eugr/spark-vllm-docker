#!/usr/bin/env python3
"""Graft BF16 MTP (next-n predict) weights into a cached Step-3.7-Flash-NVFP4
snapshot so MTP speculative decoding works.

The official NVFP4 export ships no MTP weights (ModelOpt strips layers 45-47 and
truncates several per-layer config lists during quantization). This script:
  1. downloads the MTP shard from the original stepfun-ai/Step-3.7-Flash (BF16),
  2. extracts the ~51 MTP tensors (layers >= num_hidden_layers) and writes them
     into the snapshot as model-mtp.safetensors (kept BF16),
  3. registers them in model.safetensors.index.json,
  4. extends the truncated per-layer lists in config.json from the original.

Idempotent (skips if MTP weights are already present). Run inside the vllm-node
container (needs torch + safetensors + huggingface_hub). Arg: snapshot dir.
"""
import json, os, sys
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

SNAP = sys.argv[1]
ORIG_REPO = os.environ.get("STEP37_ORIG_REPO", "stepfun-ai/Step-3.7-Flash")
MTP_FILE = "model-mtp.safetensors"


def layer_of(key):
    if ".layers." in key:
        try:
            return int(key.split(".layers.")[1].split(".")[0])
        except Exception:
            return -1
    return -1


def rewrite_json(name, obj):
    """Replace a (possibly symlinked) file in the snapshot with a real edited copy,
    so we never mutate the shared content-addressed blob."""
    p = os.path.join(SNAP, name)
    if os.path.islink(p) or os.path.exists(p):
        os.remove(p)
    json.dump(obj, open(p, "w"), indent=2)


idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))
wm = idx["weight_map"]
if any(layer_of(k) >= 45 for k in wm):
    print("[graft] MTP weights already present in snapshot; nothing to do")
    sys.exit(0)

# 1) locate + download the original MTP shard(s)
o_idx_path = hf_hub_download(ORIG_REPO, "model.safetensors.index.json")
o_wm = json.load(open(o_idx_path))["weight_map"]
mtp_keys = sorted(k for k in o_wm if layer_of(k) >= 45)
shards = sorted({o_wm[k] for k in mtp_keys})
print("[graft] MTP keys=%d shards=%s -> downloading from %s" % (len(mtp_keys), shards, ORIG_REPO))

tensors = {}
for sh in shards:
    shard_path = hf_hub_download(ORIG_REPO, sh)
    with safe_open(shard_path, framework="pt") as f:
        for k in mtp_keys:
            if o_wm[k] == sh:
                tensors[k] = f.get_tensor(k)  # BF16 in the original checkpoint

# 2) write the MTP shard into the snapshot
save_file(tensors, os.path.join(SNAP, MTP_FILE), metadata={"format": "pt"})

# 3) register in the index
nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
for k in mtp_keys:
    wm[k] = MTP_FILE
idx.setdefault("metadata", {})
idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + nbytes
rewrite_json("model.safetensors.index.json", idx)

# 4) extend per-layer config lists (layer_types, partial_rotary_factors, swiglu_*) to cover MTP layers
o_cfg = json.load(open(hf_hub_download(ORIG_REPO, "config.json")))
cfg = json.load(open(os.path.join(SNAP, "config.json")))


def extend_lists(node, onode):
    if not (isinstance(node, dict) and isinstance(onode, dict)):
        return
    for k, v in list(node.items()):
        ov = onode.get(k)
        if isinstance(v, list) and isinstance(ov, list) and len(ov) > len(v):
            node[k] = v + ov[len(v):]
        elif isinstance(v, dict) and isinstance(ov, dict):
            extend_lists(v, ov)


extend_lists(cfg, o_cfg)
rewrite_json("config.json", cfg)
print("[graft] grafted %d MTP tensors (%.2f GB), extended config lists -> %s"
      % (len(tensors), nbytes / 1e9, SNAP))
