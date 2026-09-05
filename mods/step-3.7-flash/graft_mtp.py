#!/usr/bin/env python3
"""Graft BF16 MTP (next-n predict) weights into a cached Step-3.7-Flash-NVFP4
snapshot so MTP speculative decoding works.

The official NVFP4 export ships no MTP weights (ModelOpt strips the next-n
predict layers -- those at index >= num_hidden_layers -- and truncates several
per-layer config lists during quantization). This script:
  1. downloads the MTP shard(s) from the original stepfun-ai/Step-3.7-Flash (BF16),
  2. extracts the MTP tensors (layer index >= num_hidden_layers) and writes them
     into the snapshot as model-mtp.safetensors (kept BF16),
  3. registers them in model.safetensors.index.json,
  4. extends the truncated per-layer lists in config.json from the original.

Idempotent (skips if MTP weights are already present). Run inside the vllm-node
container (needs torch + safetensors + huggingface_hub).

Usage: graft_mtp.py <nvfp4-snapshot-dir>
"""
import json, os, sys
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

if len(sys.argv) < 2:
    sys.exit("usage: graft_mtp.py <nvfp4-snapshot-dir>")
SNAP = sys.argv[1]
ORIG_REPO = os.environ.get("STEP37_ORIG_REPO", "stepfun-ai/Step-3.7-Flash")
MTP_FILE = "model-mtp.safetensors"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json_into_snapshot(name, obj):
    """Replace a (possibly symlinked) file in the snapshot with a real edited copy,
    so we never mutate the shared content-addressed blob."""
    p = os.path.join(SNAP, name)
    if os.path.islink(p) or os.path.exists(p):
        os.remove(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def num_hidden_layers(cfg):
    # Prefer the language-model sub-config (this is a multimodal config); the MTP
    # layers live in the LM, so its num_hidden_layers is the right cutoff.
    tc = cfg.get("text_config", {}) if isinstance(cfg, dict) else {}
    for node in (tc, cfg):
        if isinstance(node, dict) and "num_hidden_layers" in node:
            return int(node["num_hidden_layers"])
    raise ValueError("num_hidden_layers not found in config.json")


def layer_of(key):
    if ".layers." in key:
        try:
            return int(key.split(".layers.")[1].split(".")[0])
        except ValueError:
            return -1
    return -1


# MTP layers are those at index >= num_hidden_layers (the next-n predict blocks).
cfg = load_json(os.path.join(SNAP, "config.json"))
cutoff = num_hidden_layers(cfg)

idx = load_json(os.path.join(SNAP, "model.safetensors.index.json"))
wm = idx["weight_map"]
if any(layer_of(k) >= cutoff for k in wm):
    print("[graft] MTP weights already present in snapshot; nothing to do")
    sys.exit(0)

# 1) locate + download the original MTP shard(s)
o_wm = load_json(hf_hub_download(ORIG_REPO, "model.safetensors.index.json"))["weight_map"]
mtp_keys = sorted(k for k in o_wm if layer_of(k) >= cutoff)
if not mtp_keys:
    sys.exit("[graft] no MTP tensors (layer >= %d) found in %s" % (cutoff, ORIG_REPO))
shards = sorted({o_wm[k] for k in mtp_keys})
print("[graft] cutoff=%d  MTP keys=%d  shards=%s  (from %s)"
      % (cutoff, len(mtp_keys), shards, ORIG_REPO))

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
write_json_into_snapshot("model.safetensors.index.json", idx)

# 4) extend truncated per-layer config lists from the original config
o_cfg = load_json(hf_hub_download(ORIG_REPO, "config.json"))


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
write_json_into_snapshot("config.json", cfg)
print("[graft] grafted %d MTP tensors (%.2f GB), extended config lists -> %s"
      % (len(tensors), nbytes / 1e9, SNAP))
