# DSpark InstantTensor checkpoint filter

This runtime mod fixes pathological second-pass checkpoint loading for DSpark
draft models whose weights are embedded under `mtp.*` in the target checkpoint.

Affected loaders discard non-MTP tensors inside `load_weights()`, which is too
late for eager streaming loaders such as InstantTensor: the full target tensor
has already been read and transferred. The mod adds
`checkpoint_weight_name_prefixes = ("mtp.",)` to the compatible DSpark model
class so vLLM filters both checkpoint shards and tensor names before I/O.

The patch is discovered by source structure rather than an exact file path,
Python version, vLLM version, or repository. It intentionally does not patch
DSpark implementations that load a separate draft checkpoint.

Apply it before `exec`:

```bash
./launch-cluster.sh \
  -t vllm-node-b12x \
  --apply-mod mods/dspark-instanttensor \
  exec vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark ...
```

The mod is idempotent. If the installed implementation already defines a
checkpoint-name filter, it reports that support and makes no change.
