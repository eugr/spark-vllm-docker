# InstantTensor zero-copy weight loader

This experimental, opt-in mod changes vLLM's InstantTensor weights iterator
from `copy=True` to `copy=False`. InstantTensor then yields views into its GPU
ring buffer instead of cloning every complete checkpoint tensor into a second
CUDA allocation.

For checkpoints with very large individual tensors, this can remove a transient
allocation equal to the largest tensor. It does **not** remove InstantTensor's
ring buffer: the ring is still automatically enlarged to at least the largest
tensor in the checkpoint.

## Usage

```bash
./launch-cluster.sh --solo \
  --apply-mod mods/instanttensor-zero-copy \
  exec \
  vllm serve /model \
    --load-format instanttensor \
    ...
```

The same mod works in cluster mode. It changes only
`instanttensor_weights_iterator`; safetensors, fastsafetensors, and other load
formats are untouched.

It may be combined with `mods/instanttensor-hybrid-draft-loader`. In that case,
the primary model uses zero-copy InstantTensor while any draft switched by the
hybrid mod uses lazy safetensors.

## Safety boundary

InstantTensor reuses the ring after the consumer advances to the next tensor.
Zero-copy is therefore safe only when the model-specific vLLM weight loader
fully consumes or copies each yielded weight inline and does not retain a view
for later processing. Most conventional vLLM weight loaders copy into an
existing parameter immediately, but this is not guaranteed for every model,
quantization method, or future vLLM version.

The launcher applies the mod once to a fresh container. The patcher uses Python's
syntax tree instead of a fixed textual diff, so harmless changes to nearby code,
comments, keyword ordering, and line numbers do not prevent application. It
refuses to write unless it finds exactly one
`instanttensor.safe_open(..., copy=True)` call inside exactly one
`instanttensor_weights_iterator` function, with `copy` still expressed as a
literal boolean. It parses the result again after writing. If upstream vLLM
already uses literal `copy=False`, the mod exits successfully without changing
the file because the requested behavior is already native.

Validate a zero-copy load against the same model loaded with lazy safetensors.
At minimum, compare deterministic output tokens or logits from several prompts.
Remove the mod immediately if results diverge or startup reports missing or
unexpected weights.

## Limitations

- The minimum InstantTensor device buffer remains the largest checkpoint tensor.
- This does not add sliced, rank-local, or direct-to-parameter I/O.
- A model loader that retains a yielded tensor can observe overwritten or freed
  storage, producing incorrect weights without a clean failure.
- Mods are applied to fresh containers and do not persist across launches;
  omitting this mod leaves the next container's installed vLLM untouched.
