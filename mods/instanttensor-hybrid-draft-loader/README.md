# InstantTensor hybrid speculative-draft loader

This opt-in mod keeps InstantTensor for the primary model while allowing a
speculative draft model to use lazy safetensors. It avoids a second
InstantTensor GPU-streaming pass over a large checkpoint when an embedded MTP
or other same-checkpoint draft needs only a small subset of its tensors.

The mod patches vLLM's central `get_model()` entry point. It identifies a real
speculative draft by comparing the `ModelConfig` object with
`speculative_config.draft_model_config`; target and unrelated model loads are
left unchanged.

## Modes

Set `INSTANTTENSOR_DRAFT_LOADER` on the containers:

- `auto` (default): use lazy safetensors only when the draft and target have
  the same model path and revision.
- `safetensors`: use lazy safetensors for every speculative draft whose
  effective loader would otherwise be InstantTensor.
- `instanttensor`: preserve InstantTensor for drafts. This disables the hybrid
  behavior without removing the mod.

If the primary/effective loader is not InstantTensor, the mod does nothing.
An explicitly configured non-InstantTensor draft loader is also preserved.

## Usage

The default `auto` mode is appropriate for embedded MTP weights:

```bash
./launch-cluster.sh \
  --apply-mod mods/instanttensor-hybrid-draft-loader \
  exec \
  vllm serve /model \
    --load-format instanttensor \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    ...
```

To force lazy safetensors for a standalone speculative draft too:

```bash
./launch-cluster.sh \
  --apply-mod mods/instanttensor-hybrid-draft-loader \
  -e INSTANTTENSOR_DRAFT_LOADER=safetensors \
  exec \
  vllm serve ...
```

At runtime, a switched draft logs:

```text
Hybrid draft loading: using lazy safetensors for speculative draft weights
while preserving InstantTensor for the target model
```

## Scope and limitations

This is a vLLM integration workaround, not a selective-loading implementation
inside InstantTensor. Lazy safetensors still scans checkpoint metadata, but
weights rejected by the draft model remain CPU memory-mapped views instead of
being staged and cloned on the GPU.

Remove the mod once vLLM exposes a supported per-draft load-format option or
InstantTensor and vLLM can plan and selectively stream only requested tensor
names.
