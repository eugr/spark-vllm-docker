# fix-glm53-mm-model-path

Fixes the GLM-5-Next **multimodal** boot failure on the stock
`vllm/vllm-openai:glm53-flash-arm64-cu130` image when `--language-model-only` is
removed. Not needed for text-only.

## Symptom

```
FileNotFoundError: [Errno 2] No such file or directory:
    'zai-org/GLM-5.3-Flash/processor_config.json'
  at vllm/models/glm5next/nvidia/multimodal.py:658 -> get_hf_processor
  at vllm/transformers_utils/processors/glm5next.py:853 -> from_pretrained
```

## Why

The multidimensional path calls
`Glm5NextProcessor.from_pretrained(self.ctx.model_config.model)` with the raw HF
id (`zai-org/GLM-5.3-Flash`). `AutoTokenizer` and the `get_image_processor_config`
helper resolve the id via the hub, but the `video_processor` read does a bare
`open(os.path.join(model_path, "processor_config.json"))` that never turns the id
into a filesystem path.

## Fix

This mod patches `vllm/transformers_utils/processors/glm5next.py` `from_pretrained`
to resolve the id to the local cached snapshot directory (via
`huggingface_hub.try_to_load_from_cache` + `os.path.dirname`) when the arg isn't
already a directory. We deliberately avoid `snapshot_download(local_files_only=True)`,
which rejects a snapshot missing metadata files (`.gitattributes`/`LICENSE/README.md`)
— this image's cache is that incomplete state, though `processor_config.json` itself
is present.

## Notes

- Env-gated by removing `--language-model-only`; harmless when multimodal is off
  (text path never constructs the HF processor).
- Idempotent (marker-string check) and `py_compile`-verified. Same mod framework as
  `mods/fix-glm53-nope-rope-pad`.
