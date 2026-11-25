# RadixArk Qwen DSpark compatibility

This runtime mod enables Qwen DSpark checkpoints such as
`RadixArk/Qwen3.8-27B-DSpark` on vLLM builds that already contain the Qwen3
DSpark model implementation.

Those checkpoints declare `architectures: ["DSparkDraftModel"]` and
`model_type: "qwen3"`. Older vLLM code treats every generic
`DSparkDraftModel` as a DeepSeek-V4 draft and rewrites the config to the wrong
loader. The patch detects this Qwen combination and normalizes its architecture
to `Qwen3DSparkModel` before model resolution.

Without that normalization, the DeepSeek fallback can also copy the NVFP4
target's quantization setting onto the unquantized BF16 draft. The draft then
reaches `get_quant_config()` with a callable `hf_overrides` value and fails with
`ValueError: hf_overrides must be a dict`. Routing the checkpoint before that
fallback keeps the draft unquantized and fixes the underlying configuration
error rather than weakening quantization validation globally.

Apply the mod before `exec`, for example:

```bash
./launch-cluster.sh --solo \
  --apply-mod mods/radixark-dspark \
  exec vllm serve RadixArk/Qwen3.8-27B-NVFP4 \
    --speculative-config \
      '{"method":"dspark","model":"RadixArk/Qwen3.8-27B-DSpark","num_speculative_tokens":7}'
```

The mod is idempotent and becomes a no-op when the installed vLLM already has
equivalent Qwen draft normalization. It intentionally fails without modifying
the source when the installed speculative-config implementation does not match
a supported layout.
