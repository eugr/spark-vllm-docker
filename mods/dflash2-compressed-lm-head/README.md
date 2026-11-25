# DFlash2 compressed-tensors LM head

This runtime mod extends the target-head type check in vLLM PR
[#52816](https://github.com/vllm-project/vllm/pull/52816) to accept
`CompressedTensorsLinearMethod`. It does not dequantize or replace the target
head: DFlash2 continues to call the head's existing `quant_method.apply()` for
candidate TopK.

The mod fails closed if the expected DFlash2 implementation is absent or has
changed, and it is safe to apply more than once.
