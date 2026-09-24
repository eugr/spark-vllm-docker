# Qwen3-Next AutoRound router compatibility

Intel/Qwen3-Coder-Next-int4-AutoRound includes quantized MoE router parameters
such as `layers.0.mlp.gate.qweight`. The router must use the model's
`quant_config` to create those parameters instead of an unquantized `weight`.

Upstream [PR #35156](https://github.com/vllm-project/vllm/pull/35156) forced the
router to remain unquantized as a workaround for NVFP4 checkpoints. This mod
previously reversed that PR at launch. [PR #58234](https://github.com/vllm-project/vllm/pull/58234)
then replaced `ReplicatedLinear` with `GateLinear`, removing the lines matched
by the reverse patch. The old mod skipped that failure and allowed startup to
continue with incompatible router parameters.

The mod now edits only `Qwen3NextSparseMoeBlock.__init__` in `qwen3_next.py`:

- Pass the model's `quant_config` to the router on both `ReplicatedLinear` and
  `GateLinear` versions. Keep the constructor and all other layers unchanged.
- Recognize repeat application by inspecting the actual router configuration.
- Stop with a nonzero status on an unsupported source layout, before serving.

It requires no network access. The active Python interpreter locates vLLM
without importing it. `VLLM_PACKAGE_ROOT` can override the package path for
offline tests. Keep this mod opt-in for the AutoRound recipe; other checkpoints
may rely on upstream's unquantized-router workaround.

Run the CPU-only regression tests from the repository root:

```bash
python3 tests/test_qwen3_next_autoround_mod.py
```
