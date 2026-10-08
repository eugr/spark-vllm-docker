# GLM-5.3 NVIDIA thinking template

This mod generates an external chat template for `nvidia/GLM-5.3-Flash-NVFP4`. The checkpoint template always ends an assistant generation prompt with `<|assistant|><think>`, even when a request sets `enable_thinking=false`. With the `glm45` reasoning parser, that request was observed to report `reasoning_tokens: 0` while reasoning appeared in `content` on a 2x DGX Spark TP2 setup.

The generated template changes only that prefix: explicit thinking-off requests use the GLM-4.7 closing-only prefix. Default and thinking-on rendering remain byte-identical to the checkpoint template. The checkpoint itself is untouched. The patch is anchored to the 257-line NVIDIA template; the corresponding zai-org template has 261 lines.

## Use

Add this mod to the recipe's `mods` list and pass `--chat-template /tmp/glm53-nvidia-chat-template.jinja` to `vllm serve`. To request thinking-off, set `chat_template_kwargs: {"enable_thinking": false}`. Run `python3 patch.py --check` to verify the expected template hashes without writing the output.

## Results and limits

The observed baseline was `reasoning_tokens: 0` with reasoning in `content` for an explicit thinking-off request. No throughput or quality improvement has been measured. GLM-5.3 does not document a thinking-off mode, so the closing-only prefix follows GLM-4.7 behavior and needs output validation for the intended workload.

## Rollback

Remove the mod from `mods` and remove `--chat-template`, then recreate the service container from the original recipe.
