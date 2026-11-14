# fix-glm47-unknown-tool-name

Relaxes the GLM-4.7 (glm47) tool-call parser so that a model-emitted tool call
whose function name is **not** in the request's `tools` list is returned to the
client as a tool call instead of being silently discarded.

## Problem

`vllm/parser/glm47_moe.py` sets `validate_tool_names=True`. When the model
emits a tool call for an unknown tool name (common in long agentic sessions
where the model reconstructs tool names from history), the parser drops the
whole call. Because the tool-call markers are consumed as parser structure,
the message comes back with `content=null`, `reasoning=null`,
`tool_calls=null` and `finish_reason="stop"` — an empty response. Clients
that retry on empty responses then loop forever (observed with
vllm-copilot on Windows against GLM-5.3-Flash-FP4-Local; reproduced
server-side with a captured request at temperature 0).

With validation disabled the same generation yields a proper
`finish_reason="tool_calls"` response, the client rejects the unknown tool
and the conversation continues normally.

## Scope

- Only flips `validate_tool_names` in `vllm/parser/glm47_moe.py`.
- The `minimax_m2` parser uses the same setting but is intentionally left
  unchanged.
- The patch uses a context-free hunk so it applies across recent vLLM builds
  where the surrounding lines have drifted.
- The mod is idempotent: it skips when the flag is already `False`.
- `VLLM_PACKAGE_ROOT` overrides the vLLM package location (used by the test).

## Usage

```bash
./run-recipe.sh glm-5.3-flash --solo --apply-mod mods/fix-glm47-unknown-tool-name
```

Applied at container start; a vLLM restart is required for an already
running container.
