# fix-glm47-phantom-fallback

Rescues text swallowed by the GLM-4.7 (glm47) tool-call parser when the model
writes a stray tool-call opener inside prose and never closes it: the phantom
"tool name" (the prose after the opener) is returned to the client as regular
message content instead of being dropped or turned into a bogus tool call.

## Problem

When the model emits a tool-call opener in plain text (quoting or imitating
the call syntax), the parser enters the `TOOL_NAME` state and accumulates the
following prose as a tool name. Depending on what follows:

- **A later real tool call** (with `fix-glm47-stray-tool-start` applied): the
  stray call is closed by the recovery transition and parsing continues —
  but the phantom text itself is still dropped.
- **No later tool call** (streaming): the phantom slot is never closed and
  the accumulated prose is silently dropped at finish.
- **No later tool call** (non-streaming, with
  `fix-glm47-unknown-tool-name` applied): the prose name passes name
  validation and becomes a bogus `ToolCall(name=<prose>, arguments="{}")`.

This mod makes the parser recognize prose-like names (anything that cannot
match a real tool name) and move that text into the parser's deferred-content
channel, which the engine emits as normal message content.

## Scope

- Only `vllm/parser/glm47_moe.py` (the `Glm47MoeParser` subclass); the engine
  (`parser_engine.py`) is unchanged.
- A phantom name is a non-empty name that does not match
  `^[A-Za-z0-9_.\-]{1,128}$` (real OpenAI-style tool names are
  `[A-Za-z0-9_-]{1,64}`). This preserves
  `fix-glm47-unknown-tool-name` behavior: unknown-but-well-formed names are
  still delivered as tool calls.
- Rescued text is emitted verbatim (the prose exactly as the model wrote it
  after the opener; the opener itself is consumed as parser structure).
- Requires `fix-glm47-unknown-tool-name` and `fix-glm47-stray-tool-start`
  (apply them together); the hunks touch disjoint regions of the file and
  the three mods apply in any order.
- Known limitations:
  - Streaming: a phantom slot consumes one tool-call index, so a later real
    call arrives with a sparse `index` (clients merging by index handle this;
    documented upstream behavior).
  - Single-word prose names that happen to look like identifiers (e.g. one
    Latin word) still pass name validation and can become a tool call.
  - `tool_choice="none"` suppresses all tool events, so there is nothing to
    rescue (pre-existing semantics).
- Idempotent: skips when the `_rescue_phantom_slot` helper is already present.
- `VLLM_PACKAGE_ROOT` overrides the vLLM package location (used by the test).

## Usage

The mod is wired into the `glm-5.3-flash` recipe's `mods:` list (its two
companion mods are wired into the same recipe by #462 and the
stray-tool-start PR), so the plain `./run-recipe.sh glm-5.3-flash` applies
it automatically. To opt out, remove the `mods/fix-glm47-phantom-fallback`
line from the recipe's `mods:` list.

To apply on demand instead (or after opting out), together with its two
companion mods, which it requires:

```bash
./run-recipe.sh glm-5.3-flash \
  --apply-mod mods/fix-glm47-unknown-tool-name \
  --apply-mod mods/fix-glm47-stray-tool-start \
  --apply-mod mods/fix-glm47-phantom-fallback
```

Both forms are safe together — all three mods are idempotent and
re-applying is a no-op. A vLLM restart is required for an already running
container.

## Status

Experimental (vllm-project/vllm#60703 follow-up; issue #465): the mod
addresses the "prose tail with no subsequent tool call" limitation documented
in `fix-glm47-stray-tool-start`. No upstream counterpart exists — the
prose-name heuristic is our own design. Wired into the `glm-5.3-flash`
recipe (remove the mod from the recipe's `mods:` list to opt out); adopt or
remove based on field experience.
