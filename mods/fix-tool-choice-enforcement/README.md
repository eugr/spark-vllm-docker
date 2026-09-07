# Tool Choice Enforcement Fix

**Last updated:** `2026-09-07T16:05:00-05:00`

This runtime-only mod restores `tool_choice` enforcement on
`/v1/chat/completions`. Without it, `tool_choice="required"` and named tool
choice decode free-form: the model answers in prose and no tool call is
produced.

Verified against `vllm-node-b12x` builds `0.1.dev20489+ga50ebee1d.d20260904`
and `0.1.dev20596+g2a979314d.d20260907`. Both are affected.

## Symptom

Serving GLM-5.3-Flash with `--reasoning-parser glm45 --tool-call-parser glm47
--enable-auto-tool-choice`, against a system prompt that discourages tool use:

```text
tool_choice="required"                            -> content "Paris",
                                                     tool_calls null,
                                                     finish_reason "tool_calls"
tool_choice={"type":"function",
             "function":{"name":"get_weather"}}   -> content "Paris",
                                                     tool_calls null,
                                                     finish_reason "stop"
```

The first response is not merely a miss, it is malformed. `finish_reason`
claims a tool call that is not in the payload, because the serving layer forces
that value whenever `tool_choice` is `"required"` and then finds nothing to
report. Any OpenAI-compatible client that branches on `finish_reason` breaks on
it.

## Cause

Enforcement for a GLM tool choice is an xgrammar structural tag. It constrains
decoding into the model's native `<tool_call>name<arg_key>...` syntax, and the
ordinary auto-style extractor then reads the call back out.

The extraction half already works. `Glm47MoeModelToolParser` sets
`structural_tag_model` to `"glm_4_7"`, which makes
`AbstractToolParser.__init_subclass__` clear `supports_required_and_named`,
which routes required and named choices through the auto extraction path on the
way out. It finds nothing only because nothing constrained the decode.

The constraint half is broken twice over.

1. **Nothing on the chat-completions path calls `Parser.adjust_request`**, which
   is where the tag is built. `grep -rn adjust_request vllm/entrypoints/` finds
   exactly one call site, in the *harmony* branch of the Responses API, beside a
   TODO about unifying it with the non-harmony branch.

2. **A collapsed parser engine never builds the tag.**
   `DelegatingParser.adjust_request` is the only implementation that calls
   `_apply_structural_tag`. When the reasoning and tool adapters are backed by
   the same engine, which `glm45` + `glm47` are, `ParserManager.get_parser`
   collapses them and returns the engine class itself. `ParserEngine` overrides
   `adjust_request` with a two-line body that sets `skip_special_tokens` and
   returns, and neither it nor `Glm47MoeParser` carries a
   `structural_tag_model`. On that path there is no structural-tag code at all.

   This second break is specific to the builds this mod targets. The collapse
   is the `_get_parser_engine_cls` branch in `ParserManager.get_parser`, present
   in the B12X fork at both verified commits. Stock `vllm-project/vllm` has no
   such branch: `get_parser` always returns a `DelegatingParser` subclass, so
   part 1 alone is enough there and part 2 is applied but never reached. Check
   `grep -n _get_parser_engine_cls vllm/parser/parser_manager.py` in the image
   to tell the two apart.

On the collapsing builds, fixing either half alone changes nothing. Part 1 was
applied on its own and both probes above returned byte-identical output. Worth
knowing before bisecting this a second time.

## What the mod changes

**Part 1, model-agnostic.** Calls `parser.adjust_request(request)` in
`_create_chat_completion`, immediately after the parser is built and before the
request is rendered. That is where the Responses API makes the call, and where
the tag has to land to reach `to_sampling_params`. Skipped when the installed
vLLM already calls `adjust_request` on this path.

**Part 2, GLM only.** Gives `Glm47MoeParser` an `adjust_request` that keeps the
engine's own behaviour and then applies the tag, mirroring
`DelegatingParser._apply_structural_tag` rather than inventing a second policy.
It reproduces upstream's own predicate and arguments, including the
`VLLM_ENFORCE_STRICT_TOOL_CALLING` check that `AbstractToolParser.get_structural_tag`
applies before building a tag. Skipped when `vllm/parser/glm47_moe.py` is not
installed, so the mod is safe to layer onto non-GLM recipes for the part 1 fix
alone, and inert on builds that do not collapse the parser pair.

`reasoning=False` is upstream's argument and is correct here. The structured
output manager already defers the grammar until the reasoning parser reports
`</think>`, so a tag carrying its own reasoning prefix would double-count the
think block.

## Scope and blast radius

The tag builder returns `None` for `tool_choice: "none"`, and for `"auto"`
unless a tool declares `strict`. Ordinary auto traffic therefore decodes exactly
as before. Confirmed live: with the mod applied, an auto request still calls the
tool when the question needs one, still answers directly when it does not, and
the reasoning block is intact in both cases.

Part 1 has one further effect worth stating. `ParserEngine.adjust_request` sets
`skip_special_tokens = False`, which upstream intends but which never took
effect while the call was missing. Checked for leakage after the change: no
special tokens appear in `content` or `reasoning` in either the thinking or
non-thinking path.

`VLLM_ENFORCE_STRICT_TOOL_CALLING=0` disables the whole path, as it does
upstream.

## Usage

```bash
./run-recipe.sh --apply-mod mods/fix-tool-choice-enforcement recipes/glm-5.3-flash.yaml
```

Or declare it in a recipe:

```yaml
mods:
  - mods/fix-tool-choice-enforcement
```

To verify after launch, send a request whose answer plainly needs no tool:

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "aegis",
  "messages": [
    {"role": "system", "content": "Use a tool ONLY when strictly necessary."},
    {"role": "user", "content": "What is the capital of France? Answer in one word."}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
  "tool_choice": "required"}' | jq '.choices[0].message.tool_calls'
```

Unpatched this returns `null` with `"Paris"` in `content`. Patched it returns a
`get_weather` call.

## Validation

Measured on a 4x DGX Spark TP4 cluster, `tool-eval-bench` v2.6.1 hardmode, 88
scenarios, GLM-5.3-Flash EXL3 6bpw, `0.1.dev20489`:

| | points / 176 | score | TC-45 |
| --- | ---: | ---: | :---: |
| Before | 159 | 90 | fail |
| After, trial 1 | 161 | 91 | pass |
| After, trial 2 | 160 | 91 | pass |
| After, trial 3 | 161 | 91 | pass |

Scenario TC-45 is `tool_choice=required Compliance`. Diffing the before and
after runs scenario by scenario, ten scenarios moved and the nine besides TC-45
cancel to exactly zero, so the entire net gain is this fix.

The same comparison on EXL3 4bpw moved 154 to 158 points, 88 to 90, with TC-45
flipping from fail to pass.

## TC-45 tracks the build, not the checkpoint

Across 17 scored `tool-eval-bench` hardmode runs since 2026-09-04, TC-45
follows the engine build rather than the quantization or the checkpoint:

| build and checkpoint | TC-45 | runs |
| --- | :---: | ---: |
| `0.1.dev20051`, EXL3 6bpw | pass | 2 of 2 |
| `0.26.1rc1.dev1006`, EXL3 4bpw | pass | 1 of 1 |
| `0.1.dev20489`, EXL3 6bpw and 4bpw, no mod | fail | 10 of 10 |
| `0.1.dev20489`, EXL3, with this mod | pass | 3 of 3 |
| `0.1.dev20596`, NVFP4-Spark, no mod | fail | 1 of 1 |

The last row matters most. It is a different checkpoint, a different
quantization and a newer build, and it fails the same way, which rules out the
EXL3 checkpoints as the cause.

Prefer the curl probe above when you need certainty about a specific build.
TC-45 is a behavioural scenario: it exposes the defect by asking a question the
model can answer without a tool, so it depends on the model declining. The
probe's system prompt makes that explicit, so its answer must come from an
enforced call or not at all. Both agree on every build measured so far.
