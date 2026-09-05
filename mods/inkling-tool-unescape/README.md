# Inkling tool-call HTML-entity unescape mod

This opt-in mod repairs a sporadic Inkling tool-calling failure where the model
**samples an HTML entity in place of a structural JSON character** — most often
`&quot;` where the closing `"` of a string value belongs, seen most frequently
right after a URL (a bias from HTML-heavy pretraining, e.g. `href="..."`).

Example of the raw model output that fails:

```text
<|content_invoke_tool_json|>{"name":"fetch_url","args":{"url":"https://…/23914f4c48ba&quot;}}<|end_message|>
```

## Why it breaks

`vllm/parser/inkling.py` extracts tool arguments with a hand-rolled,
string/escape-aware scanner (not `json.loads`, because the engine diffs
successive partial outputs for streaming and needs prefix-stable substrings).
That scanner terminates a JSON string only on a literal `"`. `&quot;` contains
no `"`, so the string never closes, `args` span extraction fails, and vLLM
falls back to emitting the raw `<|content_invoke_tool_json|>…` text as message
content instead of a parsed tool call.

This is a **generation artifact, not a rendering bug**: the chat template does
not HTML-escape (it uses `tojson`, which never emits `&quot;`). Raising or
lowering sampling temperature does not remove it — Thinking Machines' own
recommended settings are `temperature 1.0, top_p 1.0`, which if anything make
the rare entity token marginally *more* likely, so a parser-level repair is the
correct fix.

## What it does

Patches `_inkling_arg_converter` to unescape **complete** JSON-structural HTML
entities (`&quot; &amp; &lt; &gt; &apos;` and the numeric `&#34; &#39; &#x22;
&#x27;`) at the very top of the converter — before the scanner runs. It is
**prefix-stable**, which the engine's argument-delta diffing requires: while
streaming, a trailing possibly-incomplete entity is withheld until it completes,
so each tick's output only ever extends the previous one. Verified against the
malformed case, clean calls, multi-arg calls, and streaming prefixes.

The patch is idempotent and refuses to apply if `_inkling_arg_converter`'s shape
has changed (it anchors on the function signature and the unique
`span = _args_value_span(raw_args)` body line), so a future vLLM that reworks the
parser fails loudly rather than silently mis-patching.

## Scope and limitations

- Only the tool-**args** region is repaired (the observed failure). A corrupted
  quote in the tool **name** is not separately handled, though unescaping the
  whole wrapper first also fixes name-region entities for args-span scanning.
- Unescaping is unconditional for the whitelisted entities, so a tool argument
  that *legitimately* contains the literal text `&amp;` would be turned into `&`.
  This mirrors the reality that a well-behaved model should emit the literal
  character, not the entity; if a workload genuinely needs literal entity text
  in tool args, do not apply this mod.
- This is a downstream band-aid. Remove it once vLLM's Inkling parser (or the
  model's chat template / rendering) handles the entity artifact upstream.

## Usage

Applied automatically by the recipe (`recipes/inkling-small-nvfp4-tuned.yaml`),
or manually:

```bash
./launch-cluster.sh --apply-mod mods/inkling-tool-unescape exec vllm serve /model ...
```

The launcher applies it inside each node's container. It requires an
Inkling-capable image (the `vllm-node-inkling` tag); it exits with an error on
an image whose vLLM has no Inkling tool parser.
