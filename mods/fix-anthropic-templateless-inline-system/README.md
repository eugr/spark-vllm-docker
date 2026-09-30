# fix-anthropic-templateless-inline-system

**A model with no jinja chat template gets its Anthropic inline `system`
messages hoisted to offset 0, which destroys the prefix cache on every turn.**
For Claude Code against DeepSeek-V4-Flash this took the prefix cache hit rate to
effectively zero on long conversations — minutes of re-prefill per turn.

One line: `if not chat_template: return True` becomes `return False`.

## Who is affected

Only models with **no jinja template at all** — `tokenizer.chat_template` is
`None` and no `--chat-template` is passed, because the prompt is rendered by a
Python encoder selected with `--tokenizer-mode`. `deepseek_v4` is the case this
was found on. Templated models take the existing render-probe path and are
**bit-for-bit unaffected**.

## What goes wrong

`AnthropicServingMessages._detect_merge_inline_system` decides whether to strip
`role: system` messages out of the messages array and concatenate them onto the
leading system block. It decides by rendering a `[system, user, system, user]`
probe against the jinja template. With no template there is nothing to probe, so
it returns its documented conservative default: merge.

Claude Code (>= 2.1.154) sends a per-turn system reminder inline in `messages`.
Merging puts each new one at offset ~0. **New reminder each turn → new byte 0 →
nothing after it is reusable.** On a 200k-token conversation that is the entire
prompt, every turn.

Measured on a synthetic 4-turn conversation through the real encoder:

| | shared prefix between consecutive turns |
|---|---|
| stock (`merge=True`) | 4198 chars — **exactly the static system prompt, nothing else** |
| patched (`merge=False`) | **100% of the previous turn** |

On a live 2-node DeepSeek-V4-Flash cluster, applying this took the prefix cache
hit rate from ~78% (and effectively 0 for long Claude Code conversations) to
**98.3%**, and prompt throughput from 1600–2400 tok/s of real prefill work down
to ~80.

The same file already fights this exact battle two functions down, where it
drops Claude Code's `x-anthropic-billing-header` because it "contains a
per-request hash that defeats prefix caching". The merge path reintroduces that
wholesale, with the entire reminder text.

## Why "no template" should mean *don't* merge

A templateless model is not an unknown quantity — it is rendered by a Python
encoder chosen with `--tokenizer-mode`, and those encoders accept inline system
messages at their original position. Merging is the destructive answer, and it
should not be the fallback for "I could not check".

`VLLM_ANTHROPIC_MERGE_INLINE_SYSTEM=1` restores stock behaviour without a
rebuild, for a templateless model that genuinely needs merging.

> **The override is unconditional in both directions** — it short-circuits the
> detector before the template is examined at all. Setting it to `1` is always
> safe (it is the stock default). Setting it to `0` on a model whose template
> *requires* system-first ordering (Qwen's `loop.first` guard) will make
> rendering raise and requests fail with 400. Only set `0` if you know the
> model tolerates mid-conversation system messages, and prefer simply not
> setting the variable — the detector already gets that case right.

## Relationship to upstream

- **#44283** (merged) introduced the merge — the original regression.
- **#44602** (merged) preserved inline position for prefix caching.
- **#46025** (merged) added `_detect_merge_inline_system` and chose
  "no template → conservative default: merge". That default is what this mod
  changes.
- **#46196** (open) resolves the *model's* chat template before calling the
  detector, instead of passing the raw `--chat-template` arg. It fixes a
  neighbouring case and is complementary — but it does **not** help here,
  because a `--tokenizer-mode` model has no jinja template to resolve. The
  detector still receives `None` and still returns `True`.
- **#52978** (open) hoists *trailing* system messages to fix tool-calling
  breakage (#48874). If merged, that path would re-hoist Claude Code's per-turn
  reminder and undo the prefix-cache win this mod recovers, so the two want
  reconciling.

If #46025's default is ever changed upstream, this mod becomes a no-op and
`git apply --reverse --check` will make `run.sh` skip cleanly.

## Not included: the generation-prompt half

This was originally found alongside a second DeepSeek-V4 defect — a conversation
ending in a `system` message got no `<|Assistant|>` generation prompt, so the
model emitted an immediate EOS. **That is now fixed upstream** in
`vllm/tokenizers/deepseek_v4_encoding.py`, with a tighter condition than the
local patch used, so it is deliberately not shipped here.

Worth knowing if you are on an older vLLM: on a build that lacks the upstream
encoder fix, **this mod alone makes things worse**. It stops the system message
being hoisted, which means it now actually reaches the encoder — where the EOS
bug lives. Check for the `role == "system"` branch in that file's
generation-prompt logic before applying.

## Verification

- Applied and served on a 2-node TP=2 DeepSeek-V4-Flash-0731 cluster
  (2x DGX Spark GB10) for several weeks, driving Claude Code as the client.
- Cache hit rate and prefill throughput as tabulated above, read from
  `vllm:gpu_prefix_cache_hit_rate` and `vllm:prompt_tokens_total`.
- Templated models: unaffected by inspection — the patch only changes the
  `not chat_template` branch and adds an env override ahead of it.
