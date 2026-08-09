# fix-dsv4-effort-levels

Restores three working `reasoning_effort` levels to vLLM's vendored DeepSeek-V4
tokenizer.

## Why

The image pins a vLLM commit that predates the effort-level fix now in vLLM
`main`. On the pinned copy of `vllm/tokenizers/deepseek_v4_encoding.py` there is
a single `REASONING_EFFORT_MAX` constant, emitted only when the normalized
effort equals `'max'`, and `deepseek_v4.py`'s normalizer folds `low` into `high`.
The net effect through `--tokenizer-mode deepseek_v4`:

| request effort | prefix the model receives | effective tier |
|---|---|---|
| `low`  | none | low |
| `high` | none | **low** (collapsed) |
| `max`  | the paragraph the model's own file labels `high` | **high** (shifted) |
| —      | — | the model's true `max` is **unreachable** |

## What it changes

Copies vLLM `main`'s fix, verbatim, into the effort path only:

- `deepseek_v4_encoding.py`: `REASONING_EFFORT_MAX` constant → the three-entry
  `REASONING_EFFORT_PROMPTS` dict + `DEFAULT_REASONING_EFFORT`; and the
  `== 'max'`-only emit condition → the all-levels condition.
- `deepseek_v4.py`: **one added `elif`** so `low`/`minimal`/`medium` map to
  `"low"` instead of falling through to `"high"`. No existing line changes; the
  thinking on/off logic is left exactly as pinned.

After the mod, `low`/`high`/`max` render 51 / 527 / 577-char prefixes — identical
to the model's own reference encoder.

Both target files are stock vLLM (SPDX: vLLM project); neither carries any
b12x / fork code, so nothing else in the build is affected.

## Safety

Anchor-matched single-occurrence replacement; aborts if any anchor is not found
exactly once. Validates the result with `ast.parse` + `compile`. Idempotent
(re-running is a no-op). Atomic write (temp file + rename). Applies at container
launch, so it survives image rebuilds by being re-applied, never by mutating
already-patched state.

## Use

Not wired into any eugr-tracked recipe. Activate per launch:

```
run-recipe.py deepseek-v4-flash-0731 --apply-mod mods/fix-dsv4-effort-levels
```
