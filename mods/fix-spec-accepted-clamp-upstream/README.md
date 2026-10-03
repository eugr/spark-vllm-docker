# fix-spec-accepted-clamp-upstream

Prevents a speculative-decoding assertion from stopping `EngineCore` when guided decoding invalidates draft tokens. The sampler counts accepted tokens against **scheduled** drafts, while the assertion compares them with the smaller **valid** draft count.

The patch keeps rollback arithmetic unchanged. In the `SOFT` case it widens only the draft count sent to metrics. In the impossible `HARD` case, accepted tokens exceed scheduled drafts; it clamps that count to avoid a negative rollback. Each branch logs with the unchanged `[SPEC-ACCEPTED-CLAMP-UPSTREAM]` prefix.

## Use

Add `mods/fix-spec-accepted-clamp-upstream` to the recipe's `mods:` list. The runner calls `patch_spec_accepted_clamp.py` against the installed vLLM package and compiles the patched scheduler. No environment variable is needed. To check anchors without writing:

```bash
python3 patch_spec_accepted_clamp.py \
  --vllm-root /usr/local/lib/python3.12/dist-packages/vllm --check
```

## Measurements and scope

A guided-decoding assertion failure was observed in a 2x DGX Spark TP2 setup with MTP. **No before/after throughput measurement is available.** Predicted overhead is approximately **0 ms/step** (two integer comparisons per request); decode throughput and acceptance length should remain unchanged on the normal path. An absent `SOFT`/`HARD` log means neither recovery branch ran.

The patch uses exact scheduler anchors and fails if they have changed. It also adjusts the metrics input because downstream assertions and histogram indexing use the same counts.

## Rollback

Remove the mod from the recipe and recreate the container from the unmodified image.
