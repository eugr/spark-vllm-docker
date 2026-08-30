# mia-backports-20260830

Two correctness patches vendored from
`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` `overlay/` (MIT).

Both `.py` files are **bit-exact** copies of Mia's originals (verified
2026-08-30 against repo HEAD `688b7ab61d54` via `cmp`). Do not edit them
here; re-vendor from upstream instead. `run.sh` is this fork's wrapper.

| File | Upstream last-touch commit | Upstream subject |
| --- | --- | --- |
| patch_kpool_tail_slotmap.py | `e2274d88b3e3` (2026-08-30) | Pin K-pool tail slot mapping to the one-block circular scratch. |
| patch_xgrammar_termination.py | `8c3d4956ac34` (2026-08-29) | Backport speculative reasoning grammar validation |

What they fix (details in each file's docstring):

- **kpool tail slot-map clamp** — the generic paged slot-mapping kernel
  indexes past the tail spec's one-block table on generations past
  ~block_size tokens, silently corrupting the shared pool. Mechanism
  credited to vcruz305.
- **XGrammar termination** — source-exact backports of vLLM PR #52805
  (`12f64b39d292`) and PR #53046: stop feeding a terminated grammar
  matcher, keep the cached termination flag consistent under multi-token
  speculative batches.

Exonerated of side effects by the 2026-08-30 KV-pin A/B
(docs/GLM53_FLASH.md).
