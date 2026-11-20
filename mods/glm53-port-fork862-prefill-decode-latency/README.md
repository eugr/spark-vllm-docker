# glm53-port-fork862-prefill-decode-latency

Port of [local-inference-lab/vllm PR #862](https://github.com/local-inference-lab/vllm/pull/862)
("bound contended prefill compute steps", author yatesdr, Apache-2.0) onto the
fork commit the image is built from. Adds a scheduler knob that caps prefill
tokens per contended step, so a long prefill cannot starve concurrent decodes.

## Why

With prefill compute-sharing, one long prefill hogs whole steps (author:
286 ms decode gap during a 229K-token prefill) and short decode requests
stall behind it. The cap only applies to contended steps; decode-only and
uncontended steps are unchanged.

## Enable

Requires `--prefill-compute-share` (e.g. `auto`) and `--max-parallel-prefills 2`
(with 1 the cap does nothing: the short request waits out the whole prefill).

```bash
VLLM_GLM53_PREFILL_DECODE_LATENCY=1  # gate, off by default
# plus: --max-num-prefill-tokens-per-step 512
```

Execution proof (logged once when actually bounding):

```text
[GLM53-PORT-FORK862-PREFILL-DECODE-LATENCY] bounding contended prefill to N tokens/step
```

## Measured numbers

Mixed load on a 2x DGX Spark TP2 setup: 3x 150K-token prompts plus 2 prose
requests (1,000 prose tokens target):

| setup | 1,000 prose tokens in | aggregate prefill |
|---|---|---|
| baseline | 359 s | — |
| `--prefill-compute-share auto` + `--max-parallel-prefills 2` + cap 1024 | 100–161 s | −12 % |
| same + cap 512 | 93–153 s | −30 % |

Steady single-request decode is unchanged (no contention, nothing to bound).
Sweep 256/512/1024: too low slows prefill without helping decode, too high
bounds nothing.

## Disable / rollback

Unset `VLLM_GLM53_PREFILL_DECODE_LATENCY` (inert without it) or drop the mod
from the mods list. With the knob at 0 the scheduler is identical to stock.
