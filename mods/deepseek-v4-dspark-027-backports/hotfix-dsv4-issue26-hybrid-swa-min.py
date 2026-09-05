#!/usr/bin/env python3
"""Issue #26 / #36 hybrid prefix-cache hotfix (v2).

DSV4-Flash + DSpark has four KV groups (1x MLA + 3x SlidingWindowMLA).
``find_longest_cache_hit`` takes the min hit length across groups.

v1 (issue #26) skipped that min for SlidingWindowSpec so a long MLA hit
survived when SWA only had a window tail. That restored warm x8 hit *rate*,
but when SWA did not actually have a tail at that length the engine still
skipped prefill and attached **null** SWA KV. Hermes-style shared ~21k
prefixes then decoded DSML / CJK salad (issue #36).

v2: SWA is allowed to shrink the common hit again. Warm long-prefix hits
come from ``VLLM_PREFIX_CACHE_RETENTION_INTERVAL`` (sparse SWA tails, default
4096), which keeps the last retained window instead of zeroing the hit.
A 21k prefix may cache ~20k instead of 21k; that is correct.

Idempotent. Reverts an in-image v1 inject. Called from the compose
entrypoint before ``exec vllm serve``.
"""
from __future__ import annotations

from pathlib import Path

P = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py"
)

MARK_V2 = "# [issue26-hotfix-v2] SWA may shrink the common hit (#36)"
MARK_V1 = "# [issue26-hotfix] SWA groups must not shrink the hybrid common hit"

STOCK = (
    "                _new_hit_length = len(hit_blocks[0]) * spec.block_size\n"
    "                if drop_eagle_block:\n"
    "                    eagle_verified.add(idx)\n"
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
)

V1_INJECT = (
    "                _new_hit_length = len(hit_blocks[0]) * spec.block_size\n"
    "                # [issue26-hotfix] SWA groups must not shrink the hybrid common hit.\n"
    "                # Sliding-window managers retain only the last window tokens;\n"
    "                # using their hit as the next candidate (min-across-groups)\n"
    "                # zeroes warm prefix-cache hits at 32K+ x8 (issue #26).\n"
    "                if isinstance(spec, SlidingWindowSpec):\n"
    "                    if drop_eagle_block:\n"
    "                        eagle_verified.add(idx)\n"
    "                    for group_id, blocks in zip(group_ids, hit_blocks):\n"
    "                        hit_blocks_by_group[group_id] = blocks\n"
    "                    continue\n"
    "                if drop_eagle_block:\n"
    "                    eagle_verified.add(idx)\n"
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
)

V2_BLOCK = (
    "                _new_hit_length = len(hit_blocks[0]) * spec.block_size\n"
    "                # [issue26-hotfix-v2] SWA may shrink the common hit (#36).\n"
    "                # v1 skipped this assign for SlidingWindowSpec so a long MLA\n"
    "                # hit could skip prefill with null SWA KV (DSML / CJK salad).\n"
    "                # Issue #26 warm hits come from VLLM_PREFIX_CACHE_RETENTION_INTERVAL\n"
    "                # keeping sparse SWA tails, not from ignoring SWA length.\n"
    "                if drop_eagle_block:\n"
    "                    eagle_verified.add(idx)\n"
    "                elif _new_hit_length < curr_hit_length:\n"
    "                    # length shrunk; invalidate previous eagle verifications\n"
    "                    eagle_verified.clear()\n"
    "                curr_hit_length = _new_hit_length\n"
)


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status) where status is v2|reverted-v1|annotated."""
    if MARK_V2 in src:
        return src, "v2"
    if V1_INJECT in src or MARK_V1 in src:
        if V1_INJECT not in src:
            raise SystemExit("issue26 v1 marker present but inject block not found")
        return src.replace(V1_INJECT, V2_BLOCK, 1), "reverted-v1"
    if STOCK not in src:
        raise SystemExit("hybrid min-hit assign anchor not found; refusing to patch")
    return src.replace(STOCK, V2_BLOCK, 1), "annotated"


def main() -> None:
    src = P.read_text()
    new, status = apply_text(src)
    if status == "v2":
        print(f"[issue26-hotfix-v2] already applied to {P}")
        return
    P.write_text(new)
    print(f"[issue26-hotfix-v2] {status}: {P}")


if __name__ == "__main__":
    main()
