#!/usr/bin/env python3
"""Prevent a speculative-decoding assertion from stopping EngineCore.

Guided decoding can invalidate scheduled draft tokens after sampling.
`Scheduler.update_from_output` then compares the sampler's accepted count
against the smaller valid-draft count. The original assertion can stop
EngineCore even though rollback uses scheduled drafts correctly.

The SOFT branch preserves the accepted count and widens only the count sent
to metrics. The HARD branch clamps an impossible accepted count that exceeds
scheduled drafts, preventing a negative rollback. Both branches log when run.

The stats consumers also require accepted <= draft count; changing only the
scheduler assertion would move the failure downstream. Exact anchors fail
clearly if the target scheduler has drifted.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "SPEC-ACCEPTED-CLAMP-UPSTREAM"
SCHED_REL = "v1/core/sched/scheduler.py"


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(
            f"ANCHOR FAILED for [{description}]: expected exactly 1 occurrence, "
            f"found {count}. The upstream file has drifted -- re-derive the "
            f"anchor before building.\n--- anchor ---\n{old}\n--------------"
        )
    return text.replace(old, new, 1)


ASSERT_ANCHOR = '''                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                assert num_accepted <= num_draft_tokens, (
                    f"{req_id}: accepted={num_accepted}, "
                    f"valid_drafts={num_draft_tokens}, "
                    f"scheduled_drafts={num_scheduled_draft_tokens}, "
                    f"grammar_invalid="
                    f"{(scheduler_output.num_invalid_spec_tokens or {}).get(req_id, 0)}"
                )
'''

ASSERT_REPLACEMENT = f'''                num_sampled = self.num_sampled_tokens_per_step
                num_accepted = max(len(generated_token_ids) - num_sampled, 0)
                # {MARKER}: the assertion compared accepted scheduled drafts
                # against the smaller grammar-filtered draft count.
                # Rollback already uses num_scheduled_draft_tokens correctly;
                # widen only the count passed to metrics here.
                num_stats_draft_tokens = num_draft_tokens
                if num_accepted > num_scheduled_draft_tokens:
                    # Impossible by construction. A negative num_rejected
                    # would advance num_computed_tokens; clamp and warn.
                    logger.warning_once(
                        "[{MARKER}] HARD: %s accepted=%d > scheduled_drafts=%d; "
                        "clamping accepted (negative rollback averted).",
                        req_id,
                        num_accepted,
                        num_scheduled_draft_tokens,
                    )
                    num_accepted = num_scheduled_draft_tokens
                if num_accepted > num_stats_draft_tokens:
                    logger.warning_once(
                        "[{MARKER}] SOFT: %s accepted=%d > valid_drafts=%d "
                        "(scheduled=%d, grammar_invalid=%d); stats widened, "
                        "rollback untouched.",
                        req_id,
                        num_accepted,
                        num_stats_draft_tokens,
                        num_scheduled_draft_tokens,
                        (scheduler_output.num_invalid_spec_tokens or {{}}).get(
                            req_id, 0
                        ),
                    )
                    num_stats_draft_tokens = num_accepted
'''

STATS_ANCHOR = '''                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )
                if request.spec_decode_metrics is not None and num_draft_tokens:
                    request.spec_decode_metrics.observe(
                        num_draft_tokens=num_draft_tokens,
                        num_accepted=num_accepted,
'''

STATS_REPLACEMENT = f'''                # {MARKER}: metrics need the widened count: observe_draft
                # asserts accepted <= draft <= k, and stats indexes the
                # histogram by accepted count. Otherwise the failure moves.
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_stats_draft_tokens,
                    num_accepted_tokens=num_accepted,
                )
                if (
                    request.spec_decode_metrics is not None
                    and num_stats_draft_tokens
                ):
                    request.spec_decode_metrics.observe(
                        num_draft_tokens=num_stats_draft_tokens,
                        num_accepted=num_accepted,
'''


def patch_scheduler(text: str) -> str:
    if MARKER in text:
        return text
    text = replace_once(
        text, ASSERT_ANCHOR, ASSERT_REPLACEMENT, "spec accepted assert"
    )
    text = replace_once(
        text, STATS_ANCHOR, STATS_REPLACEMENT, "spec decoding stats feed"
    )
    if "num_stats_draft_tokens" not in text:
        raise PatchError("clamp patch incomplete: num_stats_draft_tokens absent")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--check", action="store_true", help="report status, write nothing")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.vllm_root)
    path = root / SCHED_REL
    if not path.exists():
        raise PatchError(f"{SCHED_REL} missing under {root}")

    original = path.read_text()
    already = MARKER in original

    if args.check:
        if already:
            print(f"[{MARKER}] check: already patched")
            return 0
        patch_scheduler(original)  # raises PatchError on drift
        print(f"[{MARKER}] check: ready (anchors match, nothing written)")
        return 0

    updated = patch_scheduler(original)
    ast.parse(updated)
    if updated == original:
        print(f"[{MARKER}] already present (idempotent no-op)")
        return 0
    if args.dry_run:
        print(f"[{MARKER}] dry-run: {SCHED_REL} pending")
        return 0
    path.write_text(updated)
    print(f"[{MARKER}] applied: {SCHED_REL}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PatchError as exc:
        print(f"[{MARKER}] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
