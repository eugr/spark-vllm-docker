#!/usr/bin/env python3
"""Restore three working reasoning-effort levels to vLLM's vendored DeepSeek-V4
tokenizer.

The container pins a vLLM commit predating the fix in vLLM main where the
vendored deepseek_v4 tokenizer split reasoning effort into three levels. On the
pinned copy `low` and `high` collapse to one behavior and the model's true top
tier is unreachable. This applies vLLM main's own fix, copied verbatim, to the
effort path only. Every non-effort line is left untouched, so nothing else in
the tokenizer changes. Both target files are stock vLLM (SPDX: vLLM project);
neither carries any b12x / fork code.

Anchor-matched, single-occurrence, abort-on-mismatch, idempotent, atomic. It
never rewrites a file it does not recognize.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PREFIX = "[fix-dsv4-effort-levels]"

# ---------------------------------------------------------------------------
# deepseek_v4_encoding.py -- the prefix table and the condition that emits it.
# OLD/NEW are copied verbatim from the pinned file and from vLLM main.
# ---------------------------------------------------------------------------

ENC_CONST_OLD = '''REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\\n\\n"
)'''

ENC_CONST_NEW = '''REASONING_EFFORT_PROMPTS: Dict[str, str] = {
    "low": "",
    "high": (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\\n\\n"
    ),
    "max": (
        "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\\n"
        "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\\n"
        "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\\n\\n"
    ),
}
DEFAULT_REASONING_EFFORT = "low"'''

ENC_COND_OLD = '''    # Reasoning effort prefix (only at index 0 in thinking mode with max effort)
    assert reasoning_effort in ['max', None, 'high'], f"Invalid reasoning effort: {reasoning_effort}"
    if index == 0 and thinking_mode == "thinking" and reasoning_effort == 'max':
        prompt += REASONING_EFFORT_MAX'''

ENC_COND_NEW = '''    reasoning_effort = reasoning_effort or DEFAULT_REASONING_EFFORT
    assert reasoning_effort in REASONING_EFFORT_PROMPTS, (
        f"Invalid reasoning effort: {reasoning_effort}, expected one of "
        f"{list(REASONING_EFFORT_PROMPTS)}"
    )
    if index == 0 and thinking_mode == "thinking":
        prompt += REASONING_EFFORT_PROMPTS[reasoning_effort]'''

# ---------------------------------------------------------------------------
# deepseek_v4.py -- the normalizer. Purely additive: one elif is inserted so
# low/minimal/medium map to "low" instead of falling through to "high". No
# existing line changes; the thinking on/off logic is left exactly as pinned.
# ---------------------------------------------------------------------------

NORM_OLD = '''            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"'''

NORM_NEW = '''            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            elif reasoning_effort in ("low", "minimal", "medium"):
                reasoning_effort = "low"
            else:
                reasoning_effort = "high"'''

# target basename -> (list of (old, new) edits, list of required post-patch symbols)
EDITS = {
    "deepseek_v4_encoding.py": (
        [(ENC_CONST_OLD, ENC_CONST_NEW), (ENC_COND_OLD, ENC_COND_NEW)],
        ["REASONING_EFFORT_PROMPTS", "DEFAULT_REASONING_EFFORT"],
    ),
    "deepseek_v4.py": (
        [(NORM_OLD, NORM_NEW)],
        ['reasoning_effort in ("low", "minimal", "medium")'],
    ),
}


def _apply(text: str, edits: list[tuple[str, str]]) -> tuple[str, bool]:
    """Return (result, changed). Idempotent: if every OLD anchor is already
    gone and every NEW block is present, report no change. Otherwise every OLD
    anchor must appear exactly once, or we abort."""
    already = all(old not in text and new in text for old, new in edits)
    if already:
        return text, False

    out = text
    for old, new in edits:
        count = out.count(old)
        if count != 1:
            raise ValueError(
                f"expected exactly one occurrence of anchor, found {count}:\n"
                f"----\n{old[:120]}...\n----"
            )
        out = out.replace(old, new, 1)
    return out, True


def _validate(text: str, required: list[str]) -> None:
    ast.parse(text)                       # still valid Python
    compile(text, "<patched>", "exec")    # and compilable
    for sym in required:
        if sym not in text:
            raise ValueError(f"post-patch check failed: {sym!r} not present")


def process(target: Path, *, check: bool) -> int:
    edits, required = EDITS[target.name]
    original = target.read_text()
    patched, changed = _apply(original, edits)

    if not changed:
        _validate(patched, required)
        print(f"{PREFIX} {target.name} already patched; skipping.")
        return 0

    _validate(patched, required)

    if check:
        print(f"{PREFIX} {target.name} is compatible (would patch).")
        return 0

    tmp = target.with_suffix(target.suffix + ".dsv4-effort.tmp")
    tmp.write_text(patched)
    tmp.replace(target)                   # atomic rename
    print(f"{PREFIX} patched {target.name}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="validate compatibility without writing")
    args = ap.parse_args()

    if args.target.name not in EDITS:
        print(f"{PREFIX} ERROR: unknown target {args.target.name}", file=sys.stderr)
        return 1
    if not args.target.is_file():
        print(f"{PREFIX} ERROR: target not found: {args.target}", file=sys.stderr)
        return 1

    try:
        return process(args.target, check=args.check)
    except (SyntaxError, ValueError, KeyError) as exc:
        print(f"{PREFIX} ERROR: refusing to patch {args.target}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
