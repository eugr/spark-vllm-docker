#!/usr/bin/env bash
# hotfix-dsv4-grammar-advance.sh — Advance grammar across the reasoning boundary
# (Stage-A #44993 backport). Fixes structured-output corruption (think #24:
# `response_format json_schema` + thinking enabled → duplicated prefix).
#
# ROOT CAUSE (this fork only): `should_advance` returns False for JSON/regex/
# choice when reasoning ends mid-step, so the scheduler's
# `trim_reasoning_for_advance()` + `accept_tokens()` never run on the boundary
# step. With speculative decoding, the draft produced right after the
# reasoning_end marker never enters the grammar FSM → the model re-emits the
# opening token next step → `{"city":"Paris{"city":"Paris"}`.
#
# Upstream #44993 (merged) fixes it: should_advance takes the step's
# `new_token_ids` as the exact delta window (placeholder count is wrong under
# async+spec when drafts are rejected), records `reasoning_end_token_index` for
# ALL constraint types, and the scheduler trims + advances with the verified
# post-marker suffix.
#
# Usage (like the other hotfixes):
#   docker cp hotfix-dsv4-grammar-advance.sh <container>:/tmp/ && \
#   docker exec <container> bash /tmp/hotfix-dsv4-grammar-advance.sh
#   bash hotfix-dsv4-grammar-advance.sh --status   (inside container)
#   bash hotfix-dsv4-grammar-advance.sh --before    (host-side; doc only)
#   bash hotfix-dsv4-grammar-advance.sh --after     (host-side; doc only)
#
# Idempotent — re-running skips already-applied hunks.
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
ACTION="${1:-}"

if [ ! -d "$VLLM_ROOT" ]; then
  echo "ERROR: vLLM not found at $VLLM_ROOT (run inside the container)" >&2
  exit 1
fi

status() {
  python3 - "$VLLM_ROOT" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
so = root / "v1" / "structured_output" / "__init__.py"
sch = root / "v1" / "core" / "sched" / "scheduler.py"
t_so = so.read_text()
t_sch = sch.read_text()

def chk(label, cond):
    print(f"{label:44} :", "APPLIED" if cond else "NOT APPLIED")

chk("should_advance new_token_ids param", "new_token_ids: list[int] | None = None" in t_so)
chk("delta window uses new_token_ids", "delta_ids: Iterable[int] = new_token_ids" in t_so)
chk("boundary records end_index (all types)", "end_index = self._find_reasoning_end_index(reasoner, all_token_ids, start)" in t_so)
chk("old STRUCTURAL_TAG-only gate removed",
    '== StructuredOutputOptions.STRUCTURAL_TAG' not in t_so and
    "self.vllm_config.speculative_config is not None" not in t_so)
chk("scheduler passes new_token_ids", "should_advance(\n                request, new_token_ids=new_token_ids\n            )" in t_sch)
PY
  exit 0
}

if [ "$ACTION" = "--status" ]; then
  status
fi

# host-side before/after are documentation-only for this hotfix (no KV change
# expected: pure correctness fix). Provide the same interface as siblings.
if [ "$ACTION" = "--before" ] || [ "$ACTION" = "--after" ]; then
  if [ "$ACTION" = "--before" ]; then
    echo "No KV change expected: #44993 backport is a correctness fix (grammar"
    echo "advance across the reasoning boundary). Validate via issue #24 repro."
  else
    echo "No KV change expected. Run the #24 repro instead of --after."
  fi
  exit 0
fi

echo "=== Hotfix: DSV4 grammar advance across reasoning boundary (upstream #44993 backport) ==="
echo "vLLM root: $VLLM_ROOT  image: 0.25.2.dev0+g752a3a504.d20260714"

python3 <<PYEOF
import sys
from pathlib import Path

root = Path("$VLLM_ROOT")
applied = 0
skipped = 0
errors = []

def patch(path: str, old: str, new: str, label: str, expect: int = 1) -> None:
    global applied, skipped
    p = root / path
    if not p.exists():
        errors.append(f"File not found: {path}")
        return
    text = p.read_text()
    if new in text:
        print(f"  [skip] {label} (already applied)")
        skipped += 1
        return
    n = text.count(old)
    if n == 0 or (expect and n != expect):
        errors.append(f"[ERR] anchor x{n} (expect {expect}) for {label} in {path}")
        return
    text = text.replace(old, new)
    p.write_text(text)
    print(f"  [OK]   {label} (replaced {n})")
    applied += 1


# ---- structured_output/__init__.py: should_advance signature ----------------
patch(
    "v1/structured_output/__init__.py",
    """    def should_advance(self, request: "Request") -> bool:""",
    """    def should_advance(
        self,
        request: "Request",
        new_token_ids: list[int] | None = None,
    ) -> bool:""",
    "__init__.py: should_advance new_token_ids param (upstream #44993)",
)

# ---- structured_output/__init__.py: delta-window + boundary record ----------
patch(
    "v1/structured_output/__init__.py",
    """        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            structured_req.reasoning_ended = True

            # Reasoning just ended this step. Defer FSM advance until the next
            # pass (see reasoning_ended check above) for JSON/regex/choice/grammar:
            # advancing on the closing boundary token can accept tokens that still
            # belong to the reasoning stream. Structural tags are the only safe
            # same-step exception: they model phased output (e.g. thinking tag ->
            # answer tag), and speculative decoding must run grammar.validate_tokens
            # on draft tokens produced immediately after that transition.
            if (
                self.vllm_config.speculative_config is not None
                and structured_req.structured_output_key[0]
                == StructuredOutputOptions.STRUCTURAL_TAG
            ):
                # The scheduler will advance the grammar with this step's
                # tokens right away, but the step still contains reasoning
                # content up to and including the end marker. Record where
                # it ends so trim_reasoning_for_advance() can drop it.
                structured_req.reasoning_end_token_index = (
                    self._find_reasoning_end_index(reasoner, all_token_ids, start)
                )
                return True

        return False""",
    """        # Check if reasoning ends in *this* step.
        # When the caller passes new_token_ids (the tokens that were just
        # appended this step), use it directly as the delta window. The
        # placeholder-derived fallback assumes num_output_placeholders ==
        # len(new_token_ids), which breaks under async scheduling + spec
        # decode when some drafts are rejected (#43388): the placeholder
        # count remains > 0 after the step and the computed delta window
        # starts past the reasoning-end marker.
        all_token_ids = request.all_token_ids
        if new_token_ids:
            # The tokens were already appended this step, so the step window
            # starts exactly len(new_token_ids) from the end.
            start = len(all_token_ids) - len(new_token_ids)
            delta_ids: Iterable[int] = new_token_ids
        else:
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            start = (
                delta_from
                if delta_from >= 0
                else max(len(all_token_ids) + delta_from, 0)
            )
            delta_ids = itertools.islice(all_token_ids, start, None)
        if reasoner.is_reasoning_end_streaming(all_token_ids, delta_ids):
            structured_req.reasoning_ended = True

            # Record the boundary so the scheduler can exclude reasoning tokens.
            end_index = self._find_reasoning_end_index(reasoner, all_token_ids, start)

            structured_req.reasoning_end_token_index = end_index
            return True

        return False""",
    "__init__.py: delta window + boundary record, all constraint types (upstream #44993)",
)

# ---- scheduler.py: pass the step tokens as the exact delta window ------------
patch(
    "v1/core/sched/scheduler.py",
    """            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar""",
    """            if new_token_ids and self.structured_output_manager.should_advance(
                request, new_token_ids=new_token_ids
            ):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                grammar = struct_output_request.grammar""",
    "scheduler.py: pass new_token_ids into should_advance (upstream #44993)",
)


print(f"\nApplied: {applied}, Skipped: {skipped}, Errors: {len(errors)}")
for e in errors:
    print(f"  {e}", file=sys.stderr)

if errors:
    print("\nWARNING: Some patches could not be applied. Nothing was left half-applied.")
    sys.exit(1)

if applied == 0 and skipped > 0:
    print("Patch already applied. No changes needed.")
elif applied > 0:
    print("\nHotfix applied. Restart the vLLM process (or container) to take effect.")
    print("Validate: issue #24 repro (json_schema + thinking) x5 must be clean JSON.")
PYEOF

echo ""
echo "=== Verification ==="
bash "$0" --status
