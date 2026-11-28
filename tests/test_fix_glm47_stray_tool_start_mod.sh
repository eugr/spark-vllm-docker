#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/fix-glm47-stray-tool-start"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

TARGET="$TMP_DIR/vllm/parser/glm47_moe.py"
mkdir -p "$(dirname "$TARGET")"

# Fixture shaped like the shipped transitions dict around the insertion
# point. The mod patch is a context hunk, so the anchor entries must match
# the pinned container build exactly or the apply fails.
cat > "$TARGET" <<'EOF'
class EventType:
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"


class ParserState:
    CONTENT = "content"
    TOOL_NAME = "tool_name"


class Transition:
    def __init__(self, state, events):
        self.state = state
        self.events = events


def _transitions():
    return {
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
    }
EOF

(cd "$MOD" && VLLM_PACKAGE_ROOT="$TMP_DIR/vllm" ./run.sh) >/dev/null

grep -Fq '(ParserState.TOOL_NAME, "TOOL_START")' "$TARGET"
python3 -c "import ast; ast.parse(open('$TARGET').read())"

# Idempotent re-run must be a no-op.
(cd "$MOD" && VLLM_PACKAGE_ROOT="$TMP_DIR/vllm" ./run.sh) >/dev/null
[[ $(grep -Fc '(ParserState.TOOL_NAME, "TOOL_START")' "$TARGET") -eq 1 ]]

# Negative drift test: without the anchor block the apply must fail closed
# and leave the file untouched. GNU patch fuzz is 2, so all four leading
# context lines must be absent for the apply to refuse.
cp "$TARGET" "$TMP_DIR/patched-good.py"
cat > "$TARGET" <<'EOF'
class EventType:
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"


class ParserState:
    CONTENT = "content"
    TOOL_NAME = "tool_name"


class Transition:
    def __init__(self, state, events):
        self.state = state
        self.events = events


def _transitions():
    return {
            (ParserState.TOOL_NAME, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
    }
EOF
SHA_BEFORE=$(sha256sum "$TARGET" | awk '{print $1}')
if (cd "$MOD" && VLLM_PACKAGE_ROOT="$TMP_DIR/vllm" ./run.sh) >/dev/null 2>&1; then
    echo "FAIL: mod applied to a source without the anchor block" >&2
    exit 1
fi
SHA_AFTER=$(sha256sum "$TARGET" | awk '{print $1}')
[[ "$SHA_BEFORE" == "$SHA_AFTER" ]]

echo "test_fix_glm47_stray_tool_start_mod: OK"
