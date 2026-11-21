#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/fix-glm47-phantom-fallback"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

TARGET="$TMP_DIR/vllm/parser/glm47_moe.py"
mkdir -p "$(dirname "$TARGET")"

# Fixture reproduces the pinned container build's class-body region (the
# context hunks anchor on it). The other two hunks of this patch anchor on
# the import block and the module-level helper area, reproduced below.
cat > "$TARGET" <<'EOF'
from __future__ import annotations

import functools
import json
from typing import TYPE_CHECKING

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)


class ParserState:
    CONTENT = "content"


_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>(?=\s*(?:<arg_key>|$))",
    re.DOTALL,
)
_TAIL_ARG_RE = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*)$",
    re.DOTALL,
)


def _glm47_arg_converter(raw_args: str, partial: bool) -> str:
    params: dict[str, object] = {}
    pos = 0


class Glm47MoeParser(ParserEngine):
    """GLM-4.7 parser backed by the declarative parser engine."""

    def __init__(self, tokenizer, tools=None, **kwargs) -> None:
        super().__init__(tokenizer, tools, **kwargs)

    def _emit_name_delta(self, idx: int, deltas, name) -> None:
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if not self.thinking_enabled:
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning(self, model_output: str, request) -> tuple:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
EOF

(cd "$MOD" && VLLM_PACKAGE_ROOT="$TMP_DIR/vllm" ./run.sh) >/dev/null

grep -Fq "_rescue_phantom_slot" "$TARGET"
python3 -c "import ast; ast.parse(open('$TARGET').read())"

# Idempotent re-run must be a no-op.
(cd "$MOD" && VLLM_PACKAGE_ROOT="$TMP_DIR/vllm" ./run.sh) >/dev/null
[[ $(grep -Fc '_rescue_phantom_slot' "$TARGET") -eq 4 ]]

# Negative drift test: without the exact _handle_tool_end anchor the apply
# must fail closed and leave the file untouched.
cp "$TARGET" "$TMP_DIR/patched-good.py"
cat > "$TARGET" <<'EOF'
class Glm47MoeParser(ParserEngine):
    """GLM-4.7 parser backed by the declarative parser engine."""

    def _handle_tool_end(self, event, deltas) -> None:
        super()._handle_tool_end(event, deltas)
EOF
SHA_BEFORE=$(sha256sum "$TARGET" | awk '{print $1}')
if (cd "$MOD" && VLLM_PACKAGE_ROOT="$TMP_DIR/vllm" ./run.sh) >/dev/null 2>&1; then
    echo "FAIL: mod applied to a source without the anchor block" >&2
    exit 1
fi
SHA_AFTER=$(sha256sum "$TARGET" | awk '{print $1}')
[[ "$SHA_BEFORE" == "$SHA_AFTER" ]]

echo "test_fix_glm47_phantom_fallback_mod: OK"
