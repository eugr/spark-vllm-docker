#!/bin/bash
#
# test_tool_choice_enforcement_mod.sh - behaviour tests for the
# fix-tool-choice-enforcement mod.
#
# Fixtures mirror the upstream layout the mod anchors on. No GPU, container,
# or vLLM install is required.
#
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/fix-tool-choice-enforcement/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

SERVING_REL="entrypoints/openai/chat_completion/serving.py"
ENGINE_REL="parser/glm47_moe.py"

# Writes a fixture vLLM tree to $1. $2 selects the serving.py variant:
# "stock" (upstream, unpatched), "fixed" (upstream already calls
# adjust_request), or "drifted" (anchor no longer matches).
make_tree() {
    local root="$1" serving_variant="${2:-stock}" with_engine="${3:-yes}"
    mkdir -p "$root/$(dirname "$SERVING_REL")" "$root/$(dirname "$ENGINE_REL")"

    {
        printf '%s\n' \
            'class OpenAIServingChat:' \
            '    async def _create_chat_completion(self, request, raw_request):' \
            '        tokenizer = await self._get_tokenizer()' \
            '        chat_template_kwargs = self._effective_chat_template_kwargs(request)' \
            '        parser = None' \
            '        if self.parser_cls is not None:' \
            '            parser = self.parser_cls(' \
            '                tokenizer,' \
            '                request.tools,' \
            '                chat_template_kwargs=chat_template_kwargs,' \
            '                model_config=self.model_config,' \
            '            )'
        if [[ "$serving_variant" == "fixed" ]]; then
            printf '%s\n' '            request = parser.adjust_request(request)'
        fi
        if [[ "$serving_variant" == "drifted" ]]; then
            printf '%s\n' '        result = await self.render_chat_request(request)  # upstream drift'
        else
            printf '%s\n' '        result = await self.render_chat_request(request)'
        fi
        printf '%s\n' \
            '        conversation, engine_inputs = result' \
            '        sampling_params = request.to_sampling_params(' \
            '            default_max_tokens, self.default_sampling_params' \
            '        )' \
            '        return sampling_params' \
            '' \
            '    async def chat_completion_stream_generator(self, request):' \
            '        # A second parser_cls block the part 1 anchor must not match.' \
            '        if self.parser_cls is not None:' \
            '            parsers = [' \
            '                self.parser_cls(' \
            '                    tokenizer,' \
            '                    request.tools,' \
            '                    chat_template_kwargs=chat_template_kwargs,' \
            '                    model_config=self.model_config,' \
            '                )' \
            '                for _ in range(num_choices)' \
            '            ]' \
            '        return parsers'
    } > "$root/$SERVING_REL"

    if [[ "$with_engine" == "yes" ]]; then
        printf '%s\n' \
            'from __future__ import annotations' \
            '' \
            'import json' \
            '' \
            'from vllm.parser.engine.parser_engine import ParserEngine' \
            '' \
            '' \
            'class Glm47MoeParser(ParserEngine):' \
            '    def is_reasoning_end(self, input_ids):' \
            '        return True' \
            '' \
            '    def extract_reasoning(self, model_output, request):' \
            '        if not self.thinking_enabled:' \
            '            return None, model_output' \
            '        return super().extract_reasoning(model_output, request)' \
            > "$root/$ENGINE_REL"
    fi
}

# --- 1. Both parts apply to a stock tree -----------------------------------
ROOT="$TMP_DIR/stock/vllm"
make_tree "$ROOT"
output=$(VLLM_PACKAGE_ROOT="$ROOT" bash "$MOD")
grep -Fq 'patched serving.py' <<< "$output"
grep -Fq 'patched glm47_moe.py' <<< "$output"

python3 -m py_compile "$ROOT/$SERVING_REL" "$ROOT/$ENGINE_REL"

# The call must land inside the parser_cls block, after the parser is built and
# before the request is rendered, so the tag reaches to_sampling_params.
python3 - "$ROOT/$SERVING_REL" <<'PY'
import ast
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text()
tree = ast.parse(source)
fn = next(
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.AsyncFunctionDef) and n.name == "_create_chat_completion"
)

calls = [
    n
    for n in ast.walk(fn)
    if isinstance(n, ast.Call)
    and isinstance(n.func, ast.Attribute)
    and n.func.attr == "adjust_request"
]
assert len(calls) == 1, f"expected one adjust_request call, found {len(calls)}"

guard = next(
    n
    for n in fn.body
    if isinstance(n, ast.If) and "parser_cls" in ast.dump(n.test)
)
assert any(
    isinstance(stmt, ast.Assign)
    and isinstance(stmt.value, ast.Call)
    and getattr(stmt.value.func, "attr", None) == "adjust_request"
    for stmt in guard.body
), "adjust_request call is not inside the parser_cls guard"

render = next(
    n
    for n in ast.walk(fn)
    if isinstance(n, ast.Call)
    and isinstance(n.func, ast.Attribute)
    and n.func.attr == "render_chat_request"
)
assert calls[0].lineno < render.lineno, "adjust_request must precede rendering"

# The streaming generator must not have been touched.
stream = next(
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.AsyncFunctionDef)
    and n.name == "chat_completion_stream_generator"
)
assert not [
    n
    for n in ast.walk(stream)
    if isinstance(n, ast.Call)
    and isinstance(n.func, ast.Attribute)
    and n.func.attr == "adjust_request"
], "part 1 leaked into the streaming path"
PY

# The appended engine block must be self-contained and bind the override.
python3 - "$ROOT/$ENGINE_REL" <<'PY'
import ast
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_text())

fn = next(
    n
    for n in ast.walk(tree)
    if isinstance(n, ast.FunctionDef) and n.name == "_fix_tool_choice_adjust_request"
)
imported = {
    alias.name.split(".")[0]
    for node in ast.walk(fn)
    if isinstance(node, ast.Import)
    for alias in node.names
}
assert "json" in imported, "appended block relies on a module-level json import"

assert any(
    isinstance(n, ast.Assign)
    and any(
        isinstance(t, ast.Attribute)
        and t.attr == "adjust_request"
        and getattr(t.value, "id", None) == "Glm47MoeParser"
        for t in n.targets
    )
    for n in tree.body
), "Glm47MoeParser.adjust_request was not rebound"

# reasoning=False is deliberate: the structured output manager defers the
# grammar until the reasoning parser reports </think>.
call = next(
    n
    for n in ast.walk(fn)
    if isinstance(n, ast.Call)
    and getattr(n.func, "id", None) == "get_model_structural_tag"
)
kwargs = {kw.arg: kw.value for kw in call.keywords}
assert set(kwargs) == {"model", "tools", "tool_choice", "reasoning"}
assert kwargs["model"].value == "glm_4_7"
assert kwargs["reasoning"].value is False
PY

# --- 2. Idempotent ----------------------------------------------------------
before_serving=$(sha256sum "$ROOT/$SERVING_REL")
before_engine=$(sha256sum "$ROOT/$ENGINE_REL")
second=$(VLLM_PACKAGE_ROOT="$ROOT" bash "$MOD")
grep -Fq 'serving.py already patched' <<< "$second"
grep -Fq 'glm47_moe.py already patched' <<< "$second"
test "$before_serving" = "$(sha256sum "$ROOT/$SERVING_REL")"
test "$before_engine" = "$(sha256sum "$ROOT/$ENGINE_REL")"

# --- 3. Drifted serving.py fails closed and leaves both files alone ---------
DRIFT="$TMP_DIR/drift-serving/vllm"
make_tree "$DRIFT" drifted
drift_serving_before=$(sha256sum "$DRIFT/$SERVING_REL")
drift_engine_before=$(sha256sum "$DRIFT/$ENGINE_REL")
if VLLM_PACKAGE_ROOT="$DRIFT" bash "$MOD" >/dev/null 2>&1; then
    echo "[FAIL] mod accepted a drifted serving.py layout" >&2
    exit 1
fi
test "$drift_serving_before" = "$(sha256sum "$DRIFT/$SERVING_REL")"
test "$drift_engine_before" = "$(sha256sum "$DRIFT/$ENGINE_REL")"

# --- 4. Drifted engine tail fails closed ------------------------------------
DRIFT2="$TMP_DIR/drift-engine/vllm"
make_tree "$DRIFT2"
printf '\n\n    def new_upstream_method(self):\n        return None\n' >> "$DRIFT2/$ENGINE_REL"
drift2_engine_before=$(sha256sum "$DRIFT2/$ENGINE_REL")
if VLLM_PACKAGE_ROOT="$DRIFT2" bash "$MOD" >/dev/null 2>&1; then
    echo "[FAIL] mod appended to an unrecognised glm47_moe.py" >&2
    exit 1
fi
test "$drift2_engine_before" = "$(sha256sum "$DRIFT2/$ENGINE_REL")"

# --- 5. Non-GLM image applies part 1 only -----------------------------------
NOGLM="$TMP_DIR/no-glm/vllm"
make_tree "$NOGLM" stock no
noglm_output=$(VLLM_PACKAGE_ROOT="$NOGLM" bash "$MOD")
grep -Fq 'patched serving.py' <<< "$noglm_output"
grep -Fq 'GLM parser engine not installed' <<< "$noglm_output"
python3 -m py_compile "$NOGLM/$SERVING_REL"

# --- 6. An upstream that already calls adjust_request is left alone ---------
FIXED="$TMP_DIR/upstream-fixed/vllm"
make_tree "$FIXED" fixed
fixed_before=$(sha256sum "$FIXED/$SERVING_REL")
fixed_output=$(VLLM_PACKAGE_ROOT="$FIXED" bash "$MOD")
grep -Fq 'already calls adjust_request upstream' <<< "$fixed_output"
test "$fixed_before" = "$(sha256sum "$FIXED/$SERVING_REL")"

# --- 7. Missing vLLM is an error, not a silent success ----------------------
if VLLM_PACKAGE_ROOT="$TMP_DIR/absent/vllm" bash "$MOD" >/dev/null 2>&1; then
    echo "[FAIL] mod reported success without a vLLM install" >&2
    exit 1
fi

echo "[PASS] fix-tool-choice-enforcement mod is targeted, valid, and idempotent"
