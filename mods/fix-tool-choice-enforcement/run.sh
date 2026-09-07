#!/bin/bash
# Restore tool_choice enforcement on /v1/chat/completions.
#
# Symptom: tool_choice="required" and named tool choice decode free-form. The
# model answers in prose and no tool call is produced. A "required" request is
# also malformed on the way out: finish_reason is "tool_calls" while tool_calls
# is null, because the serving layer forces that value and then finds nothing
# to report. Any client that branches on finish_reason breaks on it.
#
# Cause: enforcement is an xgrammar structural tag built in Parser.adjust_request.
# Two independent breaks keep it from ever reaching the sampler.
#
#   1. Nothing on the chat-completions path calls adjust_request.
#      `grep -rn adjust_request vllm/entrypoints/` finds one call site, in the
#      *harmony* branch of the Responses API, next to a TODO about unifying it
#      with the non-harmony branch.
#
#   2. Even when called, a collapsed parser engine never builds the tag.
#      DelegatingParser.adjust_request is the only implementation that calls
#      _apply_structural_tag. When the reasoning and tool adapters share one
#      engine -- glm45 + glm47 do -- ParserManager.get_parser collapses them and
#      returns the engine class instead. ParserEngine overrides adjust_request
#      with a two-line body, and neither it nor Glm47MoeParser carries a
#      structural_tag_model. On that path there is no structural-tag code at all.
#
# Fixing either half alone changes nothing, which is worth knowing before
# bisecting this: verified 2026-09-07 by applying part 1 on its own and getting
# byte-identical output from both probes.
#
# Part 1 is model-agnostic. Part 2 is GLM-specific and is skipped when the GLM
# parser engine is not installed.
set -euo pipefail

PREFIX="[fix-tool-choice-enforcement]"

# VLLM_PACKAGE_ROOT and VLLM_SITE_PACKAGES are what the mod tests and unusual
# image layouts set; PYTHON_ROOT stays supported for consistency with the
# older mods. vLLM is never imported here: doing so during container
# preparation can initialize CUDA before the serving process starts.
if [ -z "${VLLM_PACKAGE_ROOT:-}" ]; then
  if [ -n "${VLLM_SITE_PACKAGES:-}" ]; then
    VLLM_PACKAGE_ROOT="$VLLM_SITE_PACKAGES/vllm"
  else
    VLLM_PACKAGE_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}/vllm"
  fi
fi

SERVING="$VLLM_PACKAGE_ROOT/entrypoints/openai/chat_completion/serving.py"
ENGINE="$VLLM_PACKAGE_ROOT/parser/glm47_moe.py"

if [ ! -d "$VLLM_PACKAGE_ROOT" ]; then
  echo "$PREFIX vLLM package not found at $VLLM_PACKAGE_ROOT" >&2
  exit 1
fi
if [ ! -f "$SERVING" ]; then
  echo "$PREFIX chat-completions serving module not found at $SERVING" >&2
  echo "$PREFIX This mod targets the split entrypoints layout." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Part 1: call the parser's adjust_request from _create_chat_completion.
# Anchored on the parser construction. The call goes inside the `if`, so
# `parser` is known non-None, and ahead of render_chat_request so a reasoning
# parser's own adjust_request can still affect rendering -- the same order the
# Responses API uses, and the order the structural tag needs in order to reach
# to_sampling_params.
# ---------------------------------------------------------------------------
python3 - "$SERVING" "$PREFIX" <<'PY'
import ast
import sys
from pathlib import Path

path, prefix = Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
MARK = "# fix-tool-choice-enforcement: apply the tool-choice structural tag"

if MARK in text:
    print(f"{prefix} serving.py already patched; skipping.")
    raise SystemExit(0)
if ".adjust_request(" in text:
    print(f"{prefix} serving.py already calls adjust_request upstream; skipping.")
    raise SystemExit(0)

OLD = """        if self.parser_cls is not None:
            parser = self.parser_cls(
                tokenizer,
                request.tools,
                chat_template_kwargs=chat_template_kwargs,
                model_config=self.model_config,
            )
        result = await self.render_chat_request(request)
"""
NEW = f"""        if self.parser_cls is not None:
            parser = self.parser_cls(
                tokenizer,
                request.tools,
                chat_template_kwargs=chat_template_kwargs,
                model_config=self.model_config,
            )
            {MARK}.
            # Upstream builds it in Parser.adjust_request but only ever calls
            # that from the Responses API. No-op for requests without tools.
            request = parser.adjust_request(request)
        result = await self.render_chat_request(request)
"""

n = text.count(OLD)
if n != 1:
    print(
        f"{prefix} parser-construction anchor matched {n} times, expected 1; "
        "serving.py layout changed. Refusing to write a partial patch.",
        file=sys.stderr,
    )
    raise SystemExit(1)

patched = text.replace(OLD, NEW, 1)
try:
    ast.parse(patched, filename=str(path))
except SyntaxError as exc:
    print(f"{prefix} patched serving.py failed to parse: {exc}", file=sys.stderr)
    raise SystemExit(1)

path.write_text(patched)
print(f"{prefix} patched serving.py: chat completions now applies the tool-choice tag.")
PY

# ---------------------------------------------------------------------------
# Part 2 (GLM only): give the collapsed parser engine a structural tag.
# Appended at module scope rather than spliced into the class body: the class
# has no stable anchor near its end, and an assignment after it is unambiguous
# and easy to read against upstream. Mirrors
# DelegatingParser._apply_structural_tag rather than inventing a second policy.
# reasoning=False is upstream's argument and is correct here: the structured
# output manager already defers the grammar until the reasoning parser reports
# </think>, so a tag carrying its own reasoning prefix would double-count the
# think block.
# ---------------------------------------------------------------------------
if [ ! -f "$ENGINE" ]; then
  echo "$PREFIX GLM parser engine not installed; part 2 not needed for this model."
  echo "=====> tool_choice enforcement restored on /v1/chat/completions."
  exit 0
fi

python3 - "$ENGINE" "$PREFIX" <<'PY'
import ast
import sys
from pathlib import Path

path, prefix = Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
MARK = "# fix-tool-choice-enforcement: structural tag on the collapsed engine"

if MARK in text:
    print(f"{prefix} glm47_moe.py already patched; skipping.")
    raise SystemExit(0)

SENTINEL = "return super().extract_reasoning(model_output, request)"
if not text.rstrip().endswith(SENTINEL):
    print(
        f"{prefix} glm47_moe.py does not end in Glm47MoeParser.extract_reasoning; "
        "layout changed. Refusing to append to an unrecognised module.",
        file=sys.stderr,
    )
    raise SystemExit(1)
if "class Glm47MoeParser(ParserEngine):" not in text:
    print(f"{prefix} Glm47MoeParser class not found; layout changed.", file=sys.stderr)
    raise SystemExit(1)

appended = text + f'''

{MARK}.
# ParserManager.get_parser collapses the glm45 reasoning adapter and the glm47
# tool adapter into this engine, whose adjust_request does not build the
# xgrammar structural tag that constrains a required or named tool choice.
def _fix_tool_choice_adjust_request(self, request):
    request = ParserEngine.adjust_request(self, request)

    import json

    from vllm import envs

    if not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING or not request.tools:
        return request

    from openai.types.responses.tool_choice_function import ToolChoiceFunction

    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionNamedToolChoiceParam,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

    tool_choice = request.tool_choice
    if not (
        tool_choice == "auto"
        or tool_choice == "required"
        or isinstance(
            tool_choice, (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction)
        )
    ):
        return request

    tag = get_model_structural_tag(
        model="glm_4_7",
        tools=request.tools,
        tool_choice=tool_choice,
        reasoning=False,
    )
    if tag is None:
        return request

    request.structured_outputs = StructuredOutputsParams(
        structural_tag=json.dumps(tag.model_dump())
    )
    if isinstance(request, ResponsesRequest):
        request.text = None
    else:
        request.response_format = None
    return request


Glm47MoeParser.adjust_request = _fix_tool_choice_adjust_request
'''

try:
    ast.parse(appended, filename=str(path))
except SyntaxError as exc:
    print(f"{prefix} patched glm47_moe.py failed to parse: {exc}", file=sys.stderr)
    raise SystemExit(1)

path.write_text(appended)
print(f"{prefix} patched glm47_moe.py: collapsed engine now builds the glm_4_7 tag.")
PY

echo "=====> tool_choice enforcement restored on /v1/chat/completions."
