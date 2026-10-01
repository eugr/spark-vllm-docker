#!/bin/bash
set -e

# tool_choice="required" / named tool choice / strict tools are silently
# unenforced when the reasoning and tool parsers share a parser engine
# (e.g. --reasoning-parser qwen3 with --tool-call-parser qwen3_xml — and
# every other engine-backed pair: DeepSeek V3.2/V4, MiniMax M2, Gemma4,
# Kimi K2, GLM 4.7, Seed-OSS, Nemotron V3, Inkling, Mistral).
#
# ParserManager.get_parser returns the engine class directly on that path,
# bypassing DelegatingParser.adjust_request — the only place the xgrammar
# structural tag was applied. The model can then answer tool_choice="required"
# requests in plain text with no error. Found via tool-eval-bench TC-45.
#
# Upstream fix: hoists the structural-tag application to the Parser base and
# propagates the tool adapter's structural_tag_model to shared engines.
echo "Patching tool_choice enforcement for shared parser engines"
patch -p1 -d /usr/local/lib/python3.12/dist-packages \
  < tool_choice_enforcement.diff \
  || echo "Patch not applicable (already fixed upstream?), skipping"

# Cheap self-check: the resolved qwen3 parser must carry the tag model.
python3 - <<'PY' || echo "WARNING: tool-choice enforcement self-check failed"
from vllm.parser.parser_manager import ParserManager
cls = ParserManager.get_parser(
    tool_parser_name="qwen3_xml",
    reasoning_parser_name="qwen3",
    enable_auto_tools=True,
)
assert getattr(cls, "structural_tag_model", None), "structural_tag_model missing"
print("tool-choice enforcement fix active:", cls.__name__)
PY
