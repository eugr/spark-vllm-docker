#!/usr/bin/env python3
"""
Inline fallback patch for vLLM #38855: Gemma 4 reasoning parser gets text
with special tokens already stripped, so <|channel> / <channel|> delimiters
are missing and reasoning_content is never populated.

Fix: before calling extract_reasoning(), check if reasoning boundary token IDs
are present in the output. If so, re-decode with skip_special_tokens=False.

This mirrors the approach from (closed) PR #38858.
"""

import glob
import sys

# Find the chat completion serving module
pattern = "/usr/local/lib/python*/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py"
matches = glob.glob(pattern)
if not matches:
    print(f"ERROR: Could not find serving.py matching {pattern}", file=sys.stderr)
    sys.exit(1)

serving_path = matches[0]
print(f"Patching {serving_path}")

with open(serving_path, "r") as f:
    content = f.read()

# The helper function to inject
helper_fn = '''
def _get_reasoning_text(output, tokenizer, reasoning_parser):
    """Re-decode output with skip_special_tokens=False if reasoning
    boundary tokens are present (fix for #38855)."""
    if reasoning_parser is None:
        return output.text
    start_id = getattr(reasoning_parser, "start_token_id", None)
    end_id = getattr(reasoning_parser, "end_token_id", None)
    if start_id is None and end_id is None:
        return output.text
    token_ids = output.token_ids
    if (start_id is not None and start_id in token_ids) or \\
       (end_id is not None and end_id in token_ids):
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    return output.text
'''

# Check if already patched
if "_get_reasoning_text" in content:
    print("Already patched, skipping.")
    sys.exit(0)

# Strategy: inject the helper function after the imports, then replace
# the call to extract_reasoning to use re-decoded text.
#
# We look for the pattern where output.text is passed to extract_reasoning
# in the non-streaming (full generator) path, and replace it with our helper.

# Step 1: Inject helper after the last top-level import block
# Find a good injection point - after "from vllm" imports
import_marker = "from vllm.entrypoints.openai"
last_import_pos = content.rfind(import_marker)
if last_import_pos == -1:
    # Try a broader marker
    last_import_pos = content.rfind("from vllm.")
if last_import_pos == -1:
    print("ERROR: Could not find import block to inject helper", file=sys.stderr)
    sys.exit(1)

# Find end of that import line
end_of_line = content.index("\n", last_import_pos)
# Find end of import block (next blank line or non-import line)
pos = end_of_line + 1
while pos < len(content):
    line_end = content.find("\n", pos)
    if line_end == -1:
        line_end = len(content)
    line = content[pos:line_end].strip()
    if line and not line.startswith(("from ", "import ", "#", ")")):
        break
    pos = line_end + 1

inject_pos = pos
content = content[:inject_pos] + helper_fn + "\n" + content[inject_pos:]

# Step 2: Replace extract_reasoning calls to use re-decoded text
# The typical pattern is:
#   reasoning_parser.extract_reasoning(output.text, request=request)
# We replace output.text with _get_reasoning_text(output, tokenizer, reasoning_parser)
content = content.replace(
    "reasoning_parser.extract_reasoning(output.text,",
    "reasoning_parser.extract_reasoning(_get_reasoning_text(output, tokenizer, reasoning_parser),",
)

with open(serving_path, "w") as f:
    f.write(content)

print("Patch applied: extract_reasoning now re-decodes with special tokens when needed.")
