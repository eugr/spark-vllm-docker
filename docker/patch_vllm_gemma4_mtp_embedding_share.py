#!/usr/bin/env python3
"""Keep the EAGLE embedding-width guard from breaking Gemma4 MTP."""

import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "vllm/v1/spec_decode/llm_base_proposer.py"
old = """            if share_embeddings:
                draft_embed = self.model.model.embed_tokens
                # Only share when both models use the same embedding width.
                # Guard with isinstance so non-Tensor weights (e.g. in tests)
"""
new = (
    '            if share_embeddings and hasattr(self.model, "has_own_embed_tokens"):\n'
    "                draft_embed = self.model.model.embed_tokens\n"
    "                # Only share when both models use the same embedding width.\n"
    "                # Guard with isinstance so non-Tensor weights (e.g. in tests)\n"
)

if not target.exists():
    print(f"{target} not found; skipping Gemma4 MTP embedding-share workaround")
else:
    text = target.read_text()
    if 'if share_embeddings and hasattr(self.model, "has_own_embed_tokens"):' in text:
        print("Gemma4 MTP embedding-share workaround already present; skipping")
    elif old in text:
        target.write_text(text.replace(old, new, 1))
        print("Applied Gemma4 MTP embedding-share workaround")
    else:
        print("Known Gemma4 MTP embedding-share pattern not found; skipping")
