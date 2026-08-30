#!/bin/bash
# Vendored verbatim from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks overlay/
# (2026-08-30): kpool tail slot-map clamp (silent long-generation corruption,
# mechanism credited to vcruz305) + XGrammar termination backports (vLLM
# #52805/#53046). Both preflight pinned anchors and fail closed.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/patch_kpool_tail_slotmap.py"
python3 "$SCRIPT_DIR/patch_xgrammar_termination.py"
