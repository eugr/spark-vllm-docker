#!/bin/bash
set -e

# Fix: vLLM unconditionally passes reasoning_effort to apply_chat_template,
# but mistral_common <1.11 doesn't support it, causing 400 on every request.
# Workaround until PR #37081 lands in a release.
#
# Safe to apply on any version: skips if already patched or not applicable.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "Applying fix-mistral-reasoning-effort patch..."
if patch --dry-run -p1 -d "$SITE_PACKAGES" < "$SCRIPT_DIR/fix.diff" &>/dev/null; then
    patch -p1 -d "$SITE_PACKAGES" < "$SCRIPT_DIR/fix.diff"
    echo "Patch applied successfully."
else
    echo "Patch not applicable (already applied or code has changed), skipping."
fi
