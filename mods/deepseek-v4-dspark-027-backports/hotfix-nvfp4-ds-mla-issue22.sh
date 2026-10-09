#!/usr/bin/env bash
# hotfix-nvfp4-ds-mla-issue22.sh — Fix nvfp4_ds_mla long-context decode regression
#
# ACTUAL ROOT CAUSE: flashmla_sparse.py dispatches nvfp4_ds_mla to the slow
# _forward_bf16_kv path instead of the fast _forward_fp8_kv path.  The584-byte
# KV layout is identical for both dtypes; only the kernel dispatch differs.
#
# Usage:
#   docker exec <container> bash /path/to/hotfix-nvfp4-ds-mla-issue22.sh
#   # Then restart the vLLM process inside the container.
#
# Safe to re-run (idempotent — skips already-applied patches).
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"

if [ ! -d "$VLLM_ROOT" ]; then
  echo "ERROR: vLLM not found at $VLLM_ROOT" >&2
  exit 1
fi

echo "=== Hotfix: nvfp4_ds_mla long-context decode (Issue #22) ==="
echo "vLLM root: $VLLM_ROOT"

python3 <<'PYEOF'
import sys
from pathlib import Path

root = Path("/usr/local/lib/python3.12/dist-packages/vllm")
applied = 0
skipped = 0
errors = []

def patch(path: str, old: str, new: str, label: str) -> None:
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
    if old not in text:
        errors.append(f"[ERR] Anchor not found for {label} in {path}")
        return
    p.write_text(text.replace(old, new, 1))
    print(f"  [OK]   {label}")
    applied += 1


# ============================================================
# THE FIX: Route nvfp4_ds_mla to the fast FP8 kernel path
# ============================================================
# The584-byte KV layout is identical for fp8_ds_mla and nvfp4_ds_mla on DSV4.
# The only difference is the kernel dispatch.  nvfp4_ds_mla was incorrectly
# routed to the slow _forward_bf16_kv path.
patch(
    "v1/attention/backends/mla/flashmla_sparse.py",
    '        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"',
    '        use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")',
    "Route nvfp4_ds_mla to fast FP8 kernel path",
)


# ============================================================
# Summary
# ============================================================
print(f"\nApplied: {applied}, Skipped: {skipped}, Errors: {len(errors)}")
for e in errors:
    print(f"  {e}", file=sys.stderr)

if errors:
    print("\nWARNING: Some patches could not be applied.")
    sys.exit(1)

if applied == 0 and skipped > 0:
    print("Patch already applied. No changes needed.")
elif applied > 0:
    print("\nHotfix applied successfully. Restart the vLLM process to take effect.")
PYEOF

echo ""
echo "=== Verification ==="
python3 <<'PYEOF'
p = __import__("pathlib").Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py")
text = p.read_text()
if 'self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")' in text:
    print("[OK] nvfp4_ds_mla routed to fast FP8 kernel path")
elif 'self.kv_cache_dtype == "fp8_ds_mla"' in text:
    print("[FAIL] Still using slow BF16 path for nvfp4_ds_mla")
else:
    print("[WARN] Could not verify patch state")
PYEOF
