#!/bin/bash
set -euo pipefail

PREFIX="[inkling-sm12-paged-kv]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
VLLM_ROOT="$PYTHON_ROOT/vllm"
VENDOR_SOURCE="$MOD_DIR/vendor/inkling_sm120_fa4"
VENDOR_TARGET="$VLLM_ROOT/third_party/inkling_sm120_fa4"
ADAPTER_SOURCE="$MOD_DIR/adapter.py"
ADAPTER_TARGET="$VLLM_ROOT/third_party/inkling_sm120_fa4_adapter.py"
PATCHER="$MOD_DIR/patch_inkling.py"

echo "=== Inkling SM12 paged-KV FA4 mod ==="

if [[ ! -d "$VLLM_ROOT" ]]; then
    echo "$PREFIX vLLM package not found at $VLLM_ROOT" >&2
    exit 1
fi

for required in "$VENDOR_SOURCE/interface.py" "$VENDOR_SOURCE/LICENSE" \
                "$ADAPTER_SOURCE" "$PATCHER"; do
    if [[ ! -f "$required" ]]; then
        echo "$PREFIX required mod file not found: $required" >&2
        exit 1
    fi
done

TARGET=""
for candidate in \
    "$VLLM_ROOT/models/inkling/nvidia/ops/fa4_rel_attention.py" \
    "$VLLM_ROOT/model_executor/models/inkling/nvidia/ops/fa4_rel_attention.py"; do
    if [[ -f "$candidate" ]]; then
        TARGET="$candidate"
        break
    fi
done

if [[ -z "$TARGET" ]]; then
    echo "$PREFIX Inkling NVIDIA FA4 module was not found under $VLLM_ROOT." >&2
    echo "$PREFIX This mod requires a vLLM build with Inkling FA4 relative attention." >&2
    exit 1
fi

# Refuse an unknown Inkling implementation before installing anything.
python3 "$PATCHER" --check "$TARGET"

if [[ "${INKLING_SM12_MOD_SKIP_RUNTIME_CHECK:-0}" != "1" ]]; then
    python3 - <<'PY'
import importlib
import sys

try:
    import torch
    for module in ("cuda.bindings.driver", "cutlass", "einops", "quack", "tvm_ffi"):
        importlib.import_module(module)
    import cutlass.cute as cute
except Exception as exc:
    print(
        "[inkling-sm12-paged-kv ERROR] FA4 Python dependency preflight failed: "
        f"{type(exc).__name__}: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)

missing_cute_apis = [
    name for name in ("ThrMma", "make_rmem_tensor") if not hasattr(cute, name)
]
if missing_cute_apis:
    print(
        "[inkling-sm12-paged-kv ERROR] CUTLASS DSL is missing required APIs: "
        + ", ".join(missing_cute_apis),
        file=sys.stderr,
    )
    raise SystemExit(1)

if not torch.cuda.is_available():
    print("[inkling-sm12-paged-kv ERROR] CUDA is unavailable in the mod container.", file=sys.stderr)
    raise SystemExit(1)

capability = torch.cuda.get_device_capability()
if capability[0] != 12:
    print(
        "[inkling-sm12-paged-kv ERROR] this mod is restricted to SM12x; "
        f"detected SM{capability[0]}{capability[1]}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

cuda_version = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
if cuda_version < (12, 8):
    print(
        "[inkling-sm12-paged-kv ERROR] CUDA 12.8+ is required; "
        f"PyTorch reports CUDA {torch.version.cuda}.",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(
    f"[inkling-sm12-paged-kv] Runtime preflight passed: "
    f"SM{capability[0]}{capability[1]}, CUDA {torch.version.cuda}."
)
PY
else
    echo "$PREFIX Skipping CUDA/dependency preflight (test override)."
fi

python3 - "$VENDOR_SOURCE" "$VENDOR_TARGET" "$ADAPTER_SOURCE" "$ADAPTER_TARGET" <<'PY'
from pathlib import Path
import shutil
import sys

vendor_source = Path(sys.argv[1])
vendor_target = Path(sys.argv[2])
adapter_source = Path(sys.argv[3])
adapter_target = Path(sys.argv[4])

vendor_target.mkdir(parents=True, exist_ok=True)
for source in vendor_source.rglob("*"):
    if not source.is_file() or "__pycache__" in source.parts:
        continue
    relative = source.relative_to(vendor_source)
    destination = vendor_target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
shutil.copy2(adapter_source, adapter_target)
print(f"[inkling-sm12-paged-kv] Installed vendored FA4 bundle at {vendor_target}.")
PY

if [[ "${INKLING_SM12_MOD_SKIP_RUNTIME_CHECK:-0}" != "1" ]]; then
    python3 - <<'PY'
from vllm.third_party.inkling_sm120_fa4 import __version__
from vllm.third_party.inkling_sm120_fa4_adapter import flash_attn_varlen_func

assert callable(flash_attn_varlen_func)
print(
    "[inkling-sm12-paged-kv] Vendored package import passed "
    f"(bundle version {__version__})."
)
PY
fi

# Patch Inkling only after the complete vendored package imports successfully.
python3 "$PATCHER" "$TARGET"

find "$(dirname "$TARGET")" "$VENDOR_TARGET" -name "__pycache__" \
    -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: Inkling alone will use the vendored SM12 paged-KV FA4 kernel ==="
