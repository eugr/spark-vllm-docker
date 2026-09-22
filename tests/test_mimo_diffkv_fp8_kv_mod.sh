#!/bin/bash
set -euo pipefail

# Focused check for mods/mimo-diffkv-fp8-kv/run.sh: applies the two file
# patches to fixtures modeled on vllm main, is idempotent on re-run, and keeps
# the narrowed non-e4m3 rejection.

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD="$PROJECT_DIR/mods/mimo-diffkv-fp8-kv/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

ROOT="$TMP_DIR/site"
mkdir -p "$ROOT/vllm/model_executor/models" "$ROOT/vllm/v1/attention/backends"
MIMO="$ROOT/vllm/model_executor/models/mimo_v2.py"
DIFFKV="$ROOT/vllm/v1/attention/backends/triton_attn_diffkv.py"

cat > "$MIMO" <<'EOF'
from vllm.utils import (
    get_current_vllm_config,
)
class MiMoV2Attention(nn.Module):
    def __init__(self, cache_config=None):
        requested = get_current_vllm_config().attention_config.backend
        self.attn = Attention(
            self.num_heads,
            cache_config=cache_config,
        )
EOF

cat > "$DIFFKV" <<'EOF'
class TritonAttentionDiffKVBackend(TritonAttentionBackend):
    # No FP8 / int8 KV cache for the DiffKV path yet; require fp16/bf16/fp32.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

class TritonAttentionDiffKVImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonAttentionDiffKVBackend does not yet support quantized "
                f"KV cache (got kv_cache_dtype={self.kv_cache_dtype!r})."
            )

    def forward(self, ...):
        # Triton DiffKV kernels consume (B, N, H, D) cache views.
        kv_cache = kv_cache.transpose(1, 2)
        key_cache = kv_cache[..., :head_size_qk]
EOF

fail() { echo "FAIL: $1" >&2; exit 1; }

# 1. First run applies both patches.
out1=$(PYTHON_ROOT="$ROOT" bash "$MOD") || fail "mod exited non-zero"
grep -qF "cache_config = get_current_vllm_config().cache_config" "$MIMO" \
    || fail "cache_config wiring missing"
grep -qF '"fp8_e4m3",' "$DIFFKV" || fail "fp8 dtypes not added"
grep -qF 'self.kv_cache_dtype not in ("fp8", "fp8_e4m3")' "$DIFFKV" \
    || fail "non-e4m3 rejection not narrowed correctly"
grep -qF "kv_cache = kv_cache.view(self.fp8_dtype)" "$DIFFKV" \
    || fail "fp8 view on read missing"
grep -qF "Patched mimo_v2.py." <<< "$out1" || fail "missing patch report (mimo)"
grep -qF "Patched triton_attn_diffkv.py." <<< "$out1" || fail "missing patch report (diffkv)"

# 2. Second run is a no-op (idempotent within a fresh container).
sum1=$(cat "$MIMO" "$DIFFKV" | sha256sum)
out2=$(PYTHON_ROOT="$ROOT" bash "$MOD") || fail "re-run exited non-zero"
sum2=$(cat "$MIMO" "$DIFFKV" | sha256sum)
[ "$sum1" = "$sum2" ] || fail "re-run modified already-patched files"
grep -qF "already patched; skipping" <<< "$out2" || fail "missing skip report"

# 3. Missing anchor fails fast instead of writing garbage.
cat > "$MIMO" <<'EOF'
def something_else():
    pass
EOF
if PYTHON_ROOT="$ROOT" bash "$MOD" 2>/dev/null; then
    fail "mod should fail when the anchor is missing"
fi

echo "PASS: mimo-diffkv-fp8-kv mod"