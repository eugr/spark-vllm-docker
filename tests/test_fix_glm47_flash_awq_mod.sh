#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MOD_DIR="$PROJECT_DIR/mods/fix-glm-4.7-flash-AWQ"
MOD="$MOD_DIR/run.sh"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

TRITON_REL="v1/attention/backends/mla/triton_mla.py"
MLA_REL="model_executor/layers/attention/mla_attention.py"

# Network access must never be needed; fail if anything tries to download.
mkdir -p "$TMP_DIR/bin"
printf '#!/bin/bash\necho "curl must not be called" >&2\nexit 99\n' > "$TMP_DIR/bin/curl"
chmod +x "$TMP_DIR/bin/curl"
export PATH="$TMP_DIR/bin:$PATH"

fail() {
    echo "[FAIL] $*" >&2
    exit 1
}

# Reconstruct the pre-patch source a unified diff expects (context and
# removed lines), padded so hunks land near their recorded line numbers.
preimage() {
    python3 - "$1" "$2" <<'PY'
import re
import sys
from pathlib import Path

patch_path, target = sys.argv[1], sys.argv[2]
out = []
current = None
for line in Path(patch_path).read_text().splitlines():
    if line.startswith("+++ "):
        current = line[4:].split("/", 1)[1]
        continue
    if line.startswith("--- ") or line.startswith("diff "):
        continue
    if current != target:
        continue
    m = re.match(r"@@ -(\d+)", line)
    if m:
        start = int(m.group(1))
        while len(out) < start - 1:
            out.append(f"# filler {len(out) + 1}")
        continue
    if line.startswith("+"):
        continue
    if line.startswith("-") or line.startswith(" "):
        out.append(line[1:])
    elif line == "":
        out.append("")
print("\n".join(out))
PY
}

make_root() {
    local root="$1/vllm"
    mkdir -p "$root/$(dirname "$TRITON_REL")" "$root/$(dirname "$MLA_REL")"
    echo "$root"
}

run_mod() {
    VLLM_PACKAGE_ROOT="$1" bash "$MOD"
}

# (a) Old upstream source: both patches apply, and a repeat run is a no-op.
root=$(make_root "$TMP_DIR/old")
preimage "$MOD_DIR/glm47_flash.patch" "vllm/$TRITON_REL" > "$root/$TRITON_REL"
preimage "$MOD_DIR/glm47_vllm_bug.patch" "vllm/$MLA_REL" > "$root/$MLA_REL"
out=$(run_mod "$root") || fail "(a) mod failed on old source: $out"
grep -Fq 'max(32, min(128, max_seq_len // 1500))' "$root/$TRITON_REL" \
    || fail "(a) speed patch not applied"
grep -Fq 'and hasattr(self.kv_b_proj, "weight")' "$root/$MLA_REL" \
    || fail "(a) crash patch not applied"
before=$(sha256sum "$root/$TRITON_REL" "$root/$MLA_REL")
out=$(run_mod "$root") || fail "(a) repeat run failed: $out"
grep -Fq 'Speed patch is already applied' <<< "$out" || fail "(a) repeat not detected"
grep -Fq 'PR 34695 is already applied' <<< "$out" || fail "(a) repeat not detected"
test "$before" = "$(sha256sum "$root/$TRITON_REL" "$root/$MLA_REL")" \
    || fail "(a) repeat run modified files"

# (b) Current upstream triton_mla.py: speed patch is skipped, crash patch
# is still attempted and applied.
root=$(make_root "$TMP_DIR/new")
printf '%s\n' \
    'def _compute_num_kv_splits(max_seq_len: int, sm_count: int) -> int:' \
    '    return 1' \
    '' \
    'class TritonMLAImpl:' \
    '    def forward_mqa(self):' \
    '        # For batch invariance, use only 1 split to ensure deterministic reduction' \
    '        if envs.VLLM_BATCH_INVARIANT:' \
    '            num_kv_splits = 1' \
    '        else:' \
    '            num_kv_splits = _compute_num_kv_splits(' \
    '                attn_metadata.max_seq_len, self._sm_count' \
    '            )' \
    > "$root/$TRITON_REL"
preimage "$MOD_DIR/glm47_vllm_bug.patch" "vllm/$MLA_REL" > "$root/$MLA_REL"
triton_before=$(sha256sum "$root/$TRITON_REL")
out=$(run_mod "$root") || fail "(b) mod failed on superseded source: $out"
grep -Fq '_compute_num_kv_splits), skipping' <<< "$out" || fail "(b) skip message missing"
test "$triton_before" = "$(sha256sum "$root/$TRITON_REL")" \
    || fail "(b) superseded triton_mla.py was modified"
grep -Fq 'and hasattr(self.kv_b_proj, "weight")' "$root/$MLA_REL" \
    || fail "(b) crash patch not applied after skipping speed patch"

# (c) mla_attention.py already contains the PR 34695 fix.
root=$(make_root "$TMP_DIR/merged")
preimage "$MOD_DIR/glm47_flash.patch" "vllm/$TRITON_REL" > "$root/$TRITON_REL"
printf '%s\n' \
    '        self.is_aiter_triton_fp4_bmm_enabled = (' \
    '            rocm_aiter_ops.is_fp4bmm_enabled()' \
    '            and hasattr(self.kv_b_proj, "weight")' \
    '            and self.kv_b_proj.weight.dtype == torch.bfloat16' \
    '        )' \
    > "$root/$MLA_REL"
mla_before=$(sha256sum "$root/$MLA_REL")
out=$(run_mod "$root") || fail "(c) mod failed: $out"
grep -Fq 'PR 34695 is already applied, skipping' <<< "$out" || fail "(c) skip message missing"
test "$mla_before" = "$(sha256sum "$root/$MLA_REL")" || fail "(c) mla_attention.py was modified"
grep -Fq 'max(32, min(128, max_seq_len // 1500))' "$root/$TRITON_REL" \
    || fail "(c) speed patch not applied"

# (d) Unrecognized triton_mla.py: fail loudly and leave it untouched.
root=$(make_root "$TMP_DIR/unknown")
printf '%s\n' 'def forward_mqa(self):' '    num_kv_splits = 8' > "$root/$TRITON_REL"
preimage "$MOD_DIR/glm47_vllm_bug.patch" "vllm/$MLA_REL" > "$root/$MLA_REL"
triton_before=$(sha256sum "$root/$TRITON_REL")
if out=$(run_mod "$root" 2>&1); then
    fail "(d) mod accepted an unexpected triton_mla.py"
fi
grep -Fq 'unexpected state' <<< "$out" || fail "(d) error message missing: $out"
test "$triton_before" = "$(sha256sum "$root/$TRITON_REL")" \
    || fail "(d) unexpected triton_mla.py was modified"
ls "$root/$(dirname "$TRITON_REL")" | grep -q '\.rej$\|\.orig$' \
    && fail "(d) patch left reject or backup files"

echo '[PASS] fix-glm-4.7-flash-AWQ mod skips superseded patches and fails closed'
