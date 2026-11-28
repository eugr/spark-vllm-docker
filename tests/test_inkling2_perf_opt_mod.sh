#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MOD_DIR="$PROJECT_DIR/mods/inkling2-perf-opt"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

python3 - "$MOD_DIR" "$TMP_DIR" <<'PY'
import importlib.util
from pathlib import Path
import sys

mod_dir = Path(sys.argv[1])
root = Path(sys.argv[2])

spec = importlib.util.spec_from_file_location(
    "patch_inkling2_opt", mod_dir / "patch_inkling2_opt.py"
)
assert spec is not None and spec.loader is not None
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

# Create minimal syntactically valid fixture files containing every expected target hunk.
fixtures = {
    "models/inkling2/nvidia/moe.py": """class Inkling2Experts:
    def __init__(self):
        self.sm100 = use_sm100_kernels()
        self.sm120 = current_platform.is_device_capability_family(120)
        vllm_config = get_current_vllm_config_or_none()
        self.humming_nvfp4 = (
            self.sm120
            and self.nvfp4
            and vllm_config is not None
            and vllm_config.kernel_config.moe_backend == "humming"
        )

    def process_weights_after_loading(self) -> None:
        if self.sm100 or (self.sm120 and not self.humming_nvfp4):
            return
        self.humming = (
            HummingNVFP4Experts(weights, self.shapes)
            if self.humming_nvfp4
            else HummingExperts(weights, self.shapes, self.nvfp4)
        )

    def forward(self):
        if self.sm100 or self.sm120:
            src2dst, gammas, rows, tiles = routing

class Inkling2MoE:
    def forward(self):
        if not (experts.sm100 or experts.sm120):
            return None
""",
    "models/inkling2/nvidia/ops/moe_humming.py": """class HummingNVFP4Experts:
    def bmm(self, name, ids, rows):
        sorted_ids, expert_ids, count = moe_align_block_size(
            ids,
            self._block_m(name, rows),
            self.num_experts,
            ignore_invalid_experts=True,
        )[:3]
        return sorted_ids, expert_ids, count
""",
    "v1/worker/gpu/spec_decode/dflash/utils.py": """def load_dflash_model(target_model, vllm_config):
    dflash_model = None
    target_lm_head = None
    return dflash_model
""",
    "model_executor/models/qwen3_dflash.py": """class DFlashQwen3Model:
    def _project_context_kv_token_major(self, normed_context_states):
        if self._fused_kv_weight is not None:
            all_kv_flat = F.linear(
                normed_context_states, self._fused_kv_weight, self._fused_kv_bias
            )
            all_kv = all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
        return all_kv
""",
    "models/inkling2/nvidia/attention.py": """class Inkling2Attention:
    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)

    def _project(self, x_q, x_k, x_v):
        weight = self.qkvr.weight
        if x_k is x_q and x_v is x_q:
            projected = linear(x_q, weight)
            add_linear_lora(self.qkvr, projected, x_q, range(6))
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        projected = linear(x_q, weight[: sum(self.q_side_splits)])
        k = linear(x_k, weight[q_side:k_end])
        v = linear(x_v, weight[k_end : k_end + kv_size])
        return projected, k, v

    def forward(self, output, out):
        out = linear(output, self.wo_ud.weight, out)
        return out
""",
    "models/inkling2/nvidia/model.py": """class Inkling2LogitsProcessor:
    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        if (
            self.head_dtype == torch.float32
            and isinstance(
                lm_head.quant_method,
                (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
            )
            and use_sm12x_bf16_gemv(hidden_states, lm_head.weight, embedding_bias)
        ):
            return sm12x_bf16_gemv(
                hidden_states, lm_head.weight, embedding_bias, out_dtype=torch.float32
            )
        return None
""",
}

for rel, content in fixtures.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)

# First apply must patch all files and install helper module.
patcher.apply(root, mod_dir)
helper_dst = root / "models/inkling2/nvidia/inkling2_humming_opt.py"
assert helper_dst.exists(), "Helper module was not installed"

# Second apply must be idempotent.
before = {rel: (root / rel).read_text() for rel in fixtures}
patcher.apply(root, mod_dir)
after = {rel: (root / rel).read_text() for rel in fixtures}
assert before == after, "Reapplying patch modified files"

# Fail closed when a source file does not match.
bad_root = root / "bad"
for rel, content in fixtures.items():
    p = bad_root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# mismatched\n" if rel.endswith("moe.py") else content)
try:
    patcher.apply(bad_root, mod_dir)
    raise AssertionError("Expected ValueError on mismatched source")
except ValueError:
    pass
PY

echo "[PASS] inkling2-perf-opt mod applies cleanly, is idempotent, and fails closed"
