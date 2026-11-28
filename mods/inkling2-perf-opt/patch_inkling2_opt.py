"""Apply env-configurable SM12x optimizations to the pinned Inkling 2 Mini vLLM build."""

from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import shutil
import sys


SUPPORTED_VLLM_VERSION = "0.1.dev25855+g7cb5dae22.cu130"

PATCHES: dict[str, tuple[tuple[str, str], ...]] = {
    "models/inkling2/nvidia/moe.py": (
        (
            "        self.sm100 = use_sm100_kernels()\n"
            "        self.sm120 = current_platform.is_device_capability_family(120)\n"
            "        vllm_config = get_current_vllm_config_or_none()\n"
            "        self.humming_nvfp4 = (\n"
            "            self.sm120\n"
            "            and self.nvfp4\n"
            "            and vllm_config is not None\n"
            '            and vllm_config.kernel_config.moe_backend == "humming"\n'
            "        )\n",
            """        import os
        self.sm100 = use_sm100_kernels()
        self.sm120 = current_platform.is_device_capability_family(120)
        vllm_config = get_current_vllm_config_or_none()
        _opt_moe = os.environ.get("INKLING2_OPT_MOE", "").strip().lower()
        _is_humming = (
            self.sm120
            and vllm_config is not None
            and vllm_config.kernel_config.moe_backend == "humming"
        )
        self.use_humming_a16 = _is_humming and (
            _opt_moe in ("w4a16", "w4a16_sk")
            or (_opt_moe == "mxfp8_only" and not self.nvfp4)
        )
        self.humming_stream_k = _opt_moe == "w4a16_sk" and self.nvfp4
        self.humming_nvfp4 = _is_humming and self.nvfp4 and not self.use_humming_a16
""",
        ),
        (
            "    def process_weights_after_loading(self) -> None:\n"
            "        if self.sm100 or (self.sm120 and not self.humming_nvfp4):\n"
            "            return\n",
            "    def process_weights_after_loading(self) -> None:\n"
            "        if self.sm100 or (self.sm120 and not self.humming_nvfp4 and not self.use_humming_a16):\n"
            "            return\n",
        ),
        (
            "        self.humming = (\n"
            "            HummingNVFP4Experts(weights, self.shapes)\n"
            "            if self.humming_nvfp4\n"
            "            else HummingExperts(weights, self.shapes, self.nvfp4)\n"
            "        )\n",
            """        self.humming = (
            HummingNVFP4Experts(weights, self.shapes)
            if self.humming_nvfp4
            else HummingExperts(weights, self.shapes, self.nvfp4)
        )
        if self.humming_stream_k and self.humming is not None:
            import json
            for _name in ("w13", "w2"):
                for _, _, _kc in self.humming.tunings[_name]:
                    _kc["use_stream_k"] = True
                self.humming.tuning_strs[_name] = json.dumps(self.humming.tunings[_name])
""",
        ),
        (
            "        if self.sm100 or self.sm120:\n"
            "            src2dst, gammas, rows, tiles = routing\n",
            "        if (self.sm100 or self.sm120) and not self.use_humming_a16:\n"
            "            src2dst, gammas, rows, tiles = routing\n",
        ),
        (
            "        if not (experts.sm100 or experts.sm120):\n",
            "        if not (experts.sm100 or experts.sm120) or experts.use_humming_a16:\n",
        ),
    ),
    "models/inkling2/nvidia/ops/moe_humming.py": (
        (
            "        sorted_ids, expert_ids, count = moe_align_block_size(\n"
            "            ids,\n"
            "            self._block_m(name, rows),\n"
            "            self.num_experts,\n"
            "            ignore_invalid_experts=True,\n"
            "        )[:3]\n",
            """        bm = self._block_m(name, rows)
        cache_key = (tiles.data_ptr(), rows, bm)
        cached = getattr(self, "_last_align", None)
        if name == "w2" and cached is not None and cached[0] == cache_key:
            sorted_ids, expert_ids, count = cached[1]
        else:
            sorted_ids, expert_ids, count = moe_align_block_size(
                ids,
                bm,
                self.num_experts,
                ignore_invalid_experts=True,
            )[:3]
            if name == "w13":
                self._last_align = (cache_key, (sorted_ids, expert_ids, count))
""",
        ),
    ),
    "v1/worker/gpu/spec_decode/dflash/utils.py": (
        (
            "    return dflash_model\n",
            """    from vllm.models.inkling2.nvidia.inkling2_humming_opt import (
        quantize_dflash_model_inplace,
    )
    quantize_dflash_model_inplace(dflash_model, target_lm_head)
    return dflash_model
""",
        ),
    ),
    "model_executor/models/qwen3_dflash.py": (
        (
            "        if self._fused_kv_weight is not None:\n"
            "            all_kv_flat = F.linear(\n"
            "                normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
            "            )\n"
            "            all_kv = all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)\n",
            """        _h_kv_op = getattr(self, "_humming_fused_kv_op", None)
        if _h_kv_op is not None:
            all_kv_flat = _h_kv_op(normed_context_states)
            all_kv = all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
        elif self._fused_kv_weight is not None:
            all_kv_flat = F.linear(
                normed_context_states, self._fused_kv_weight, self._fused_kv_bias
            )
            all_kv = all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
""",
        ),
    ),
    "models/inkling2/nvidia/attention.py": (
        (
            "    def process_weights_after_loading(self) -> None:\n"
            "        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)\n",
            """    def process_weights_after_loading(self) -> None:
        self.rel_proj_shifted = shift_rel_proj(self.rel_logits_proj, self.is_local)
        import os
        _td = os.environ.get("INKLING2_OPT_TARGET_DENSE", "").strip().lower()
        self._humming_qkvr_full = None
        self._humming_q_side = None
        self._humming_k = None
        self._humming_v = None
        self._humming_wo = None
        if _td in ("mxfp8", "mxfp8_all"):
            from .inkling2_humming_opt import HummingDenseOp
            w = self.qkvr.weight.data
            q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
            k_end = q_side + kv_size
            self._humming_qkvr_full = HummingDenseOp(w, mode="mxfp8", use_stream_k=False)
            self._humming_q_side = HummingDenseOp(w[: sum(self.q_side_splits)], mode="mxfp8", use_stream_k=False)
            self._humming_k = HummingDenseOp(w[q_side:k_end], mode="mxfp8", use_stream_k=False)
            self._humming_v = HummingDenseOp(w[k_end : k_end + kv_size], mode="mxfp8", use_stream_k=False)
            self._humming_wo = HummingDenseOp(self.wo_ud.weight.data, mode="mxfp8", use_stream_k=False)
            self.qkvr.weight.data = w.new_empty((0,))
            self.wo_ud.weight.data = self.wo_ud.weight.data.new_empty((0,))
""",
        ),
        (
            "        weight = self.qkvr.weight\n"
            "        if x_k is x_q and x_v is x_q:\n"
            "            projected = linear(x_q, weight)\n"
            "            add_linear_lora(self.qkvr, projected, x_q, range(6))\n"
            "            return list(projected.split(self.split_sizes, dim=-1)[:6])\n"
            "        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]\n"
            "        k_end = q_side + kv_size\n"
            "        projected = linear(x_q, weight[: sum(self.q_side_splits)])\n"
            "        k = linear(x_k, weight[q_side:k_end])\n"
            "        v = linear(x_v, weight[k_end : k_end + kv_size])\n",
            """        if getattr(self, "_humming_qkvr_full", None) is not None:
            if x_k is x_q and x_v is x_q:
                projected = self._humming_qkvr_full(x_q)
                return list(projected.split(self.split_sizes, dim=-1)[:6])
            projected = self._humming_q_side(x_q)
            k = self._humming_k(x_k)
            v = self._humming_v(x_v)
            return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]
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
""",
        ),
        (
            "        out = linear(output, self.wo_ud.weight, out)\n",
            """        if getattr(self, "_humming_wo", None) is not None:
            out = self._humming_wo(output)
        else:
            out = linear(output, self.wo_ud.weight, out)
""",
        ),
    ),
    "models/inkling2/nvidia/model.py": (
        (
            "        if (\n"
            "            self.head_dtype == torch.float32\n"
            "            and isinstance(\n"
            "                lm_head.quant_method,\n"
            "                (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),\n"
            "            )\n"
            "            and use_sm12x_bf16_gemv(hidden_states, lm_head.weight, embedding_bias)\n"
            "        ):\n"
            "            return sm12x_bf16_gemv(\n"
            "                hidden_states, lm_head.weight, embedding_bias, out_dtype=torch.float32\n"
            "            )\n",
            """        import os
        if os.environ.get("INKLING2_OPT_TARGET_DENSE", "").strip().lower() == "mxfp8_all":
            op = getattr(lm_head, "_humming_target_lmhead_op", None)
            if op is None:
                from .inkling2_humming_opt import HummingDenseOp
                op = HummingDenseOp(
                    lm_head.weight.data, mode="mxfp8", use_stream_k=False, slice_n=True
                )
                lm_head._humming_target_lmhead_op = op
            out = op(hidden_states)
            return out.to(self.head_dtype) if self.head_dtype is not None else out
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
""",
        ),
    ),
}


def apply(root: Path, mod_dir: Path | None = None) -> None:
    if mod_dir is None:
        mod_dir = Path(__file__).resolve().parent

    staged: dict[Path, str] = {}
    for relative, replacements in PATCHES.items():
        path = root / relative
        original = updated = path.read_text()
        for before, after in replacements:
            if updated.count(after) == 1:
                continue
            if updated.count(before) != 1:
                raise ValueError(
                    f"Unexpected source in {relative}: {before.splitlines()[0]}"
                )
            updated = updated.replace(before, after, 1)
        compile(updated, str(path), "exec")
        if original != updated:
            staged[path] = updated

    helper_src = mod_dir / "inkling2_humming_opt.py"
    compile(helper_src.read_text(), str(helper_src), "exec")
    helper_dst = root / "models/inkling2/nvidia/inkling2_humming_opt.py"
    helper_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(helper_src, helper_dst)

    for path, updated in staged.items():
        temporary = path.with_suffix(".py.inkling2-perf-opt-tmp")
        temporary.write_text(updated)
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)

    result = "Optimizations applied." if staged else "Optimizations already applied."
    print(f"[inkling2-perf-opt] {result}", flush=True)


if __name__ == "__main__":
    try:
        installed = version("vllm")
        if installed != SUPPORTED_VLLM_VERSION:
            raise ValueError(f"Unsupported vLLM build: {installed}")
        spec = find_spec("vllm")
        if spec is None or spec.origin is None:
            raise ValueError("Could not locate the vLLM package")
        apply(Path(spec.origin).parent)
    except (OSError, ValueError, SyntaxError) as error:
        sys.exit(f"[inkling2-perf-opt] {error}")
