#!/bin/bash
set -e

echo "Patching vLLM fused-MoE LoRA for unsupported MXFP4 backends"
python3 - <<'PY'
from pathlib import Path
import re

target = Path('/usr/local/lib/python3.12/dist-packages/vllm/lora/layers/fused_moe.py')
text = target.read_text(encoding='utf-8')
orig = text

if 'from vllm.logger import init_logger' not in text:
	text = text.replace(
		'from transformers import PretrainedConfig\n\nfrom vllm import envs\n',
		'from transformers import PretrainedConfig\n\nfrom vllm.logger import init_logger\nfrom vllm import envs\n',
		1,
	)

if 'logger = init_logger(__name__)' not in text:
	text = text.replace(
		'from .utils import _get_lora_device\n\n\nclass FusedMoEWithLoRA(BaseLayerWithLoRA):\n',
		'from .utils import _get_lora_device\n\n\nlogger = init_logger(__name__)\n\n\nclass FusedMoEWithLoRA(BaseLayerWithLoRA):\n',
		1,
	)

old_block = '''        prepare_finalize = MoEPrepareAndFinalizeNoEP()\n        m_fused_moe_fn = FusedMoEModularKernel(\n            prepare_finalize,\n            self.base_layer.quant_method.select_gemm_impl(\n                prepare_finalize, self.base_layer\n            ),\n            self.base_layer.shared_experts,\n        )\n'''

new_block = '''        prepare_finalize = MoEPrepareAndFinalizeNoEP()\n        try:\n            fused_experts_impl = self.base_layer.quant_method.select_gemm_impl(\n                prepare_finalize, self.base_layer\n            )\n        except NotImplementedError as exc:\n            # CUTLASS Blackwell MXFP4 can still serve attention-path LoRA, but\n            # fused-MoE LoRA injection is not available in this backend yet.\n            # Keep serving alive by skipping only this unsupported injection.\n            logger.warning(\n                "Skipping fused-MoE LoRA injection for layer %s due to unsupported "\n                "MXFP4 backend path: %s",\n                getattr(self.base_layer, "prefix", "<unknown>"),\n                exc,\n            )\n            return\n        m_fused_moe_fn = FusedMoEModularKernel(\n            prepare_finalize,\n            fused_experts_impl,\n            self.base_layer.shared_experts,\n        )\n'''

if old_block in text:
	text = text.replace(old_block, new_block, 1)

if text == orig:
	print('No source changes needed (already patched or upstream changed).')
else:
	target.write_text(text, encoding='utf-8')
	print('Applied fused-MoE LoRA MXFP4 hotpath patch successfully.')

mxfp4_target = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py')
mxfp4_text = mxfp4_target.read_text(encoding='utf-8')
mxfp4_orig = mxfp4_text

# Replace the LoRA fallback gate inside the `if should_quantize:` block with a
# stable logger+return path, regardless of whitespace differences upstream.
pattern = re.compile(
	r"(?ms)^([ \t]*)if should_quantize:\n"
	r"\1[ \t]*# Check LoRA compatibility - MXFP4 weights are incompatible with LoRA\n"
	r"\1[ \t]*try:\n"
	r"\1[ \t]*vllm_config = get_current_vllm_config\(\)\n"
	r"\1[ \t]*if vllm_config\.lora_config is not None:\n"
	r"\1[ \t]*logger\.warning_once\(\n"
	r"\1[ \t]*f\"\[MXFP4\] Skipping MXFP4 for \{prefix\} because LoRA is enabled\. \"\n"
	r"\1[ \t]*\"FP4-packed weights are incompatible with LoRA's additive updates\.\"\n"
	r"\1[ \t]*\)\n"
	r"\1[ \t]*return UnquantizedLinearMethod\(\)\n"
	r"\1[ \t]*except Exception:\n"
	r"\1[ \t]*pass\n"
	r"[ \t\n]*"
	r"\1[ \t]*logger\.debug\(\n"
	r"\1[ \t]*f\"\[MXFP4\] Using Mxfp4LinearMethod for \{layer_type\} \(\{prefix\}\)\"\n"
	r"\1[ \t]*\)\n"
	r"\1[ \t]*return Mxfp4LinearMethod\(\)",
)

replacement = (
	"\\1if should_quantize:\n"
	"\\1    # LoRA wrappers apply additive deltas to layer outputs, not to\n"
	"\\1    # packed base weights directly. Keep MXFP4 linear kernels enabled\n"
	"\\1    # for qkv/o when LoRA is active to avoid unnecessary BF16 fallback.\n"
	"\\1    logger.debug(\n"
	"\\1        f\"[MXFP4] Using Mxfp4LinearMethod for {layer_type} ({prefix})\"\n"
	"\\1    )\n"
	"\\1    return Mxfp4LinearMethod()"
)

mxfp4_text, replacements = pattern.subn(replacement, mxfp4_text, count=1)

if replacements == 0 and 'Skipping MXFP4 for {prefix} because LoRA is enabled' in mxfp4_text:
	print('Warning: Found LoRA fallback warning text but regex replacement did not match.')

# Remove the overly broad lm_head LoRA gate while preserving the tied-embeddings
# safety check. In this setup the adapter does not target lm_head, and vLLM's
# blanket `lora_config is not None` gate unnecessarily forces BF16 logits.
lm_head_pattern = re.compile(
	r"(?ms)^([ \t]*)# Check 1: LoRA incompatibility\n"
	r"\1[ \t]*# LoRA adapters expect to interact with unquantized weight representations\.\n"
	r"\1[ \t]*# FP4-packed weights are incompatible with LoRA's additive updates\.\n"
	r"\1[ \t]*if vllm_config\.lora_config is not None:\n"
	r"\1[ \t]*logger\.warning_once\(\n"
	r"\1[ \t]*f\"\[MXFP4\] Skipping MXFP4 quantization for lm_head \(\{prefix\}\) \"\n"
	r"\1[ \t]*\"because LoRA is enabled\. FP4-packed weights are incompatible \"\n"
	r"\1[ \t]*\"with LoRA's weight modification\. Using BF16 for lm_head instead\.\"\n"
	r"\1[ \t]*\)\n"
	r"\1[ \t]*return None\n[ \t\n]*"
)

lm_head_replacement = (
	"\\1# Check 1: LoRA compatibility\\n"
	"\\1# vLLM may enable LoRA globally even when the adapter does not target\\n"
	"\\1# lm_head. Keep MXFP4 enabled here; actual lm_head LoRA deltas, if any,\\n"
	"\\1# are applied by the dedicated logits/embedding wrappers.\\n"
)

mxfp4_text, lm_head_replacements = lm_head_pattern.subn(
	lm_head_replacement,
	mxfp4_text,
	count=1,
)

if lm_head_replacements == 0 and 'Skipping MXFP4 quantization for lm_head' in mxfp4_text:
	print('Warning: Found lm_head LoRA fallback warning text but regex replacement did not match.')

if mxfp4_text == mxfp4_orig:
	print('No MXFP4 linear LoRA-gate changes needed (already patched or upstream changed).')
else:
	mxfp4_target.write_text(mxfp4_text, encoding='utf-8')
	print('Enabled MXFP4 qkv/o/lm_head kernels under LoRA in mxfp4.py.')
PY
