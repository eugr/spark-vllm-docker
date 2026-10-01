#!/bin/bash
set -euo pipefail

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
MODELOPT="$SITE_PACKAGES/vllm/model_executor/layers/quantization/modelopt.py"

cd "$SITE_PACKAGES"

echo "[fix-modelopt-mixed-mxfp8] Patching ModelOpt mixed-precision MXFP8 dispatch"

python3 - <<'PY'
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py')
text = path.read_text()
orig = text

# 1) ModelOpt MXFP8 checkpoints commonly name the E8M0 scale tensor
#    weight_scale_inv, while ModelOptMxFp8LinearMethod expects weight_scale.
#    Register a duplicate parameter alias so the existing checkpoint loader can
#    populate the same tensor through either name.
old = '''        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Validate weight tensor
'''
new = '''        layer.register_parameter("weight_scale", weight_scale)
        # ModelOpt MIXED_PRECISION checkpoints such as MiMo-V2.5-NVFP4 store
        # MXFP8 microscale tensors as `.weight_scale_inv`.  The MXFP8 linear
        # method consumes `layer.weight_scale`, so expose the same Parameter
        # under both names.  `named_parameters(remove_duplicate=False)` used by
        # vLLM loading will then find the checkpoint name and load into the
        # tensor consumed by the MXFP8 kernel/emulation path.
        if "weight_scale_inv" not in layer._parameters:
            layer.register_parameter("weight_scale_inv", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Validate weight tensor
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-modelopt-mixed-mxfp8] ERROR: MXFP8 weight_scale anchor not found')
    text = text.replace(old, new, 1)
else:
    print('[fix-modelopt-mixed-mxfp8] weight_scale_inv alias already present')

# 2) Add an MXFP8 sub-config to ModelOptMixedPrecisionConfig.
old = '''        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
    ) -> None:
        super().__init__(exclude_modules)
        self.kv_cache_quant_method = kv_cache_quant_method
        self.quantized_layers = quantized_layers
        self.fp8_config = fp8_config
        self.nvfp4_config = nvfp4_config
'''
new = '''        quantized_layers: dict[str, dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        mxfp8_config: ModelOptMxFp8Config,
        nvfp4_config: ModelOptNvFp4Config,
    ) -> None:
        super().__init__(exclude_modules)
        self.kv_cache_quant_method = kv_cache_quant_method
        self.quantized_layers = quantized_layers
        self.fp8_config = fp8_config
        self.mxfp8_config = mxfp8_config
        self.nvfp4_config = nvfp4_config
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-modelopt-mixed-mxfp8] ERROR: mixed __init__ anchor not found')
    text = text.replace(old, new, 1)
else:
    print('[fix-modelopt-mixed-mxfp8] mixed __init__ already patched')

# 3) Construct that MXFP8 config in _from_config().
old = '''        nvfp4_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
            group_size=group_size,
        )

        return cls(
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
        )
'''
new = '''        mxfp8_config = ModelOptMxFp8Config(
            is_checkpoint_mxfp8_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
        )
        nvfp4_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_method,
            exclude_modules=[],
            group_size=group_size,
        )

        return cls(
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            mxfp8_config=mxfp8_config,
            nvfp4_config=nvfp4_config,
        )
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-modelopt-mixed-mxfp8] ERROR: mixed _from_config anchor not found')
    text = text.replace(old, new, 1)
else:
    print('[fix-modelopt-mixed-mxfp8] mixed _from_config already patched')

# 4) Dispatch MXFP8 LinearBase layers through ModelOptMxFp8LinearMethod.
old = '''        if isinstance(layer, LinearBase):
            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            # Layer not in quantized_layers — leave unquantized
            return UnquantizedLinearMethod()
'''
new = '''        if isinstance(layer, LinearBase):
            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if quant_algo == "MXFP8":
                return ModelOptMxFp8LinearMethod(self.mxfp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4LinearMethod(self.nvfp4_config)
            # Layer not in quantized_layers — leave unquantized
            return UnquantizedLinearMethod()
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-modelopt-mixed-mxfp8] ERROR: LinearBase dispatch anchor not found')
    text = text.replace(old, new, 1)
else:
    print('[fix-modelopt-mixed-mxfp8] LinearBase MXFP8 dispatch already patched')

# 5) Dispatch MXFP8 RoutedExperts as well for completeness.  MiMo-V2.5-NVFP4
#    experts are NVFP4, but mixed ModelOpt can legitimately contain MXFP8 MoE.
old = '''        if isinstance(layer, RoutedExperts):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
                    quant_config=self.fp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.nvfp4_config,
                    moe_config=layer.moe_config,
                )
            return None
'''
new = '''        if isinstance(layer, RoutedExperts):
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
                    quant_config=self.fp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.nvfp4_config,
                    moe_config=layer.moe_config,
                )
            return None
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-modelopt-mixed-mxfp8] ERROR: RoutedExperts dispatch anchor not found')
    text = text.replace(old, new, 1)
else:
    print('[fix-modelopt-mixed-mxfp8] RoutedExperts MXFP8 dispatch already patched')

if text != orig:
    path.write_text(text)
    print('[fix-modelopt-mixed-mxfp8] patched', path)
else:
    print('[fix-modelopt-mixed-mxfp8] no changes needed')
PY

# Clear stale bytecode for the patched module.
find "$SITE_PACKAGES/vllm/model_executor/layers/quantization" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 -m py_compile "$MODELOPT"
python3 - <<'PY'
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8LinearMethod,
)

ql = {
    "model.layers.0.self_attn.qkv_proj": {"quant_algo": "MXFP8", "group_size": 32},
    "model.layers.1.mlp.experts.0.gate_proj": {"quant_algo": "NVFP4", "group_size": 16},
}
config = ModelOptMixedPrecisionConfig._from_config(
    quant_method="MIXED_PRECISION",
    kv_cache_quant_method=None,
    exclude_modules=[],
    original_config={"quantized_layers": ql},
    group_size=None,
)
assert hasattr(config, "mxfp8_config")
print("[fix-modelopt-mixed-mxfp8] validation OK: mixed config has MXFP8 dispatch support")
print("[fix-modelopt-mixed-mxfp8] MXFP8 linear method:", ModelOptMxFp8LinearMethod)
PY
