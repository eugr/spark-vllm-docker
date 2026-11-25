# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpts from upstream b409a4f0d787fdddb75ec2e8ddb6dfcfd36235b7 (with preceding series entries).
# vllm/model_executor/layers/quantization/modelopt.py

class ModelOptMixedPrecisionConfig(object):
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        """Return quantize-method based on layer."""
        # KV-cache quantization
        if isinstance(layer, Attention):
            if self.kv_cache_quant_method:
                return ModelOptKVCacheMethod(self)
            return None

        # Excluded layers
        if self.is_layer_excluded(prefix):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            return None

        quant_algo = self._resolve_quant_algo(prefix)

        if quant_algo is not None and quant_algo.lower() in B12X_BLOCK_CODECS:
            from .modelopt_block_quant import (
                ModelOptBlockQuantLinearMethod,
                ModelOptBlockQuantMoEMethod,
            )

            if isinstance(layer, RoutedExperts):
                return ModelOptBlockQuantMoEMethod(
                    layer.moe_config, codec=quant_algo.lower()
                )
            if isinstance(layer, (LinearBase, VocabParallelEmbedding)):
                return ModelOptBlockQuantLinearMethod(codec=quant_algo.lower())
            raise ValueError(
                f"{quant_algo} requires a linear, embedding or routed experts"
            )

        if (
            quant_algo == "W4A16_NVFP4"
            and isinstance(layer, VocabParallelEmbedding)
            and not isinstance(layer, ParallelLMHead)
        ):
            model_config = getattr(
                get_current_vllm_config_or_none(), "model_config", None
            )
            # The language model ties according to its own (text) config.
            if model_config is not None and getattr(
                model_config.hf_text_config, "tie_word_embeddings", False
            ):
                raise ValueError(_NVFP4_EMBEDDING_TIED)
            return ModelOptNvFp4EmbeddingMethod()

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            # Per-prefix algo -> its sub-config, then the generic linear method.
            # resolve() reads group_size off whichever sub-config it is handed.
            # Read-only — never write back, or a linear-only change leaks into
            # the MoE method that shares the sub-config.
            if quant_algo is None or quant_algo not in LINEAR_ALGOS:
                # Layer not in quantized_layers — leave unquantized
                return UnquantizedLinearMethod()
            _, subcfg_attr = LINEAR_ALGOS[quant_algo]
            return build_linear_method(getattr(self, subcfg_attr), quant_algo, prefix)

        if isinstance(layer, RoutedExperts):
            _reject_int8_pb_wo_moe(quant_algo, prefix)
            if quant_algo in _BLOCK_FP8_MOE_ALGOS:
                return Fp8MoEMethod(self.fp8_block_config, layer)
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(
                    quant_config=self.fp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.nvfp4_config,
                    moe_config=layer.moe_config,
                    scale_encoding=self._resolve_layer_field(
                        prefix, "weight_scale_encoding"
                    ),
                )
            if quant_algo == "W4A16_NVFP4":
                return ModelOptNvFp4FusedMoE(
                    quant_config=self.w4a16_nvfp4_config,
                    moe_config=layer.moe_config,
                    scale_encoding=self._resolve_layer_field(
                        prefix, "weight_scale_encoding"
                    ),
                )
            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None

        return None
