# Upstream 8084f0624dba24aba6ebac62d8e10cc00730a0fe:
# vllm/model_executor/layers/fused_moe/routed_experts.py

class RoutedExperts:
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        expert_mapping = self.get_expert_mapping(include_fused=True)
        mapping_by_expert = _index_expert_mapping(expert_mapping)

        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            expert_key = qual_name.partition("experts.")[2].partition(".")[0]
            named_expert = (
                mapping_by_expert.get(expert_key)
                if qual_name.count("experts.") == 1
                else None
            )
            # Encoded per-expert [N, K / block_size, bytes_per_block] weights
            # have the same rank as fused [E, N, K] weights. An explicit expert
            # name identifies the per-expert layout without treating N as E.
            is_fused = loaded_weight.dim() == 3 and named_expert is None
            # Fused tensors and ambiguous names keep the full mapping.
            candidates = expert_mapping
            if not is_fused and qual_name.count("experts.") == 1:
                candidates = named_expert or expert_mapping

            matched = False
            for param_name, weight_name, expert_id, shard_id in candidates:
                if weight_name not in qual_name:
                    if matched and is_fused:
                        # Fused tensors use the first contiguous run of matches.
                        break
                    continue
                matched = True
                is_per_expert_fused_w13 = (
                    not is_fused
                    and shard_id in {"w1", "w3"}
                    and any(
                        f".{fused_name}." in qual_name
                        for fused_name in ("gate_up_proj", "w13")
                    )
                )
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                if self.shared_expert_online_loader is not None:
                    param_name = self.shared_expert_online_loader.redirect_param_name(
                        param_name, shard_id, expert_id
                    )
                param = getattr(self, param_name, None)
                if param is None:
                    if param_name.endswith(("w13_bias", "w2_bias")):
                        continue
                    raise AttributeError(
                        f"Layer {self.layer_name} has no parameter {param_name!r} "
                        f"for checkpoint weight {qual_name!r}"
                    )
                if is_fused:
                    quant_method = getattr(param, "quant_method", None)
                    # Block scales share the weight's two-dimensional layout.
                    # Other quantization metadata can use independent layouts.
                    uses_weight_layout = (
                        "scale" not in weight_name
                        or quant_method == FusedMoeWeightScaleSupported.BLOCK.value
                    )
                    # w1 and w3 share one fused tensor; use a local copy so the
                    # transpose below doesn't mutate loaded_weight across
                    # iterations (else w3 is transposed twice and wrongly chunked)
                    fused_weight = self._orient_fused_weight(
                        loaded_weight,
                        self.is_fused_checkpoint_transposed and uses_weight_layout,
                    )
                    if shard_id in {"w1", "w3"} and self.moe_config.is_act_and_mul:
                        # Repurpose expert_id for deconcatenating w1 and w3
                        experts_shard = fused_weight.chunk(2, dim=1)[expert_id]
                    else:
                        experts_shard = fused_weight
                    start = 0
                elif is_per_expert_fused_w13:
                    shard_index = 0 if shard_id == "w1" else 1
                    experts_shard = loaded_weight.chunk(2, dim=0)[
                        shard_index
                    ].unsqueeze(0)
                    start = expert_id
                else:
                    # loaded_weight is a single expert weight, so we add a dummy expert
                    # dimension to unify the loading logic with the fused case
                    experts_shard = loaded_weight.unsqueeze(0)
                    start = expert_id

                # Unified loading logic for fused and non-fused experts
                loaded_experts = experts_shard.unbind()
                for expert_id, loaded_expert in enumerate(loaded_experts, start=start):
                    success = param.weight_loader(
                        param=param,
                        loaded_weight=loaded_expert,
                        weight_name=weight_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        logger.debug(
                            "Loaded expert %d of shard %s into %s for layer %s",
                            expert_id,
                            shard_id,
                            param_name,
                            self.layer_name,
                        )
                        yield param_name
