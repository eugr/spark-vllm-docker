# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated-input expert parallel adapter for the b12x W4A16 MoE path."""

from collections.abc import Iterable
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x_moe import (
    B12xExperts,
    _b12x_scratch_nbytes,
    _ceil_div,
    _dtype_element_size,
    _is_current_stream_capturing,
    _normalize_b12x_moe_topk_ids,
    _normalize_b12x_moe_topk_weights,
    _workspace2_as_b12x_scratch,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey


def _plan_b12x_ep_moe_fp4_scratch(
    *,
    tokens: int,
    topk: int,
    global_num_experts: int,
    device: torch.device,
    experts: Any,
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
):
    from b12x.moe.ep_moe import Caps as EPMoEScratchCaps
    from b12x.moe.ep_moe import plan as plan_ep_moe_scratch

    return plan_ep_moe_scratch(
        EPMoEScratchCaps(
            max_tokens=max(int(tokens), 1),
            num_topk=int(topk),
            global_num_experts=int(global_num_experts),
            device=device,
            weight_plan=experts.plan,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
    )


def _run_b12x_ep_moe_fp4(
    *,
    a: torch.Tensor,
    experts: Any,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: Any,
    plan: Any,
    scratch: torch.Tensor,
) -> None:
    from b12x.moe.ep_moe import run as b12x_ep_moe_fp4

    binding = plan.bind(
        scratch=scratch,
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        output=output,
    )
    b12x_ep_moe_fp4(binding=binding)


class B12xEPExperts(B12xExperts):
    """Conservative b12x specialization for vLLM's no-alltoall EP contract.

    Each rank consumes replicated standard-format activations and global route
    ids, computes only its local experts, and leaves the final all-reduce to
    ``MoERunner``.  DeepEP/NIXL batched formats, DP+EP, SP, and EPLB are not
    part of this specialization.
    """

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self._prepared_ep_expert_map: Any | None = None

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.in_dtype != torch.bfloat16:
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )
        if moe_config.num_experts < moe_config.moe_parallel_config.ep_size:
            return (
                False,
                "kernel requires at least one local expert on every EP rank",
            )
        return mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )

    @staticmethod
    def _supports_current_device() -> bool:
        if not B12xExperts._supports_current_device():
            return False
        try:
            from b12x.moe.ep_moe import run as b12x_ep_moe_fp4  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            moe_parallel_config.use_ep
            and moe_parallel_config.ep_size > 1
            and moe_parallel_config.tp_size == 1
            and moe_parallel_config.dp_size == 1
            and moe_parallel_config.pcp_size == 1
            and moe_parallel_config.sp_size == 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    def _quant_mode(self) -> str:
        # EP intentionally stays on the mapped W4A16 route-pack path.  The
        # NVFP4 dynamic/micro paths index route ids directly and are TP-only.
        return "w4a16"

    def _activation_amax_enabled_for_layer(self) -> bool:
        # Activation calibration has a local-expert indexing contract that is
        # intentionally outside this first EP specialization.
        return False

    def supports_expert_map(self) -> bool:
        return True

    def _prepare_ep_expert_map(
        self,
        expert_map: torch.Tensor,
        *,
        global_num_experts: int,
        local_num_experts: int,
        device: torch.device,
    ):
        prepared = getattr(self, "_prepared_ep_expert_map", None)
        if prepared is not None:
            if (
                prepared.tensor is expert_map
                and prepared.global_num_experts == int(global_num_experts)
                and prepared.local_num_experts == int(local_num_experts)
                and prepared.device == device
            ):
                prepared.validate_static()
                return prepared
            if _is_current_stream_capturing():
                raise RuntimeError(
                    "B12X EP expert_map changed before/during CUDA graph capture"
                )

        from b12x.moe.ep_moe import prepare_expert_map as prepare_ep_expert_map

        prepared = prepare_ep_expert_map(
            expert_map,
            local_num_experts=local_num_experts,
            global_num_experts=global_num_experts,
            device=device,
        )
        self._prepared_ep_expert_map = prepared
        return prepared

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare both local weights and the static map before graph capture."""

        super().process_weights_after_loading(layer)
        prepared = self._lookup_prepared_experts()
        if prepared is None:
            raise RuntimeError("B12X EP weights were not prepared after loading")
        expert_map = getattr(layer, "expert_map", None)
        if not isinstance(expert_map, torch.Tensor):
            raise RuntimeError("B12X EP requires a materialized expert_map")
        self._prepare_ep_expert_map(
            expert_map,
            global_num_experts=int(getattr(layer, "global_num_experts", 0)),
            local_num_experts=prepared.num_experts,
            device=prepared.w1_fp4.device,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del N, expert_tokens_meta
        prepared = self._lookup_prepared_experts()
        if prepared is None:
            raise RuntimeError(
                "B12X EP workspace planning requires prepared weights; "
                "process_weights_after_loading must run first"
            )
        if prepared.num_experts != int(local_num_experts):
            raise ValueError(
                "B12X EP local expert metadata does not match prepared weights: "
                f"metadata={int(local_num_experts)}, "
                f"prepared={prepared.num_experts}"
            )
        device = prepared.w1_fp4.device
        swiglu_limit, swiglu_alpha, swiglu_beta = self._b12x_swiglu_params(activation)
        plan = _plan_b12x_ep_moe_fp4_scratch(
            tokens=max(int(M), 1),
            topk=int(topk),
            global_num_experts=int(global_num_experts),
            device=device,
            experts=prepared,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
        workspace_dtype = getattr(self.moe_config, "in_dtype", torch.bfloat16)
        scratch_elements = max(
            1,
            _ceil_div(
                _b12x_scratch_nbytes(plan),
                _dtype_element_size(workspace_dtype),
            ),
        )
        return (0,), (scratch_elements,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del workspace13
        if expert_map is None:
            raise RuntimeError("B12X EP requires vLLM's global-to-local expert_map")
        if a1q_scale is not None or a2_scale is not None:
            raise RuntimeError("B12X W4A16 EP expects unquantized BF16 activations")
        if expert_tokens_meta is not None:
            raise RuntimeError("B12X EP does not accept batched-expert metadata")

        prepared = self._get_or_prepare_experts(
            w1=w1,
            w2=w2,
            activation=activation,
            params_dtype=hidden_states.dtype,
        )
        prepared_map = self._prepare_ep_expert_map(
            expert_map,
            global_num_experts=int(global_num_experts),
            local_num_experts=prepared.num_experts,
            device=hidden_states.device,
        )
        topk_ids = _normalize_b12x_moe_topk_ids(topk_ids)
        topk_weights = _normalize_b12x_moe_topk_weights(topk_weights)
        swiglu_limit, swiglu_alpha, swiglu_beta = self._b12x_swiglu_params(activation)
        plan = _plan_b12x_ep_moe_fp4_scratch(
            tokens=int(hidden_states.shape[0]),
            topk=int(topk_ids.shape[1]),
            global_num_experts=int(global_num_experts),
            device=hidden_states.device,
            experts=prepared,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
        scratch = _workspace2_as_b12x_scratch(workspace2, plan)
        _run_b12x_ep_moe_fp4(
            a=hidden_states,
            experts=prepared,
            output=output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=prepared_map,
            plan=plan,
            scratch=scratch,
        )

    def warmup_dynamic_signature(
        self,
        layer: torch.nn.Module,
    ) -> tuple[Any, ...] | None:
        meta = self._warmup_metadata(layer)
        if meta is None:
            return None
        expert_map = getattr(layer, "expert_map", None)
        global_num_experts = int(getattr(layer, "global_num_experts", 0))
        if not isinstance(expert_map, torch.Tensor):
            raise RuntimeError("B12X EP warmup requires a materialized expert_map")
        self._prepare_ep_expert_map(
            expert_map,
            global_num_experts=global_num_experts,
            local_num_experts=meta.num_experts,
            device=meta.device,
        )
        return (
            "replicated-input-ep-w4a16",
            meta.device.type,
            meta.device.index,
            meta.dtype,
            self._source_format(),
            self._w13_layout(),
            global_num_experts,
            meta.num_experts,
            meta.k,
            meta.n,
            meta.topk,
            meta.activation_name,
            meta.apply_router_weight_on_input,
            meta.swiglu_limit,
            meta.swiglu_alpha,
            meta.swiglu_beta,
        )

    @torch.inference_mode()
    def warmup_dynamic_launches(
        self,
        layer: torch.nn.Module,
        *,
        token_counts: Iterable[int],
    ) -> int:
        """Warm every serving token shape on the mapped W4A16 EP path."""

        meta = self._warmup_metadata(layer)
        if meta is None:
            return 0
        expert_map = getattr(layer, "expert_map", None)
        global_num_experts = int(getattr(layer, "global_num_experts", 0))
        if not isinstance(expert_map, torch.Tensor):
            raise RuntimeError("B12X EP warmup requires a materialized expert_map")

        prepared = self._get_or_prepare_experts(
            w1=meta.w1,
            w2=meta.w2,
            activation=meta.activation,
            params_dtype=meta.dtype,
        )
        prepared_map = self._prepare_ep_expert_map(
            expert_map,
            global_num_experts=global_num_experts,
            local_num_experts=prepared.num_experts,
            device=meta.device,
        )
        counts = tuple(sorted({max(int(count), 1) for count in token_counts}))
        for tokens in counts:
            plan = _plan_b12x_ep_moe_fp4_scratch(
                tokens=tokens,
                topk=meta.topk,
                global_num_experts=global_num_experts,
                device=meta.device,
                experts=prepared,
                apply_router_weight_on_input=meta.apply_router_weight_on_input,
                swiglu_limit=meta.swiglu_limit,
                swiglu_alpha=meta.swiglu_alpha,
                swiglu_beta=meta.swiglu_beta,
            )
            hidden_states = torch.zeros(
                (tokens, meta.k),
                dtype=meta.dtype,
                device=meta.device,
            )
            output = torch.empty_like(hidden_states)
            topk_ids = torch.arange(
                tokens * meta.topk,
                dtype=torch.int32,
                device=meta.device,
            ).reshape(tokens, meta.topk)
            topk_ids.remainder_(global_num_experts)
            topk_weights = torch.full(
                (tokens, meta.topk),
                1.0 / max(meta.topk, 1),
                dtype=torch.float32,
                device=meta.device,
            )
            scratch = torch.empty(
                (_b12x_scratch_nbytes(plan),),
                dtype=torch.uint8,
                device=meta.device,
            )
            _run_b12x_ep_moe_fp4(
                a=hidden_states,
                experts=prepared,
                output=output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                expert_map=prepared_map,
                plan=plan,
                scratch=scratch,
            )
        return len(counts)


__all__ = ["B12xEPExperts"]
