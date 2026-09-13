# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
import typing
from collections import defaultdict
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import (
    HCHeadOp,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.attention import (
    DeepseekV4Indexer,
    DeepseekV4MLAModules,
    DeepseekV4MultiHeadLatentAttentionWrapper,
    get_deepseek_v4_padded_num_q_heads,
)
from vllm.models.deepseek_v4.nvidia.ops import prepare_megamoe_inputs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _env_flag(*names: str) -> bool:
    return any(
        os.environ.get(name, "").lower() not in ("", "0", "false", "no")
        for name in names
    )


_B12X_MHC_TRACE = _env_flag("B12X_TRACE_MHC", "VLLM_TRACE_B12X_MHC")
_B12X_MHC_TRACE_LIMIT = int(
    os.environ.get(
        "B12X_TRACE_MHC_LIMIT",
        os.environ.get("VLLM_TRACE_B12X_MHC_LIMIT", "240"),
    )
)
_B12X_MHC_TRACE_COUNTS: dict[str, int] = {}


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )


_DSPARK_DEFER_TARGET_CAPTURE = _env_enabled(
    "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE"
)
_DSPARK_DEFER_TARGET_CAPTURE_EXACT = _env_enabled(
    "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT"
)
_DSPARK_TARGET_TIMING = _env_enabled("VLLM_DSPARK_TARGET_TIMING")
_DSPARK_TARGET_TIMING_LOG_EVERY = int(
    os.environ.get("VLLM_DSPARK_TARGET_TIMING_LOG_EVERY", "20")
)
_DSPARK_TARGET_TIMING_TOTALS: defaultdict[str, float] = defaultdict(float)
_DSPARK_TARGET_TIMING_COUNTS: defaultdict[str, int] = defaultdict(int)
_DSPARK_TARGET_TIMING_FORWARDS = 0


def _dspark_target_timing_active() -> bool:
    if not _DSPARK_TARGET_TIMING:
        return False
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is not None and is_compiling():
        return False
    return True


def _dspark_target_timing_start() -> float:
    if not _dspark_target_timing_active():
        return 0.0
    if current_platform.is_cuda():
        torch.cuda.synchronize()
    return time.perf_counter()


def _dspark_target_timing_record(stage: str, started: float) -> None:
    if not _dspark_target_timing_active() or started == 0.0:
        return
    if current_platform.is_cuda():
        torch.cuda.synchronize()
    _DSPARK_TARGET_TIMING_TOTALS[stage] += (time.perf_counter() - started) * 1000.0
    _DSPARK_TARGET_TIMING_COUNTS[stage] += 1


def _dspark_target_timing_finish(total_started: float, tokens: int, layers: int) -> None:
    global _DSPARK_TARGET_TIMING_FORWARDS

    if not _dspark_target_timing_active():
        return
    _dspark_target_timing_record("forward_total", total_started)
    _DSPARK_TARGET_TIMING_FORWARDS += 1
    every = max(1, _DSPARK_TARGET_TIMING_LOG_EVERY)
    if _DSPARK_TARGET_TIMING_FORWARDS % every != 0:
        return

    forwards = max(1, _DSPARK_TARGET_TIMING_FORWARDS)
    stages = (
        "embed_or_input",
        "layers_total",
        "layer_attn_mhc",
        "layer_attn",
        "layer_ffn_mhc",
        "layer_ffn",
        "dspark_capture",
        "final_hc_post",
        "mtp_hidden_copy",
        "hc_head_norm",
        "forward_total",
    )
    avg = {
        stage: _DSPARK_TARGET_TIMING_TOTALS.get(stage, 0.0) / forwards
        for stage in stages
    }
    logger.info(
        "DSpark target timing forwards=%d last_tokens=%d layers=%d avg_ms=%s",
        _DSPARK_TARGET_TIMING_FORWARDS,
        tokens,
        layers,
        ", ".join(f"{stage}:{avg[stage]:.3f}" for stage in stages),
    )


def _trace_b12x_mhc_call(
    op_name: str,
    layer_name: str,
    tokens: int,
    run: Callable[[], typing.Any],
) -> typing.Any:
    if not _B12X_MHC_TRACE:
        return run()

    call_count = _B12X_MHC_TRACE_COUNTS.get(op_name, 0) + 1
    _B12X_MHC_TRACE_COUNTS[op_name] = call_count
    if call_count > _B12X_MHC_TRACE_LIMIT:
        if call_count == _B12X_MHC_TRACE_LIMIT + 1:
            logger.info(
                "b12x mHC trace limit reached for %s at %d calls.",
                op_name,
                _B12X_MHC_TRACE_LIMIT,
            )
        return run()

    is_capturing = False
    is_current_stream_capturing = getattr(
        torch.cuda, "is_current_stream_capturing", None
    )
    if is_current_stream_capturing is not None:
        try:
            is_capturing = bool(is_current_stream_capturing())
        except Exception:
            is_capturing = False

    started = time.perf_counter()
    try:
        return run()
    finally:
        logger.info(
            "b12x mHC %s call=%d layer=%s tokens=%d capturing=%s elapsed=%.3f ms",
            op_name,
            call_count,
            layer_name,
            tokens,
            is_capturing,
            (time.perf_counter() - started) * 1000.0,
        )


def _use_b12x_mhc() -> bool:
    if not envs.VLLM_USE_B12X_MHC:
        return False
    if not current_platform.is_cuda():
        raise RuntimeError("VLLM_USE_B12X_MHC requires CUDA.")
    if not current_platform.is_device_capability_family(120):
        raise RuntimeError("VLLM_USE_B12X_MHC currently requires an SM120 GPU.")
    return True


def _b12x_mhc_max_tokens() -> int:
    raw = os.environ.get("B12X_MHC_MAX_TOKENS", "16")
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"B12X_MHC_MAX_TOKENS must be an integer, got {raw!r}"
        ) from exc


def _empty_b12x_plan_scratch(
    plan: object,
    device: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    specs = plan.shapes_and_dtypes()
    if not specs:
        raise ValueError("b12x scratch plan did not provide any scratch specs")
    buffers = tuple(
        torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs
    )
    if len(buffers) == 1:
        return buffers[0]
    return buffers


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def make_deepseek_v4_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    return [
        (
            "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_",
            f"experts.{expert_id}.{weight_name}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(num_experts)
        for shard_id, weight_name in [
            ("w1", "w1"),
            ("w2", "w2"),
            ("w3", "w3"),
        ]
    ]


class DeepseekV4MegaMoEExperts(nn.Module):
    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], object] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        self.w13_weight_scale.quant_method = "block"

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)
        self.w2_weight_scale.quant_method = "block"

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None

        # Register in the static forward context so the custom-op wrapper
        # can look up this module by name from within a torch.compile graph.
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _map_global_expert_id(self, expert_id: int) -> int:
        if expert_id < self.experts_start_idx or expert_id >= self.experts_end_idx:
            return -1
        return expert_id - self.experts_start_idx

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        local_expert_id = self._map_global_expert_id(expert_id)
        if local_expert_id == -1:
            return False if return_success else None

        expert_data = param.data[local_expert_id]
        if shard_id in ("w1", "w3"):
            if "w13_" not in weight_name:
                return False if return_success else None
            shard_offset = 0 if shard_id == "w1" else self.intermediate_size
            expert_data = expert_data.narrow(0, shard_offset, self.intermediate_size)
        elif shard_id == "w2":
            if "w2_" not in weight_name:
                return False if return_success else None
        else:
            raise ValueError(f"Unsupported expert shard id: {shard_id}")

        if expert_data.shape != loaded_weight.shape:
            raise ValueError(
                f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                f"{weight_name}: parameter shard {tuple(expert_data.shape)} "
                f"vs checkpoint {tuple(loaded_weight.shape)}"
            )
        expert_data.copy_(loaded_weight)
        return True if return_success else None

    @staticmethod
    def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
        return (sf.to(torch.int32) << 23).view(torch.float32)

    def _check_runtime_supported(self) -> None:
        if not torch.cuda.is_available():
            raise NotImplementedError("DeepSeek V4 MegaMoE requires CUDA.")
        device = self.w13_weight.device
        if device.type != "cuda":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE expert weights must be loaded on CUDA."
            )
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return

        self._check_runtime_supported()
        import vllm.third_party.deep_gemm as deep_gemm

        w13_scale = deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_uint8_to_float(self.w13_weight_scale.data).contiguous(),
            2 * self.intermediate_size,
            self.hidden_size,
            (1, 32),
            self.num_local_experts,
        )
        w2_scale = deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_uint8_to_float(self.w2_weight_scale.data).contiguous(),
            self.hidden_size,
            self.intermediate_size,
            (1, 32),
            self.num_local_experts,
        )
        self._transformed_l1_weights, self._transformed_l2_weights = (
            deep_gemm.transform_weights_for_mega_moe(
                (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
                (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
            )
        )
        # Drop the original loader-side parameters: the MegaMoE kernels only
        # consume the transformed views above. transform_weights_for_mega_moe
        # allocates a fresh tensor for the L1 weight (see _interleave_l1_weights)
        # and fresh SF tensors for L1/L2; the L2 weight is the only tensor that
        # aliases the original storage, and _transformed_l2_weights still holds
        # it, so the storage stays live after we drop the Parameter.
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None

    def get_symm_buffer(self):
        import vllm.third_party.deep_gemm as deep_gemm

        group = get_ep_group().device_group
        device = torch.accelerator.current_device_index()
        key = (
            id(group),
            device,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but the symmetric buffer was sized for {self.max_num_tokens}."
            )
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        torch.ops.vllm.deepseek_v4_mega_moe_experts(
            hidden_states,
            topk_weights,
            topk_ids,
            y,
            self.prefix,
            activation_clamp,
            fast_math,
        )
        return y

    def _run_mega_moe(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        y: torch.Tensor,
        activation_clamp: float | None,
        fast_math: bool,
    ) -> None:
        import vllm.third_party.deep_gemm as deep_gemm

        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        prepare_megamoe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
        )

        # This method must have been already called during the weight loading phase.
        # We call it again here to cover the dummy weight loading case.
        self.finalize_weights()

        assert self._transformed_l1_weights is not None
        assert self._transformed_l2_weights is not None
        deep_gemm.fp8_fp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )


DeepseekV4MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def _deepseek_v4_mega_moe_experts_op(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
    activation_clamp: float | None,
    fast_math: bool,
) -> None:
    self = get_forward_context().no_compile_layers[layer_name]
    self._run_mega_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        out,
        activation_clamp,
        fast_math,
    )


def _deepseek_v4_mega_moe_experts_op_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
    activation_clamp: float | None,
    fast_math: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_mega_moe_experts",
    op_func=_deepseek_v4_mega_moe_experts_op,
    mutates_args=["out"],
    fake_impl=_deepseek_v4_mega_moe_experts_op_fake,
)


def _deepseek_v4_b12x_mhc_post_pre_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    self = get_forward_context().no_compile_layers[layer_name]
    return _trace_b12x_mhc_call(
        "post_pre",
        layer_name,
        int(residual.shape[0]),
        lambda: self._run_b12x_mhc_post_pre(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            norm_weight,
            norm_eps,
        ),
    )


def _deepseek_v4_b12x_mhc_post_pre_op_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del x, post, comb, hc_fn, hc_scale, hc_base, norm_weight, norm_eps, layer_name
    tokens, hc_mult, hidden_size = residual.shape
    residual_out = torch.empty_like(residual)
    post_out = torch.empty(
        (tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    comb_out = torch.empty(
        (tokens, hc_mult, hc_mult), dtype=torch.float32, device=residual.device
    )
    y_out = torch.empty(
        (tokens, hidden_size), dtype=residual.dtype, device=residual.device
    )
    return residual_out, post_out, comb_out, y_out


direct_register_custom_op(
    op_name="deepseek_v4_b12x_mhc_post_pre",
    op_func=_deepseek_v4_b12x_mhc_post_pre_op,
    fake_impl=_deepseek_v4_b12x_mhc_post_pre_op_fake,
)


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.use_mega_moe and self.scoring_func != "sqrtsoftplus":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently supports sqrtsoftplus routing only."
            )
        if self.use_mega_moe and getattr(config, "expert_dtype", "fp4") != "fp4":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE only supports fp4 experts; got expert_dtype="
                f"{config.expert_dtype!r}. Drop --kernel-config moe_backend="
                "deep_gemm_mega_moe for this checkpoint."
            )

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32
        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=self.use_mega_moe,
                prefix=f"{prefix}.shared_experts",
            )

        if self.use_mega_moe:
            self._init_mega_moe_experts(vllm_config, config, prefix)
        else:
            self._init_fused_moe_experts(config, quant_config, prefix)

    def _init_mega_moe_experts(
        self,
        vllm_config: VllmConfig,
        config,
        prefix: str,
    ) -> None:
        self.ep_group = get_ep_group()
        self.ep_size = self.ep_group.world_size
        self.ep_rank = self.ep_group.rank_in_group
        assert config.n_routed_experts % self.ep_size == 0

        self.n_local_experts = config.n_routed_experts // self.ep_size
        self.experts_start_idx = self.ep_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.experts = DeepseekV4MegaMoEExperts(
            vllm_config,
            num_experts=config.n_routed_experts,
            num_local_experts=self.n_local_experts,
            experts_start_idx=self.experts_start_idx,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
        )

    def _init_fused_moe_experts(
        self,
        config,
        quant_config,
        prefix: str,
    ) -> None:
        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0

        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        if not self.use_mega_moe:
            return self._forward_fused_moe(hidden_states, input_ids)

        org_shape = hidden_states.shape
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias.data
            if self.gate.e_score_correction_bias is not None
            else None,
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        activation_clamp = (
            float(self.swiglu_limit) if self.swiglu_limit is not None else None
        )
        final_hidden_states = self.experts(
            hidden_states,
            topk_weights,
            topk_ids,
            activation_clamp=activation_clamp,
        )

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            final_hidden_states += shared_output

        return final_hidden_states.view(org_shape)

    def _forward_fused_moe(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        org_shape = hidden_states.shape
        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
            )

        return final_hidden_states.view(org_shape)

    def finalize_mega_moe_weights(self) -> None:
        if self.use_mega_moe:
            self.experts.finalize_weights()


class DeepseekV4Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        layer_id = extract_layer_index(prefix)

        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.n_heads % tp_size == 0

        self.n_local_heads = self.n_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        # NOTE(zyongye) Compress ratio can't be 0
        # we do this for because MTP layer is not included
        # in the compress ratio list
        if layer_id < config.num_hidden_layers:
            self.compress_ratio = max(1, config.compress_ratios[layer_id])
        else:
            self.compress_ratio = 1
        self.eps = config.rms_norm_eps
        self.max_position_embeddings = config.max_position_embeddings

        # Must match DeepseekV4MLAAttention.padded_heads; padded entries stay
        # -inf so they contribute no sink effect.
        padded_heads = get_deepseek_v4_padded_num_q_heads(self.n_local_heads)
        self.attn_sink = nn.Parameter(
            torch.full((padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )

        self.fused_wqa_wkv = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,  # fused ReplicatedLinear
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )

        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        if envs.VLLM_USE_B12X_WO_PROJECTION:
            if not hasattr(self.wo_a, "weight_scale_inv"):
                raise RuntimeError(
                    "VLLM_USE_B12X_WO_PROJECTION requires FP8 wo_a.weight_scale_inv"
                )
            # Preserve checkpoint UE8M0 scales for the fused b12x WO kernel.
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        if envs.VLLM_USE_B12X_WO_PROJECTION:
            if not hasattr(self.wo_b, "weight_scale_inv"):
                raise RuntimeError(
                    "VLLM_USE_B12X_WO_PROJECTION requires FP8 wo_b.weight_scale_inv"
                )
            self.wo_a.b12x_skip_generic_block_fp8_linear = True
            self.wo_b.b12x_skip_generic_block_fp8_linear = True

        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = config.quantization_config["scale_fmt"]

        self.rope_parameters = config.rope_scaling

        # Initialize rotary embedding BEFORE DeepseekV4MLAModules (which needs it)
        rope_parameters = config.rope_parameters
        rope_parameters["rope_theta"] = (
            config.compress_rope_theta if self.compress_ratio > 1 else config.rope_theta
        )
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        rope_parameters["mscale"] = 0  # Disable mscale
        rope_parameters["mscale_all_dim"] = 0  # Disable mscale
        rope_parameters["is_deepseek_v4"] = True
        rope_parameters["rope_dim"] = self.rope_head_dim
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=False,
        )

        self.indexer = None
        if self.compress_ratio == 4:
            # Only C4A uses sparse attention and hence has indexer.
            # aux_stream_list[0] runs indexer.forward() in the wrapper; [2] is
            # free here (outer GEMMs joined) for the inner overlap of
            # wq_b+fused_indexer_q_rope_quant vs compressor.
            indexer_aux_stream = (
                aux_stream_list[2] if aux_stream_list is not None else None
            )
            self.indexer = DeepseekV4Indexer(
                vllm_config,
                config=config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=vllm_config.cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=self.compress_ratio,
                prefix=f"{prefix}.indexer",
                aux_stream=indexer_aux_stream,
            )

        mla_modules = DeepseekV4MLAModules(
            vllm_config=vllm_config,
            fused_wqa_wkv=self.fused_wqa_wkv,
            q_norm=self.q_norm,
            wq_b=self.wq_b,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            rotary_emb=self.rotary_emb,
            indexer=self.indexer,
            indexer_rotary_emb=self.rotary_emb,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.mla_attn = DeepseekV4MultiHeadLatentAttentionWrapper(
            hidden_size=self.hidden_size,
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.softmax_scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            v_head_dim=self.head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.head_dim,
            o_lora_rank=self.o_lora_rank,
            mla_modules=mla_modules,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ):
        return self.mla_attn(positions, hidden_states, llama_4_scaling)

    def setup_b12x_wo_projection(self) -> None:
        self.mla_attn.setup_b12x_wo_projection()


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.layer_name = prefix
        self._dspark_prev_capture_buffer: torch.Tensor | None = None
        self._dspark_prev_capture_start = 0
        self._dspark_prev_capture_end = 0
        self._dspark_prev_capture_exact = False
        self._use_b12x_mhc = _use_b12x_mhc()
        self._b12x_mhc_max_tokens = _b12x_mhc_max_tokens() if self._use_b12x_mhc else 0
        if self._use_b12x_mhc:
            if not prefix:
                raise RuntimeError("DeepSeek V4 b12x mHC decoder layer needs a prefix")
            compilation_config = vllm_config.compilation_config
            if prefix in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {prefix}")
            compilation_config.static_forward_context[prefix] = self

            if self._b12x_mhc_max_tokens <= 0:
                logger.info_once("DeepSeek V4 b12x mHC enabled for all token counts.")
            else:
                logger.info_once(
                    "DeepSeek V4 b12x mHC enabled for token counts <= %d; "
                    "using TileLang mHC above that.",
                    self._b12x_mhc_max_tokens,
                )

        # Registers torch.ops.vllm.mhc_* and provides the fallback path for
        # mixed/prefill capture sizes when b12x mHC is decode-limited.
        from vllm.model_executor.layers.mhc import (
            MHCFusedPostPreOp,
            MHCPostOp,
            MHCPreOp,
        )

        self.hidden_size = config.hidden_size

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = DeepseekV4Attention(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
        )
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        if self._use_b12x_mhc:
            from b12x.integration.residual import (
                MHC_DEFAULT_BLOCK_K,
                MHC_DEFAULT_SPLIT_K,
                MHC_MULT,
            )

            if self.hc_mult != MHC_MULT:
                raise NotImplementedError(
                    f"DeepSeek V4 b12x mHC requires hc_mult={MHC_MULT}, "
                    f"got {self.hc_mult}."
                )
            if self.hidden_size != 4096:
                raise NotImplementedError(
                    "DeepSeek V4 b12x mHC currently requires hidden_size=4096, "
                    f"got {self.hidden_size}."
                )
            self._b12x_mhc_block_k = int(MHC_DEFAULT_BLOCK_K)
            total_k = self.hc_mult * self.hidden_size
            if total_k % self._b12x_mhc_block_k != 0:
                raise ValueError(
                    "DeepSeek V4 b12x mHC requires hc_mult * hidden_size to "
                    f"be divisible by block_k={self._b12x_mhc_block_k}, got {total_k}."
                )
            self._b12x_mhc_split_k = total_k // self._b12x_mhc_block_k
            if self._b12x_mhc_split_k != MHC_DEFAULT_SPLIT_K:
                raise NotImplementedError(
                    "DeepSeek V4 b12x mHC currently requires "
                    f"split_k={MHC_DEFAULT_SPLIT_K}, got {self._b12x_mhc_split_k}."
                )
        else:
            self._b12x_mhc_block_k = 0
            self._b12x_mhc_split_k = 0
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()
        self.mhc_fused_post_pre = MHCFusedPostPreOp()

    def set_dspark_previous_layer_capture(
        self,
        buffer: torch.Tensor,
        start: int,
        end: int,
        *,
        exact: bool = False,
    ) -> None:
        self._dspark_prev_capture_buffer = buffer
        self._dspark_prev_capture_start = int(start)
        self._dspark_prev_capture_end = int(end)
        self._dspark_prev_capture_exact = bool(exact)

    def _maybe_capture_dspark_previous_layer_exact(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> None:
        buffer = self._dspark_prev_capture_buffer
        if buffer is None or not self._dspark_prev_capture_exact:
            return
        start = self._dspark_prev_capture_start
        end = self._dspark_prev_capture_end
        from vllm.models.deepseek_v4.nvidia.dspark_kernels import (
            dspark_hc_post_mean,
        )

        dspark_hc_post_mean(
            x,
            residual,
            post,
            comb,
            buffer[: x.shape[0], start:end],
        )

    def _maybe_capture_dspark_previous_layer(
        self,
        residual: torch.Tensor,
    ) -> None:
        buffer = self._dspark_prev_capture_buffer
        if buffer is None:
            return
        if self._dspark_prev_capture_exact:
            return
        start = self._dspark_prev_capture_start
        end = self._dspark_prev_capture_end
        buffer[: residual.shape[0], start:end].copy_(residual.mean(dim=1))

    def _should_run_b12x_mhc(self, tokens: int) -> bool:
        if not self._use_b12x_mhc:
            return False
        max_tokens = self._b12x_mhc_max_tokens
        return max_tokens <= 0 or int(tokens) <= max_tokens

    def _get_b12x_mhc_binding(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> object:
        from b12x.integration.residual import B12XMHCScratchCaps, plan_mhc_scratch

        tokens = int(x.shape[0])
        plan = plan_mhc_scratch(
            B12XMHCScratchCaps(
                device=x.device,
                dtype=x.dtype,
                max_tokens=max(1, tokens),
                hidden_size=self.hidden_size,
                split_k=self._b12x_mhc_split_k,
            )
        )
        scratch = _empty_b12x_plan_scratch(plan, x.device)
        return plan.bind(
            scratch=scratch,
            tokens=tokens,
            y=y,
            post=post,
            comb=comb,
            out=out,
        )

    def _run_b12x_mhc_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from b12x.integration.residual import b12x_mhc_post_pre

        tokens, hc_mult, hidden_size = residual.shape
        residual_out = torch.empty_like(residual)
        y_out = torch.empty(
            (tokens, hidden_size), dtype=residual.dtype, device=residual.device
        )
        post_out = torch.empty(
            (tokens, hc_mult), dtype=torch.float32, device=residual.device
        )
        comb_out = torch.empty(
            (tokens, hc_mult, hc_mult), dtype=torch.float32, device=residual.device
        )
        binding = self._get_b12x_mhc_binding(
            residual,
            y=y_out,
            post=post_out,
            comb=comb_out,
            out=residual_out,
        )
        return b12x_mhc_post_pre(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            rms_eps=self.rms_norm_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            binding=binding,
            block_k=self._b12x_mhc_block_k,
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ):
        assert self.mhc_pre is not None
        post_mix, res_mix, layer_input = self.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        assert self.mhc_post is not None
        return self.mhc_post(x, residual, post, comb)

    def hc_post_pre(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._should_run_b12x_mhc(int(residual.shape[0])):
            if not is_forward_context_available():
                return self._run_b12x_mhc_post_pre(
                    x,
                    residual,
                    post,
                    comb,
                    hc_fn,
                    hc_scale,
                    hc_base,
                    norm_weight,
                    norm_eps,
                )
            return torch.ops.vllm.deepseek_v4_b12x_mhc_post_pre(
                x,
                residual,
                post,
                comb,
                hc_fn,
                hc_scale,
                hc_base,
                norm_weight,
                norm_eps,
                self.layer_name,
            )

        assert self.mhc_fused_post_pre is not None
        return self.mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            hc_fn,
            hc_scale,
            hc_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

    def _forward_cuda(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._should_run_b12x_mhc(int(x.shape[0])):
            attn_norm_weight = self.attn_norm.weight.data
            attn_norm_eps = self.attn_norm.variance_epsilon
            stage_start = _dspark_target_timing_start()
            if residual is None:
                residual = x
                x, post_mix, res_mix = self.hc_pre(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                )
            else:
                assert post_mix is not None
                assert res_mix is not None
                self._maybe_capture_dspark_previous_layer_exact(
                    x, residual, post_mix, res_mix
                )
                residual, post_mix, res_mix, x = self.hc_post_pre(
                    x,
                    residual,
                    post_mix,
                    res_mix,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    norm_weight=attn_norm_weight,
                    norm_eps=attn_norm_eps,
                )
                self._maybe_capture_dspark_previous_layer(residual)
            _dspark_target_timing_record("layer_attn_mhc", stage_start)

            stage_start = _dspark_target_timing_start()
            x = self.attn(positions, x, None)
            _dspark_target_timing_record("layer_attn", stage_start)

            ffn_norm_weight = self.ffn_norm.weight.data
            ffn_norm_eps = self.ffn_norm.variance_epsilon
            stage_start = _dspark_target_timing_start()
            residual, post_mix, res_mix, x = self.hc_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                norm_weight=ffn_norm_weight,
                norm_eps=ffn_norm_eps,
            )
            _dspark_target_timing_record("layer_ffn_mhc", stage_start)
            stage_start = _dspark_target_timing_start()
            x = self.ffn(x, input_ids)
            _dspark_target_timing_record("layer_ffn", stage_start)
            return x, residual, post_mix, res_mix

        assert self.mhc_fused_post_pre is not None
        attn_norm_weight = self.attn_norm.weight.data
        attn_norm_eps = self.attn_norm.variance_epsilon
        stage_start = _dspark_target_timing_start()
        if residual is None:
            # Run standalone hc_pre on first layer
            residual = x
            x, post_mix, res_mix = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
        else:
            assert post_mix is not None
            assert res_mix is not None
            self._maybe_capture_dspark_previous_layer_exact(
                x, residual, post_mix, res_mix
            )
            residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
            )
            self._maybe_capture_dspark_previous_layer(residual)
        _dspark_target_timing_record("layer_attn_mhc", stage_start)

        # attn_norm is fused into hc_pre / mhc_fused_post_pre above.
        stage_start = _dspark_target_timing_start()
        x = self.attn(positions, x, None)
        _dspark_target_timing_record("layer_attn", stage_start)

        ffn_norm_weight = self.ffn_norm.weight.data
        ffn_norm_eps = self.ffn_norm.variance_epsilon
        stage_start = _dspark_target_timing_start()
        residual, post_mix, res_mix, x = self.mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            n_splits=1,
            tile_n=1,
            norm_weight=ffn_norm_weight,
            norm_eps=ffn_norm_eps,
        )
        _dspark_target_timing_record("layer_ffn_mhc", stage_start)

        stage_start = _dspark_target_timing_start()
        x = self.ffn(x, input_ids)
        _dspark_target_timing_record("layer_ffn", stage_start)
        return x, residual, post_mix, res_mix

    def _forward_native(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(positions, x, None)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x, None, None, None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        if current_platform.is_rocm() or current_platform.is_xpu():
            return self._forward_native(
                x, positions, input_ids, post_mix, res_mix, residual
            )

        return self._forward_cuda(x, positions, input_ids, post_mix, res_mix, residual)


@support_torch_compile
class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4MultiHeadLatentAttentionWrapper.attn_gemm_parallel_execute
        # (compressor kv_score, indexer.weights_proj, indexer.compressor
        # kv_score). fused_wqa_wkv stays on the default stream.
        # Disable them on ROCm / XPU because of hang issues / no overlap.
        aux_stream_list = (
            None
            if current_platform.is_rocm() or current_platform.is_xpu()
            else [torch.cuda.Stream() for _ in range(3)]
        )

        self.device = current_platform.device_type
        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_dim,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(
                self.hc_mult,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_op = HCHeadOp()
        # Pre-hc_head residual stream buffer for the MTP draft. Stable
        # address (outside the cudagraph pool) so the copy_ in forward()
        # refreshes it correctly across captured shapes.
        # refreshes it correctly across captured shapes. Only allocated on
        # the last PP rank — that's where MTP target hidden states are
        # produced.
        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        else:
            self._mtp_hidden_buffer = None

        self._dspark_target_layer_ids = tuple(
            int(layer_id) for layer_id in getattr(config, "dspark_target_layer_ids", ())
        )
        if get_pp_group().is_last_rank and self._dspark_target_layer_ids:
            self._dspark_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                len(self._dspark_target_layer_ids) * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
            self._dspark_layer_to_buffer_index = {
                layer_id: idx
                for idx, layer_id in enumerate(self._dspark_target_layer_ids)
            }
            self._dspark_deferred_capture_layer_ids = set()
            if _DSPARK_DEFER_TARGET_CAPTURE:
                self._setup_dspark_deferred_target_capture(config.hidden_size)
        else:
            self._dspark_hidden_buffer = None
            self._dspark_layer_to_buffer_index = {}
            self._dspark_deferred_capture_layer_ids = set()

    def _setup_dspark_deferred_target_capture(self, hidden_size: int) -> None:
        assert self._dspark_hidden_buffer is not None
        layers_by_id = {
            extract_layer_index(layer.layer_name): layer
            for layer in self.layers
            if hasattr(layer, "layer_name")
        }
        for target_layer_id, buffer_idx in self._dspark_layer_to_buffer_index.items():
            if target_layer_id < 0:
                continue
            next_layer = layers_by_id.get(target_layer_id + 1)
            if next_layer is None or not hasattr(
                next_layer, "set_dspark_previous_layer_capture"
            ):
                continue
            start = buffer_idx * hidden_size
            end = start + hidden_size
            next_layer.set_dspark_previous_layer_capture(
                self._dspark_hidden_buffer,
                start,
                end,
                exact=_DSPARK_DEFER_TARGET_CAPTURE_EXACT,
            )
            self._dspark_deferred_capture_layer_ids.add(target_layer_id)
        if self._dspark_deferred_capture_layer_ids:
            mode = "exact" if _DSPARK_DEFER_TARGET_CAPTURE_EXACT else "fused"
            logger.info_once(
                "DeepSeek V4 DSpark deferred target-layer capture enabled for "
                "layers %s with %s capture.",
                tuple(sorted(self._dspark_deferred_capture_layer_ids)),
                mode,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until hc_head() collapses it.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        total_start = _dspark_target_timing_start()
        stage_start = _dspark_target_timing_start()
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        _dspark_target_timing_record("embed_or_input", stage_start)

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        residual, post_mix, res_mix = None, None, None
        layer = None
        num_layers = 0
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            num_layers += 1
            layer_start = _dspark_target_timing_start()
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
            _dspark_target_timing_record("layers_total", layer_start)
            layer_idx = getattr(layer, "layer_name", "")
            layer_id = extract_layer_index(layer_idx) if layer_idx else None
            if (
                layer_id in self._dspark_layer_to_buffer_index
                and layer_id not in self._dspark_deferred_capture_layer_ids
            ):
                stage_start = _dspark_target_timing_start()
                if current_platform.is_cuda() and residual is not None:
                    hidden_states = layer.hc_post(
                        hidden_states, residual, post_mix, res_mix
                    )
                    residual, post_mix, res_mix = None, None, None
                buffer_idx = self._dspark_layer_to_buffer_index[layer_id]
                if self._dspark_hidden_buffer is not None:
                    start = buffer_idx * self.config.hidden_size
                    end = start + self.config.hidden_size
                    self._dspark_hidden_buffer[
                        : hidden_states.shape[0], start:end
                    ].copy_(hidden_states.mean(dim=1))
                _dspark_target_timing_record("dspark_capture", stage_start)
        if layer is not None and current_platform.is_cuda() and residual is not None:
            stage_start = _dspark_target_timing_start()
            hidden_states = layer.hc_post(hidden_states, residual, post_mix, res_mix)
            _dspark_target_timing_record("final_hc_post", stage_start)

        if not get_pp_group().is_last_rank:
            _dspark_target_timing_finish(
                total_start, int(hidden_states.shape[0]), num_layers
            )
            return IntermediateTensors({"hidden_states": hidden_states})

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        num_tokens = hidden_states.shape[0]
        stage_start = _dspark_target_timing_start()
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
        _dspark_target_timing_record("mtp_hidden_copy", stage_start)

        stage_start = _dspark_target_timing_start()
        hidden_states = self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        _dspark_target_timing_record("hc_head_norm", stage_start)
        _dspark_target_timing_finish(total_start, int(num_tokens), num_layers)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        first_layer = next(iter(islice(self.layers, self.start_layer, self.end_layer)))
        if first_layer.ffn.use_mega_moe:
            return make_deepseek_v4_expert_params_mapping(self.config.n_routed_experts)
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.ffn.finalize_mega_moe_weights()

    def setup_b12x_wo_projection(self) -> None:
        if not envs.VLLM_USE_B12X_WO_PROJECTION:
            return
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            layer.attn.setup_b12x_wo_projection()


def _make_deepseek_v4_weights_mapper(expert_dtype: str) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). FP8 linear and
        # shared experts use Fp8LinearMethod's block scales, which
        # register as ``weight_scale_inv``.
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".attn.compressor.": ".attn.mla_attn.compressor.",
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


class DeepseekV4ForCausalLM(nn.Module, SupportsPP):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        if expert_dtype != "fp4":
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def get_dspark_target_hidden_states(self) -> torch.Tensor | None:
        """Concatenated DSpark target-layer features for the draft model.

        Shape is (max_num_batched_tokens,
        len(dspark_target_layer_ids) * hidden_size). Populated by forward().
        """
        return getattr(self.model, "_dspark_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.model.finalize_mega_moe_weights()
        self.model.setup_b12x_wo_projection()
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
