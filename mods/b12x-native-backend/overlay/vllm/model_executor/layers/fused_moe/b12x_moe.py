# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X modular fused-MoE backend for FP4 weights."""

import os
import re
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)

_moe_repeat_check_reports = 0
_activation_amax_save_step = 0


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning_once(
            "Ignoring invalid integer environment value %s=%r", name, value
        )
        return default


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value
    return None


def _moe_repeat_check_enabled() -> bool:
    return _env_flag("B12X_MOE_REPEAT_CHECK") or _env_flag("VLLM_B12X_MOE_REPEAT_CHECK")


def _moe_repeat_check_after_engine_start() -> bool:
    return _env_flag("B12X_MOE_REPEAT_CHECK_AFTER_ENGINE_START") or _env_flag(
        "VLLM_B12X_MOE_REPEAT_CHECK_AFTER_ENGINE_START"
    )


def _moe_zero_scratch_enabled() -> bool:
    return _env_flag("B12X_MOE_ZERO_SCRATCH") or _env_flag("VLLM_B12X_MOE_ZERO_SCRATCH")


def _moe_force_a8_enabled() -> bool:
    return _env_flag("B12X_MOE_FORCE_A8") or _env_flag("B12X_FORCE_MOE_A8")


def _moe_force_a16_enabled() -> bool:
    return _env_flag("B12X_MOE_FORCE_A16")


def _moe_activation_amax_enabled() -> bool:
    return _env_flag("VLLM_B12X_MOE_ACTIVATION_AMAX")


def _moe_activation_amax_save_every() -> int:
    value = _env_first(
        "VLLM_B12X_MOE_ACTIVATION_AMAX_SAVE_EVERY",
    )
    if value is None:
        return 0
    try:
        return max(int(value), 0)
    except ValueError:
        logger.warning_once(
            "Ignoring invalid B12X MoE activation-amax save interval %r",
            value,
        )
        return 0


def _moe_activation_amax_file() -> str | None:
    return _env_first(
        "VLLM_B12X_MOE_ACTIVATION_AMAX_FILE",
    )


def _is_w4a8_quant_mode(quant_mode: str) -> bool:
    return quant_mode in ("w4a8_mx", "w4a8_nvfp4")


def _supports_swiglu_limit(quant_mode: str) -> bool:
    return quant_mode == "w4a16" or _is_w4a8_quant_mode(quant_mode)


def _moe_core_plan(plan: Any) -> Any:
    return getattr(plan, "_core_workspace_plan", plan)


def _dtype_element_size(dtype: torch.dtype) -> int:
    return dtype.itemsize


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


class _B12XMoeActivationAmaxState:
    def __init__(
        self,
        *,
        model_tag: str,
        device: torch.device,
        num_experts: int,
    ) -> None:
        self.model_tag = model_tag
        self.device = device
        self.num_experts = int(num_experts)
        self.tensor = torch.zeros(
            (0, self.num_experts, 2),
            dtype=torch.float32,
            device=device,
        )
        self.owner_slots: dict[int, int] = {}
        self.layers: dict[int, dict[str, Any]] = {}
        self.used = False

    def _next_free_slot(self) -> int:
        slot = 0
        while slot in self.layers:
            slot += 1
        return slot

    def _grow_to(self, rows: int) -> None:
        rows = int(rows)
        if rows <= int(self.tensor.shape[0]):
            return
        if self.used or _is_current_stream_capturing():
            raise RuntimeError(
                "B12X MoE activation-amax tensor would reallocate after use. "
                "All calibrated W4A16 MoE layers must be registered before "
                "CUDA graph capture or first launch."
            )
        grown = torch.zeros(
            (rows, self.num_experts, 2),
            dtype=torch.float32,
            device=self.device,
        )
        if self.tensor.numel():
            grown[: self.tensor.shape[0]].copy_(self.tensor)
        self.tensor = grown

    def register(
        self,
        *,
        owner: object,
        prefix: str,
        external_layer_idx: int | None,
        slot_hint: int | None,
    ) -> int:
        owner_id = id(owner)
        existing = self.owner_slots.get(owner_id)
        if existing is not None:
            return existing

        slot = int(slot_hint) if slot_hint is not None else self._next_free_slot()
        if slot < 0:
            raise ValueError(
                f"B12X MoE activation-amax slot must be non-negative: {slot}"
            )

        occupant = self.layers.get(slot)
        if occupant is not None and occupant.get("owner_id") != owner_id:
            raise RuntimeError(
                "B12X MoE activation-amax slot collision for "
                f"{self.model_tag!r} slot {slot}: {prefix!r} conflicts with "
                f"{occupant.get('prefix')!r}."
            )

        self._grow_to(slot + 1)
        self.owner_slots[owner_id] = slot
        self.layers[slot] = {
            "owner_id": owner_id,
            "slot": slot,
            "prefix": prefix,
            "external_layer_idx": external_layer_idx,
        }
        return slot

    def mark_used(self) -> None:
        self.used = True

    def payload(self, step: int) -> dict[str, Any]:
        layers = [
            {
                key: value
                for key, value in self.layers.get(slot, {"slot": slot}).items()
                if key != "owner_id"
            }
            for slot in range(int(self.tensor.shape[0]))
        ]
        return {
            "step": int(step),
            "model": self.model_tag,
            "device": str(self.device),
            "num_experts": self.num_experts,
            "columns": ("fc1", "fc2"),
            "layers": layers,
            "activation_amax": self.tensor.detach().cpu(),
        }


_activation_amax_states: dict[tuple[str, str, int], _B12XMoeActivationAmaxState] = {}


def _parse_layer_index(prefix: str) -> int | None:
    match = _LAYER_INDEX_RE.search(prefix)
    if match is None:
        return None
    return int(match.group(1))


def _current_config_num_hidden_layers() -> int | None:
    try:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
    except Exception:
        return None

    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        return None
    return int(num_hidden_layers)


def _activation_amax_model_tag(
    prefix: str,
    external_layer_idx: int | None,
    base_num_layers: int | None,
) -> str:
    prefix_lower = prefix.lower()
    if (
        "mtp" in prefix_lower
        or "draft_model" in prefix_lower
        or (
            base_num_layers is not None
            and external_layer_idx is not None
            and external_layer_idx >= base_num_layers
        )
    ):
        return "mtp"
    return "main"


def _activation_amax_slot_hint(
    *,
    model_tag: str,
    external_layer_idx: int | None,
    base_num_layers: int | None,
) -> int | None:
    if external_layer_idx is None:
        return None
    if model_tag == "mtp" and base_num_layers is not None:
        return max(int(external_layer_idx) - int(base_num_layers), 0)
    return int(external_layer_idx)


def _activation_amax_state_key(
    *,
    model_tag: str,
    device: torch.device,
    num_experts: int,
) -> tuple[str, str, int]:
    return (model_tag, str(device), int(num_experts))


def _get_activation_amax_state(
    *,
    model_tag: str,
    device: torch.device,
    num_experts: int,
) -> _B12XMoeActivationAmaxState:
    key = _activation_amax_state_key(
        model_tag=model_tag,
        device=device,
        num_experts=num_experts,
    )
    state = _activation_amax_states.get(key)
    if state is None:
        state = _B12XMoeActivationAmaxState(
            model_tag=model_tag,
            device=device,
            num_experts=num_experts,
        )
        _activation_amax_states[key] = state
    return state


def _distributed_rank() -> int:
    dist = getattr(torch, "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    for env_name in ("RANK", "LOCAL_RANK"):
        value = os.getenv(env_name)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return 0


def _activation_amax_output_path(
    base_path: str,
    state: _B12XMoeActivationAmaxState,
) -> Path:
    base = Path(base_path).expanduser()
    rank = _distributed_rank()
    device = str(state.device).replace(":", "")
    suffix = f"{state.model_tag}.rank{rank}.{device}.e{state.num_experts}"
    if base.suffix:
        return base.with_name(f"{base.stem}.{suffix}{base.suffix}")
    return base / f"b12x_moe_activation_amax.{suffix}.pt"


def maybe_save_b12x_moe_activation_amax() -> None:
    """Persist vLLM-owned W4A16 MoE activation calibration state."""
    global _activation_amax_save_step

    if not _activation_amax_states or _is_current_stream_capturing():
        return

    every = _moe_activation_amax_save_every()
    base_path = _moe_activation_amax_file()
    if every <= 0 or base_path is None:
        return

    _activation_amax_save_step += 1
    if _activation_amax_save_step % every != 0:
        return

    for state in _activation_amax_states.values():
        path = _activation_amax_output_path(base_path, state)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        torch.save(state.payload(_activation_amax_save_step), tmp_path)
        os.replace(tmp_path, path)


def _reset_b12x_moe_activation_amax_for_tests() -> None:
    global _activation_amax_save_step

    _activation_amax_save_step = 0
    _activation_amax_states.clear()


def _plan_b12x_moe_fp4_scratch(
    *,
    tokens: int,
    topk: int,
    device: torch.device,
    quant_mode: str,
    experts: Any,
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    collect_activation_amax: bool = False,
):
    from b12x.moe.fused_moe import Caps as TPMoEScratchCaps
    from b12x.moe.fused_moe import plan as plan_tp_moe_scratch

    return plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=max(int(tokens), 1),
            num_topk=int(topk),
            device=device,
            weight_plan=experts.plan,
            core_token_counts=(max(int(tokens), 1),),
            route_num_experts=0,
            quant_mode=quant_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            collect_activation_amax=bool(collect_activation_amax),
            frozen=True,
        )
    )


def _plan_b12x_moe_execution(
    *,
    tokens: int,
    topk: int,
    device: torch.device,
    quant_mode: str,
    experts: Any,
    apply_router_weight_on_input: bool = False,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
):
    from b12x.moe.fused_moe import plan_execution as plan_tp_moe_execution

    return plan_tp_moe_execution(
        num_tokens=max(int(tokens), 1),
        num_topk=int(topk),
        device=device,
        weight_plan=experts.plan,
        quant_mode=quant_mode,
        apply_router_weight_on_input=apply_router_weight_on_input,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )


def _b12x_moe_plan_supports_aux_stream_overlap(plan: Any) -> bool:
    # Namespaced b12x does not yet publish a plan capability predicate.
    # Some resident-grid plans use device-wide barriers, so lack of an explicit
    # guarantee must conservatively disable auxiliary-stream overlap.
    return False


def _b12x_scratch_nbytes(plan: Any) -> int:
    specs = plan.scratch_specs()
    if len(specs) != 1:
        raise RuntimeError(f"expected one b12x MoE scratch buffer, got {len(specs)}")
    spec = specs[0]
    if spec.dtype != torch.uint8:
        raise TypeError(f"expected b12x MoE scratch dtype uint8, got {spec.dtype}")
    return int(spec.shape[0])


def _dynamic_moe_warmup_tokens(
    *,
    topk: int,
    quant_mode: str,
    requested_tokens: int,
) -> int:
    """Return a small token count that selects b12x's dynamic MoE backend."""
    # This warmup-only policy probe is not exposed by the public facade. Actual
    # planning, binding, and execution below all use the public API.
    from b12x.moe.fused_moe._impl import select_tp_moe_backend

    tokens = max(int(requested_tokens), 1)
    topk = max(int(topk), 1)
    for _ in range(16):
        if (
            select_tp_moe_backend(
                num_tokens=tokens,
                num_topk=topk,
                quant_mode=quant_mode,
            )
            == "dynamic"
        ):
            return tokens
        tokens *= 2
    raise RuntimeError(
        "could not find a B12X dynamic MoE warmup token count for "
        f"topk={topk}, quant_mode={quant_mode!r}"
    )


def _b12x_moe_warmup_token_counts(
    *,
    max_tokens: int,
    token_counts: Iterable[int] = (),
) -> tuple[int, ...]:
    """Probe powers of two plus serving sizes supplied by vLLM."""
    max_tokens = max(int(max_tokens), 1)
    counts = {
        int(token_count)
        for token_count in token_counts
        if 0 < int(token_count) <= max_tokens
    }
    token_count = 1
    while token_count < max_tokens:
        counts.add(token_count)
        token_count *= 2
    counts.add(max_tokens)
    return tuple(sorted(counts))


def _b12x_moe_warmup_plan_signature(launch_plan: Any) -> tuple[Any, ...]:
    """Fields through which the planner selects a compiled kernel regime."""
    return (
        launch_plan.implementation,
        launch_plan.execution,
    )


def _workspace2_as_b12x_scratch(
    workspace2: torch.Tensor | None,
    plan: Any,
) -> torch.Tensor:
    if workspace2 is None:
        raise RuntimeError("B12X MoE requires vLLM workspace2 scratch")
    if not workspace2.is_contiguous():
        raise ValueError("B12X MoE workspace2 must be contiguous")
    scratch = workspace2.view(-1).view(torch.uint8)
    required_nbytes = _b12x_scratch_nbytes(plan)
    if int(scratch.numel()) < required_nbytes:
        raise ValueError(
            "B12X MoE workspace2 is too small for planned scratch: "
            f"have={int(scratch.numel())} bytes, need={required_nbytes} bytes"
        )
    return scratch


def _run_b12x_moe_fp4(
    *,
    a: torch.Tensor,
    experts: Any,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    input_scales_static: bool,
    unit_scale_contract: bool,
    plan: Any,
    scratch: torch.Tensor,
    activation_amax: torch.Tensor | None = None,
    layer_idx: int | None = None,
) -> None:
    """Call b12x MoE with caller-owned live scratch."""
    from b12x.moe.fused_moe import run as b12x_moe_fp4

    if _moe_zero_scratch_enabled():
        if _is_current_stream_capturing():
            raise RuntimeError("B12X_MOE_ZERO_SCRATCH is a diagnostic eager-only mode")
        scratch.zero_()

    binding = plan.bind(
        scratch=scratch,
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=input_scales_static,
        unit_scale_contract=unit_scale_contract,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
    )
    b12x_moe_fp4(binding=binding)


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI):
        return "silu"
    if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        return activation.value
    if activation == MoEActivation.RELU2:
        return "relu2"
    return activation.value


def _first_not_none(*values: Any) -> Any | None:
    for value in values:
        if value is not None:
            return value
    return None


def _replace_parameter_with_empty(
    layer: torch.nn.Module,
    param_name: str,
) -> torch.Tensor | None:
    param = getattr(layer, param_name, None)
    if not isinstance(param, torch.Tensor):
        return None
    empty = torch.empty((0,), dtype=param.dtype, device=param.device)
    replace_parameter(layer, param_name, empty)
    return getattr(layer, param_name)


def _set_quant_config_weight_scale(
    quant_config: FusedMoEQuantConfig,
    weight_name: str,
    scale: torch.Tensor,
) -> None:
    desc = getattr(quant_config, weight_name, None)
    if desc is not None and hasattr(desc, "scale"):
        desc.scale = scale
        return

    public_name = "w1_scale" if weight_name == "_w1" else "w2_scale"
    if hasattr(quant_config, public_name):
        setattr(quant_config, public_name, scale)


def _maybe_release_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda" or _is_current_stream_capturing():
        return
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        accelerator.empty_cache()
    else:
        torch.cuda.empty_cache()


def _raise_if_capture_copy_required(tensor: torch.Tensor, description: str) -> None:
    if tensor.device.type != "cuda" or not _is_current_stream_capturing():
        return
    raise RuntimeError(
        f"B12X MoE {description} would allocate during CUDA graph capture"
    )


def _is_current_stream_capturing() -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    is_capturing = getattr(cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


def _maybe_repeat_check_b12x_moe(
    *,
    original_output: torch.Tensor,
    a: torch.Tensor,
    experts: Any,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    input_scales_static: bool,
    unit_scale_contract: bool,
    plan: Any,
    scratch: torch.Tensor,
) -> None:
    global _moe_repeat_check_reports

    if not _moe_repeat_check_enabled() or _is_current_stream_capturing():
        return
    if _moe_repeat_check_after_engine_start() and not _env_flag(
        "B12X_VLLM_ENGINE_STARTED"
    ):
        return
    max_reports = _env_int("B12X_MOE_REPEAT_CHECK_MAX_REPORTS", 8)
    if _moe_repeat_check_reports >= max_reports:
        return

    repeat_output = torch.empty_like(original_output)
    _run_b12x_moe_fp4(
        a=a,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=repeat_output,
        input_scales_static=input_scales_static,
        unit_scale_contract=unit_scale_contract,
        plan=plan,
        scratch=scratch,
    )

    original_f = original_output.float()
    repeat_f = repeat_output.float()
    finite = bool(
        torch.isfinite(original_f).all().item()
        and torch.isfinite(repeat_f).all().item()
    )
    if original_f.numel() == 0:
        max_abs = mean_abs = 0.0
        cosine = 1.0
    else:
        diff = (original_f - repeat_f).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        original_flat = original_f.flatten()
        repeat_flat = repeat_f.flatten()
        denom = original_flat.norm() * repeat_flat.norm()
        cosine = (
            float((original_flat.dot(repeat_flat) / denom).item())
            if float(denom.item()) != 0.0
            else 1.0
        )

    _moe_repeat_check_reports += 1
    core_plan = _moe_core_plan(plan)
    logger.warning(
        "B12X MoE repeat check: finite=%s max_abs=%g mean_abs=%g "
        "cosine=%g shape=%s dtype=%s quant_mode=%s activation=%s "
        "implementation=%s routed_rows=%s max_rows=%s "
        "max_tokens_per_launch=%s topk=%s",
        finite,
        max_abs,
        mean_abs,
        cosine,
        tuple(original_output.shape),
        original_output.dtype,
        getattr(core_plan, "quant_mode", None),
        getattr(core_plan, "activation", getattr(experts, "activation", None)),
        getattr(core_plan, "implementation", None),
        getattr(core_plan, "routed_rows", None),
        getattr(core_plan, "max_rows", None),
        getattr(core_plan, "max_tokens_per_launch", None),
        getattr(core_plan, "num_topk", None),
    )


def _normalize_b12x_moe_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    if topk_ids.dtype != torch.int32:
        _raise_if_capture_copy_required(topk_ids, "topk_ids dtype normalization")
        topk_ids = topk_ids.to(torch.int32)
    if not topk_ids.is_contiguous():
        _raise_if_capture_copy_required(topk_ids, "topk_ids contiguity normalization")
        topk_ids = topk_ids.contiguous()
    return topk_ids


def _normalize_b12x_moe_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    if topk_weights.dtype != torch.float32:
        _raise_if_capture_copy_required(
            topk_weights,
            "topk_weights dtype normalization",
        )
        topk_weights = topk_weights.to(torch.float32)
    if not topk_weights.is_contiguous():
        _raise_if_capture_copy_required(
            topk_weights,
            "topk_weights contiguity normalization",
        )
        topk_weights = topk_weights.contiguous()
    return topk_weights


def _normalize_modelopt_expert_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.dim() == 2:
        if scale.size(1) not in (1, 2):
            raise ValueError(
                "expected ModelOpt expert scale second dimension to be 1 or 2, "
                f"got {tuple(scale.shape)}"
            )
        scale = scale[:, 0]
    return scale.contiguous()


def _has_b12x() -> bool:
    try:
        from b12x.moe.fused_moe import run as b12x_moe_fp4  # noqa: F401

        return True
    except ImportError:
        return False


class B12xExperts(mk.FusedMoEExpertsModular):
    """Native FP4 MoE backend powered by b12x kernels."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype in ("mxfp4", "nvfp4"), (
            "B12xExperts only supports native FP4 weights, got "
            f"{quant_config.weight_quant_dtype}"
        )

        self._prepared_experts: Any | None = None
        self._source_parameters_released = False
        self._unit_scale_by_device: dict[torch.device, torch.Tensor] = {}
        self._activation_amax_enabled = _moe_activation_amax_enabled()
        self._activation_amax_base_num_layers = _current_config_num_hidden_layers()
        self._activation_amax_state_key: tuple[str, str, int] | None = None
        self._activation_amax_layer_idx: int | None = None

    def _quant_mode(self) -> str:
        source_format = self._source_format()
        if _moe_force_a8_enabled():
            if source_format == "fp4_e8m0_k32":
                logger.warning_once(
                    "B12X MoE force-A8 enabled: using quant_mode=w4a8_mx "
                    "for E8M0 FP4 weights."
                )
                return "w4a8_mx"
            if source_format == "modelopt_nvfp4":
                logger.warning_once(
                    "B12X MoE force-A8 enabled: using quant_mode=w4a8_nvfp4 "
                    "for NVFP4 weights."
                )
                return "w4a8_nvfp4"
            raise RuntimeError(
                f"B12X MoE force-A8 does not support source_format={source_format!r}"
            )
        if _moe_force_a16_enabled():
            logger.warning_once("B12X MoE force-A16 enabled: using quant_mode=w4a16.")
            return "w4a16"
        return "nvfp4" if self.quant_config.quant_dtype == "nvfp4" else "w4a16"

    def _source_format(self) -> str:
        if self.quant_config.weight_quant_dtype == "nvfp4":
            return "modelopt_nvfp4"
        return "fp4_e8m0_k32"

    def _w13_layout(self) -> str:
        if self._source_format() == "modelopt_nvfp4" and self._quant_mode() == "w4a16":
            return "w13"
        # vLLM fused MoE loading stores fused W13 as [w1/gate, w3/up], which is
        # the row order consumed by b12x for the runtime SwiGLU path. Declaring
        # "up_gate" here swaps gate/up in every expert -> corrupted MoE output.
        return "w31"

    def _unit_expert_scale(
        self, device: torch.device, num_experts: int
    ) -> torch.Tensor:
        scale = self._unit_scale_by_device.get(device)
        if scale is None or scale.numel() != num_experts:
            scale = torch.ones(num_experts, dtype=torch.float32, device=device)
            self._unit_scale_by_device[device] = scale
        return scale

    def _activation_amax_enabled_for_layer(self) -> bool:
        enabled = getattr(self, "_activation_amax_enabled", None)
        if enabled is None:
            return _moe_activation_amax_enabled()
        return bool(enabled)

    def _register_activation_amax(
        self,
        *,
        layer: torch.nn.Module,
        device: torch.device,
        num_experts: int,
    ) -> None:
        if (
            not self._activation_amax_enabled_for_layer()
            or self._quant_mode() != "w4a16"
        ):
            return

        prefix = str(getattr(layer, "layer_name", ""))
        external_layer_idx = _parse_layer_index(prefix)
        base_num_layers = getattr(
            self,
            "_activation_amax_base_num_layers",
            None,
        )
        model_tag = _activation_amax_model_tag(
            prefix,
            external_layer_idx,
            base_num_layers,
        )
        state = _get_activation_amax_state(
            model_tag=model_tag,
            device=device,
            num_experts=num_experts,
        )
        slot = state.register(
            owner=self,
            prefix=prefix,
            external_layer_idx=external_layer_idx,
            slot_hint=_activation_amax_slot_hint(
                model_tag=model_tag,
                external_layer_idx=external_layer_idx,
                base_num_layers=base_num_layers,
            ),
        )
        self._activation_amax_state_key = _activation_amax_state_key(
            model_tag=model_tag,
            device=device,
            num_experts=num_experts,
        )
        self._activation_amax_layer_idx = slot

    def _activation_amax_args(
        self,
        *,
        device: torch.device,
        num_experts: int,
    ) -> tuple[torch.Tensor | None, int | None]:
        if (
            not self._activation_amax_enabled_for_layer()
            or self._quant_mode() != "w4a16"
        ):
            return None, None

        key = getattr(self, "_activation_amax_state_key", None)
        layer_idx = getattr(self, "_activation_amax_layer_idx", None)
        if key is None or layer_idx is None:
            raise RuntimeError(
                "B12X MoE activation-amax was enabled but this W4A16 layer "
                "was not registered before launch."
            )

        state = _activation_amax_states.get(key)
        if state is None:
            raise RuntimeError("B12X MoE activation-amax state was cleared.")
        if state.device != device or state.num_experts != int(num_experts):
            raise RuntimeError(
                "B12X MoE activation-amax state does not match the live "
                f"layer: state device={state.device}, experts={state.num_experts}; "
                f"live device={device}, experts={num_experts}."
            )
        state.mark_used()
        return state.tensor, int(layer_idx)

    def _weight_global_scale(
        self,
        device: torch.device,
        num_experts: int,
        *,
        weight_name: str,
    ) -> torch.Tensor:
        if self._source_format() != "modelopt_nvfp4":
            return self._unit_expert_scale(device, num_experts)

        if weight_name == "w1":
            scale = self.g1_alphas
        elif weight_name == "w2":
            scale = self.g2_alphas
        else:
            raise ValueError(f"unknown b12x weight name: {weight_name}")

        if scale is None:
            raise RuntimeError(
                f"B12X ModelOpt NVFP4 MoE requires {weight_name} global scales"
            )
        if int(scale.numel()) != num_experts:
            raise ValueError(
                f"B12X ModelOpt NVFP4 MoE expected {num_experts} "
                f"{weight_name} global scales, got {int(scale.numel())}"
            )
        if scale.device != device:
            _raise_if_capture_copy_required(
                scale,
                f"{weight_name} global scale device normalization",
            )
            scale = scale.to(device=device)
        if scale.dtype != torch.float32:
            _raise_if_capture_copy_required(
                scale,
                f"{weight_name} global scale dtype normalization",
            )
            scale = scale.to(torch.float32)
        if not scale.is_contiguous():
            _raise_if_capture_copy_required(
                scale,
                f"{weight_name} global scale contiguity normalization",
            )
            scale = scale.contiguous()
        return scale

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare b12x-owned weight metadata before graph capture."""
        device = layer.w13_weight.device
        moe_config = getattr(self, "moe_config", None)
        params_dtype = getattr(moe_config, "in_dtype", torch.bfloat16)
        activation = getattr(layer, "activation", None)
        if activation is None:
            activation = getattr(moe_config, "activation", MoEActivation.SILU)
        activation = cast(MoEActivation, activation)

        prepared = self._get_or_prepare_experts(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            activation=activation,
            params_dtype=params_dtype,
        )
        if self._quant_mode() == "w4a16":
            self._register_activation_amax(
                layer=layer,
                device=device,
                num_experts=prepared.num_experts,
            )
        if prepared.plan.discards_source_parameters:
            self._release_source_parameters(layer)
            _maybe_release_cuda_cache(device)

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(120) and _has_b12x()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kMxfp4Static, None),
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SWIGLUOAI,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size <= 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _get_or_prepare_experts(
        self,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        params_dtype: torch.dtype,
    ):
        quant_mode = self._quant_mode()
        prepared = self._prepared_experts
        if prepared is not None:
            requested_dtype = str(params_dtype).removeprefix("torch.")
            if (
                quant_mode in prepared.plan.quant_modes
                and requested_dtype == prepared.plan.io_dtype
                and _b12x_activation_name(activation) == prepared.plan.activation
            ):
                return prepared
            raise RuntimeError(
                "B12X FP4 MoE already transferred its source allocation into "
                "one prepared expert owner; the requested runtime contract does "
                "not match that owner: "
                f"quant_mode={quant_mode!r}, dtype={requested_dtype!r}, "
                f"activation={_b12x_activation_name(activation)!r}; "
                f"prepared_modes={sorted(prepared.plan.quant_modes)}, "
                f"prepared_dtype={prepared.plan.io_dtype!r}, "
                f"prepared_activation={prepared.plan.activation!r}."
            )

        if self._source_parameters_released:
            raise RuntimeError(
                "B12X FP4 MoE source parameters were released without a "
                "prepared expert owner"
            )

        if w1.device.type == "cuda" and _is_current_stream_capturing():
            raise RuntimeError(
                "B12X FP4 MoE weights were not prepared before CUDA "
                f"graph capture for dtype {params_dtype}."
            )
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for B12xExperts"
        )

        num_experts = int(w1.shape[0])
        hidden_size = int(w2.shape[1])
        intermediate_size = int(w2.shape[2]) * 2
        unit_scale = self._unit_expert_scale(w1.device, num_experts)
        if quant_mode in ("nvfp4", "w4a8_nvfp4"):
            if self.quant_config.weight_quant_dtype != "nvfp4":
                raise RuntimeError(f"B12X {quant_mode} mode requires NVFP4 weights")
            if self.g1_alphas is None or self.g2_alphas is None:
                raise RuntimeError(
                    f"B12X {quant_mode} MoE requires w1/w2 global scales"
                )
            if self.a1_gscale is None or self.a2_gscale is None:
                raise RuntimeError(
                    f"B12X {quant_mode} MoE requires a1/a2 global scales"
                )
            w1_global_scale = self._weight_global_scale(
                w1.device, num_experts, weight_name="w1"
            )
            w2_global_scale = self._weight_global_scale(
                w2.device, num_experts, weight_name="w2"
            )
            a1_gscale = _normalize_modelopt_expert_scale(self.a1_gscale)
            a2_gscale = _normalize_modelopt_expert_scale(self.a2_gscale)
        else:
            w1_global_scale = self._weight_global_scale(
                w1.device, num_experts, weight_name="w1"
            )
            w2_global_scale = self._weight_global_scale(
                w2.device, num_experts, weight_name="w2"
            )
            a1_gscale = unit_scale
            a2_gscale = unit_scale

        from b12x.moe._shared.execution import PreparedWeightLayout
        from b12x.moe.fused_moe import (
            plan_weights as plan_b12x_fp4_moe_weights,
        )
        from b12x.moe.fused_moe import (
            prepare_weights as prepare_b12x_fp4_moe_weights,
        )

        w4a16_layout = (
            PreparedWeightLayout.SOURCE_NATIVE
            if quant_mode == "w4a16" and envs.VLLM_B12X_MOE_FORCE_MODELOPT_PREP
            else None
        )
        weight_plan = plan_b12x_fp4_moe_weights(
            quant_modes=quant_mode,
            source_format=self._source_format(),
            activation=_b12x_activation_name(activation),
            params_dtype=params_dtype,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_layout=self._w13_layout(),
            w4a16_layout=w4a16_layout,
        )
        prepared = prepare_b12x_fp4_moe_weights(
            plan=weight_plan,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_global_scale=w1_global_scale,
            a1_gscale=a1_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_global_scale=w2_global_scale,
            a2_gscale=a2_gscale,
            params_dtype=params_dtype,
        )
        self._prepared_experts = prepared
        return prepared

    def _b12x_swiglu_params(
        self,
        activation: MoEActivation,
    ) -> tuple[float | None, float | None, float | None]:
        swiglu_limit = _first_not_none(
            getattr(self.quant_config, "gemm1_clamp_limit", None),
            getattr(self.moe_config, "swiglu_limit", None),
        )
        if activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            return swiglu_limit, None, None

        swiglu_alpha = _first_not_none(
            getattr(self.quant_config, "gemm1_alpha", None),
            getattr(self.moe_config, "swiglu_alpha", None),
        )
        swiglu_beta = _first_not_none(
            getattr(self.quant_config, "gemm1_beta", None),
            getattr(self.moe_config, "swiglu_beta", None),
        )
        return swiglu_limit, swiglu_alpha, swiglu_beta

    def _lookup_prepared_experts(self) -> Any | None:
        prepared = self._prepared_experts
        if prepared is None:
            return None
        quant_mode = self._quant_mode()
        if quant_mode not in prepared.plan.quant_modes:
            raise RuntimeError(
                f"B12X quant_mode={quant_mode!r} does not match the prepared "
                f"expert owner {sorted(prepared.plan.quant_modes)}"
            )
        return prepared

    def _warmup_metadata(self, layer: torch.nn.Module) -> SimpleNamespace | None:
        w1 = getattr(layer, "w13_weight", None)
        w2 = getattr(layer, "w2_weight", None)
        if not isinstance(w1, torch.Tensor) or not isinstance(w2, torch.Tensor):
            return None

        activation = getattr(
            layer,
            "activation",
            getattr(self.moe_config, "activation", MoEActivation.SILU),
        )
        if isinstance(activation, str):
            activation = MoEActivation.from_str(activation)
        activation = cast(MoEActivation, activation)

        quant_mode = self._quant_mode()
        prepared = self._lookup_prepared_experts()
        if (w1.numel() == 0 or w2.numel() == 0) and prepared is None:
            return None

        if prepared is not None:
            num_experts = prepared.num_experts
            n = prepared.intermediate_size
            k = prepared.hidden_size
            device = prepared.w1_fp4.device
        else:
            num_experts = int(w1.shape[0])
            n = int(w2.shape[2]) * 2
            k = int(w2.shape[1])
            device = w1.device

        swiglu_limit, swiglu_alpha, swiglu_beta = self._b12x_swiglu_params(activation)
        if (
            not _supports_swiglu_limit(quant_mode)
            and activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE
        ):
            swiglu_limit = None

        return SimpleNamespace(
            w1=w1,
            w2=w2,
            activation=activation,
            activation_name=_b12x_activation_name(activation),
            quant_mode=quant_mode,
            num_experts=num_experts,
            n=n,
            k=k,
            device=device,
            topk=int(self.moe_config.experts_per_token),
            dtype=getattr(self.moe_config, "in_dtype", torch.bfloat16),
            apply_router_weight_on_input=bool(
                getattr(layer, "apply_router_weight_on_input", False)
            ),
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            collect_activation_amax=(
                self._activation_amax_enabled_for_layer() and quant_mode == "w4a16"
            ),
        )

    def warmup_dynamic_signature(
        self,
        layer: torch.nn.Module,
    ) -> tuple[Any, ...] | None:
        meta = self._warmup_metadata(layer)
        if meta is None:
            return None
        device = meta.device
        return (
            device.type,
            device.index,
            meta.dtype,
            meta.quant_mode,
            self._source_format(),
            self._w13_layout(),
            meta.num_experts,
            meta.k,
            meta.n,
            meta.topk,
            meta.activation_name,
            meta.apply_router_weight_on_input,
            meta.swiglu_limit,
            meta.swiglu_alpha,
            meta.swiglu_beta,
            meta.collect_activation_amax,
        )

    @torch.inference_mode()
    def warmup_dynamic_launches(
        self,
        layer: torch.nn.Module,
        *,
        token_counts: Iterable[int],
    ) -> int:
        """Compile one representative of every planned dynamic MoE regime."""
        meta = self._warmup_metadata(layer)
        if meta is None:
            return 0

        prepared = self._get_or_prepare_experts(
            w1=meta.w1,
            w2=meta.w2,
            activation=meta.activation,
            params_dtype=meta.dtype,
        )
        num_experts = prepared.num_experts
        input_scales_static = True
        unit_scale_contract = meta.quant_mode == "w4a16"

        activation_amax, activation_layer_idx = (None, None)
        if meta.collect_activation_amax:
            activation_amax, activation_layer_idx = self._activation_amax_args(
                device=meta.device,
                num_experts=num_experts,
            )

        # Execution planning is allocation- and compilation-free. Probe the
        # serving range, then retain the smallest token count selecting each
        # execution plan so large prefill regimes are compiled without
        # allocating max-batch inputs.
        launch_tokens: dict[tuple[Any, ...], int] = {}
        for requested_tokens in sorted(set(int(count) for count in token_counts)):
            tokens = _dynamic_moe_warmup_tokens(
                topk=meta.topk,
                quant_mode=meta.quant_mode,
                requested_tokens=requested_tokens,
            )
            launch_plan = _plan_b12x_moe_execution(
                tokens=tokens,
                topk=meta.topk,
                device=meta.device,
                quant_mode=meta.quant_mode,
                experts=prepared,
                apply_router_weight_on_input=meta.apply_router_weight_on_input,
                swiglu_limit=meta.swiglu_limit,
                swiglu_alpha=meta.swiglu_alpha,
                swiglu_beta=meta.swiglu_beta,
            )
            launch_tokens.setdefault(
                _b12x_moe_warmup_plan_signature(launch_plan),
                tokens,
            )

        for tokens in launch_tokens.values():
            plan = _plan_b12x_moe_fp4_scratch(
                tokens=tokens,
                topk=meta.topk,
                device=meta.device,
                quant_mode=meta.quant_mode,
                experts=prepared,
                apply_router_weight_on_input=meta.apply_router_weight_on_input,
                swiglu_limit=meta.swiglu_limit,
                swiglu_alpha=meta.swiglu_alpha,
                swiglu_beta=meta.swiglu_beta,
                collect_activation_amax=meta.collect_activation_amax,
            )
            hidden_states = torch.zeros(
                (tokens, meta.k),
                dtype=meta.dtype,
                device=meta.device,
            )
            output = torch.empty_like(hidden_states)
            topk_ids = (
                torch.arange(meta.topk, device=meta.device, dtype=torch.int32)
                .unsqueeze(0)
                .expand(tokens, -1)
                .contiguous()
            )
            if num_experts > 0:
                topk_ids.remainder_(num_experts)
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
            _run_b12x_moe_fp4(
                a=hidden_states,
                experts=prepared,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=output,
                input_scales_static=input_scales_static,
                unit_scale_contract=unit_scale_contract,
                plan=plan,
                scratch=scratch,
                activation_amax=activation_amax,
                layer_idx=activation_layer_idx,
            )
        return len(launch_tokens)

    def _release_source_parameters(self, layer: torch.nn.Module) -> None:
        """Leave the planner-selected expert owner as the sole allocation."""

        if self._source_parameters_released:
            return

        w1_scale = _replace_parameter_with_empty(layer, "w13_weight_scale")
        w2_scale = _replace_parameter_with_empty(layer, "w2_weight_scale")
        if w1_scale is not None:
            _set_quant_config_weight_scale(self.quant_config, "_w1", w1_scale)
        if w2_scale is not None:
            _set_quant_config_weight_scale(self.quant_config, "_w2", w2_scale)

        _replace_parameter_with_empty(layer, "w13_weight")
        _replace_parameter_with_empty(layer, "w2_weight")
        self._source_parameters_released = True

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if w1.numel() != 0 and w2.numel() != 0:
            return super().moe_problem_size(a1, w1, w2, topk_ids)

        prepared = self._lookup_prepared_experts()
        if prepared is None:
            return super().moe_problem_size(a1, w1, w2, topk_ids)

        if a1.dim() == 2:
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            m = a1.size(0)
        else:
            assert a1.dim() == 3
            m = a1.size(1)

        intermediate_size = int(prepared.intermediate_size)
        n = intermediate_size * 2
        return (
            int(prepared.num_experts),
            m,
            n,
            a1.size(-1),
            topk_ids.size(1),
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
        quant_mode = self._quant_mode()
        prepared = self._lookup_prepared_experts()
        if prepared is None:
            raise RuntimeError(
                "B12X MoE workspace planning requires prepared weights; "
                "process_weights_after_loading must run first"
            )
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        workspace_dtype = getattr(self.moe_config, "in_dtype", torch.bfloat16)
        swiglu_limit, swiglu_alpha, swiglu_beta = self._b12x_swiglu_params(activation)
        if (
            not _supports_swiglu_limit(quant_mode)
            and activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE
        ):
            swiglu_limit = None
        plan = _plan_b12x_moe_fp4_scratch(
            tokens=max(int(M), 1),
            topk=int(topk),
            device=device,
            quant_mode=quant_mode,
            experts=prepared,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            collect_activation_amax=(
                self._activation_amax_enabled_for_layer() and quant_mode == "w4a16"
            ),
        )
        scratch_elements = max(
            1,
            _ceil_div(_b12x_scratch_nbytes(plan), _dtype_element_size(workspace_dtype)),
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
        prepared = self._get_or_prepare_experts(
            w1=w1,
            w2=w2,
            activation=activation,
            params_dtype=hidden_states.dtype,
        )
        quant_mode = self._quant_mode()

        if expert_map is not None:
            raise RuntimeError(
                "B12X MoE does not support expert_map with the current b12x_moe_fp4 API"
            )

        num_experts = prepared.num_experts
        input_scales_static = True
        unit_scale_contract = quant_mode == "w4a16"
        activation_amax, activation_layer_idx = (None, None)
        if quant_mode == "w4a16" and self._activation_amax_enabled_for_layer():
            activation_amax, activation_layer_idx = self._activation_amax_args(
                device=hidden_states.device,
                num_experts=num_experts,
            )
        swiglu_limit, swiglu_alpha, swiglu_beta = self._b12x_swiglu_params(activation)
        if (
            not _supports_swiglu_limit(quant_mode)
            and activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE
        ):
            swiglu_limit = None
        topk_ids = _normalize_b12x_moe_topk_ids(topk_ids)
        topk_weights = _normalize_b12x_moe_topk_weights(topk_weights)
        plan = _plan_b12x_moe_fp4_scratch(
            tokens=int(hidden_states.shape[0]),
            topk=int(topk_ids.shape[1]),
            device=hidden_states.device,
            quant_mode=quant_mode,
            experts=prepared,
            apply_router_weight_on_input=(
                apply_router_weight_on_input
                if apply_router_weight_on_input is not None
                else False
            ),
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            collect_activation_amax=activation_amax is not None,
        )
        scratch = _workspace2_as_b12x_scratch(workspace2, plan)

        _run_b12x_moe_fp4(
            a=hidden_states,
            experts=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            input_scales_static=input_scales_static,
            unit_scale_contract=unit_scale_contract,
            plan=plan,
            scratch=scratch,
            activation_amax=activation_amax,
            layer_idx=activation_layer_idx,
        )
        if activation_amax is None:
            _maybe_repeat_check_b12x_moe(
                original_output=output,
                a=hidden_states,
                experts=prepared,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                input_scales_static=input_scales_static,
                unit_scale_contract=unit_scale_contract,
                plan=plan,
                scratch=scratch,
            )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")


def warmup_b12x_moe_dynamic(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    token_counts: Iterable[int] = (),
) -> int:
    """Warm unique b12x dynamic MoE planner regimes in a loaded model."""
    candidates = _b12x_moe_warmup_token_counts(
        max_tokens=max_tokens,
        token_counts=token_counts,
    )
    seen: set[tuple[Any, ...]] = set()
    warmed = 0
    for module in model.modules():
        routed_experts = getattr(module, "routed_experts", None)
        if routed_experts is None:
            continue
        quant_method = getattr(routed_experts, "quant_method", None)
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        fused_experts = getattr(moe_kernel, "fused_experts", None)
        if not isinstance(fused_experts, B12xExperts):
            continue

        signature = fused_experts.warmup_dynamic_signature(routed_experts)
        if signature is None or signature in seen:
            continue
        seen.add(signature)
        warmed += fused_experts.warmup_dynamic_launches(
            routed_experts,
            token_counts=candidates,
        )

    if warmed:
        logger.info(
            "Warmed up %d B12X MoE dynamic launch variant(s) across %d "
            "expert signature(s).",
            warmed,
            len(seen),
        )
    return warmed
