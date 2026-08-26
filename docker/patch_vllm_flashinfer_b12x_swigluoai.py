#!/usr/bin/env python3
"""Apply the runtime subset of vLLM PR #47392 needed by FlashInfer B12x.

The upstream PR also changes tests and documentation.  This downstream patch
only teaches the production backend to pass SwiGLU-OAI parameters through to
FlashInfer and lets the NVFP4 oracle select it when that FlashInfer API exists.

The patch is idempotent.  It skips refs that predate FlashInfer B12x and fails
on unknown source shapes instead of making a best-effort rewrite.
"""

from __future__ import annotations

import sys
from pathlib import Path


EXPERT_REL = Path(
    "vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"
)
ORACLE_REL = Path("vllm/model_executor/layers/fused_moe/oracle/nvfp4.py")
FLASHINFER_UTIL_REL = Path("vllm/utils/flashinfer.py")


class PatchError(RuntimeError):
    pass


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(
            f"expected one {description} source anchor, found {count}; "
            "the vLLM source shape has changed"
        )
    return text.replace(old, new, 1)


def patch_flashinfer_util(text: str) -> str:
    helper_name = "has_flashinfer_b12x_moe_activation"
    if f"def {helper_name}()" not in text:
        if "\nimport inspect\n" not in text:
            text = replace_once(
                text,
                "import importlib.util\n",
                "import importlib.util\nimport inspect\n",
                "FlashInfer import",
            )

        anchor = (
            "    return True\n\n\n"
            "@functools.cache\n"
            "def has_nvidia_artifactory() -> bool:"
        )
        helper = (
            "    return True\n\n\n"
            "@functools.cache\n"
            f"def {helper_name}() -> bool:\n"
            '    """Return whether B12xMoEWrapper accepts SwiGLU parameters."""\n'
            "    if not has_flashinfer_b12x_moe():\n"
            "        return False\n"
            '    mod = _get_submodule("flashinfer.fused_moe")\n'
            '    if mod is None or not hasattr(mod, "B12xMoEWrapper"):\n'
            "        return False\n"
            "    return (\n"
            '        "swiglu_limit" '
            "in inspect.signature(mod.B12xMoEWrapper).parameters\n"
            "    )\n\n\n"
            "@functools.cache\n"
            "def has_nvidia_artifactory() -> bool:"
        )
        text = replace_once(text, anchor, helper, "B12x capability helper")

    export = f'    "{helper_name}",\n'
    if export not in text:
        text = replace_once(
            text,
            '    "has_flashinfer_b12x_moe",\n',
            '    "has_flashinfer_b12x_moe",\n' + export,
            "FlashInfer __all__ entry",
        )

    if f"def {helper_name}()" not in text or export not in text:
        raise PatchError("FlashInfer B12x activation capability helper is incomplete")
    return text


def patch_expert(text: str) -> str:
    helper_name = "has_flashinfer_b12x_moe_activation"
    helper_import = f"    {helper_name},\n"
    if helper_import not in text:
        text = replace_once(
            text,
            "    has_flashinfer_b12x_moe,\n",
            "    has_flashinfer_b12x_moe,\n" + helper_import,
            "FlashInfer B12x helper import",
        )

    activation_entry = (
        '        MoEActivation.SWIGLUOAI_UNINTERLEAVE: '
        '"swigluoai_uninterleave",\n'
    )
    if activation_entry not in text:
        text = replace_once(
            text,
            '        MoEActivation.RELU2_NO_MUL: "relu2",\n',
            '        MoEActivation.RELU2_NO_MUL: "relu2",\n' + activation_entry,
            "B12x activation map entry",
        )

    if "        self.swiglu_alpha = _resolve(" not in text:
        config_block = """
        # ModelOpt NVFP4 checkpoints carry SwiGLU parameters on moe_config,
        # while other quantization paths may put them on quant_config.
        def _resolve(
            quant_val: float | None, moe_val: float | None
        ) -> float | None:
            return quant_val if quant_val is not None else moe_val

        self.swiglu_alpha = _resolve(
            quant_config.gemm1_alpha, moe_config.swiglu_alpha
        )
        self.swiglu_beta = _resolve(
            quant_config.gemm1_beta, moe_config.swiglu_beta
        )
        self.swiglu_limit = _resolve(
            quant_config.gemm1_clamp_limit, moe_config.swiglu_limit
        )
        if activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            if self.swiglu_limit is None:
                raise ValueError(
                    "swigluoai_uninterleave requires swiglu_limit "
                    "(gemm1_clamp_limit), but none was provided."
                )
        elif self.swiglu_limit is not None:
            raise ValueError(
                "FlashInferB12xExperts only applies swiglu_limit with the "
                f"swigluoai_uninterleave activation, got {activation}."
            )
"""
        text = replace_once(
            text,
            "        self._activation_str = self._ACTIVATION_MAP[activation]\n",
            "        self._activation_str = self._ACTIVATION_MAP[activation]\n"
            + config_block,
            "B12x activation configuration",
        )

    support_check = "            and has_flashinfer_b12x_moe_activation()\n"
    if support_check not in text:
        legacy_support = (
            "    def _supports_activation(activation: MoEActivation) -> bool:\n"
            "        return activation in ("
            "MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)\n"
        )
        patched_legacy_support = (
            "    def _supports_activation(activation: MoEActivation) -> bool:\n"
            "        if activation in ("
            "MoEActivation.SILU, MoEActivation.RELU2_NO_MUL):\n"
            "            return True\n"
            "        return (\n"
            "            activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE\n"
            + support_check
            + "        )\n"
        )
        gelu_support = (
            "    def _supports_activation(activation: MoEActivation) -> bool:\n"
            "        return activation in (\n"
            "            MoEActivation.SILU,\n"
            "            MoEActivation.GELU_TANH,\n"
            "            MoEActivation.RELU2_NO_MUL,\n"
            "        )\n"
        )
        patched_gelu_support = (
            "    def _supports_activation(activation: MoEActivation) -> bool:\n"
            "        if activation in (\n"
            "            MoEActivation.SILU,\n"
            "            MoEActivation.GELU_TANH,\n"
            "            MoEActivation.RELU2_NO_MUL,\n"
            "        ):\n"
            "            return True\n"
            "        return (\n"
            "            activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE\n"
            + support_check
            + "        )\n"
        )
        known_support_shapes = (
            (legacy_support, patched_legacy_support),
            (gelu_support, patched_gelu_support),
        )
        matches = [(old, new) for old, new in known_support_shapes if old in text]
        match_count = sum(text.count(old) for old, _ in known_support_shapes)
        if match_count != 1:
            raise PatchError(
                "expected one B12x activation support predicate source anchor, "
                f"found {match_count}; the vLLM source shape has changed"
            )
        old_support, new_support = matches[0]
        text = text.replace(old_support, new_support, 1)

    if "        swiglu_kwargs: dict[str, float] = {}\n" not in text:
        wrapper_kwargs = """        swiglu_kwargs: dict[str, float] = {}
        if self.swiglu_limit is not None:
            # vLLM defaults unset alpha/beta to 1.0/0.0, while FlashInfer's
            # wrapper defaults are 1.702/1.0, so pass the resolved values.
            swiglu_kwargs = {
                "swiglu_limit": self.swiglu_limit,
                "swiglu_alpha": (
                    1.0 if self.swiglu_alpha is None else self.swiglu_alpha
                ),
                "swiglu_beta": (
                    0.0 if self.swiglu_beta is None else self.swiglu_beta
                ),
            }

"""
        text = replace_once(
            text,
            "        self._wrapper = B12xMoEWrapper(\n",
            wrapper_kwargs + "        self._wrapper = B12xMoEWrapper(\n",
            "B12x wrapper construction",
        )

    if "            **swiglu_kwargs,\n" not in text:
        text = replace_once(
            text,
            "            activation=self._activation_str,\n",
            "            activation=self._activation_str,\n"
            "            **swiglu_kwargs,\n",
            "B12x wrapper SwiGLU keyword forwarding",
        )

    required = (
        helper_import,
        activation_entry,
        "        self.swiglu_alpha = _resolve(",
        support_check,
        "            **swiglu_kwargs,\n",
    )
    missing = [marker.strip() for marker in required if marker not in text]
    if missing:
        raise PatchError(f"FlashInferB12xExperts patch is incomplete: {missing}")
    return text


def patch_oracle(text: str) -> str:
    activation_import = (
        "from vllm.model_executor.layers.fused_moe.activation import "
        "MoEActivation\n"
    )
    if activation_import not in text:
        text = replace_once(
            text,
            "from vllm.logger import init_logger\n",
            "from vllm.logger import init_logger\n" + activation_import,
            "MoEActivation import",
        )

    helper_import = (
        "from vllm.utils.flashinfer import "
        "has_flashinfer_b12x_moe_activation\n"
    )
    if helper_import not in text:
        text = replace_once(
            text,
            "\nlogger = init_logger(__name__)\n",
            helper_import + "\nlogger = init_logger(__name__)\n",
            "B12x activation helper import",
        )

    set_marker = "    NVFP4_BACKENDS_WITH_CLAMP = {\n"
    set_start = text.find(set_marker)
    if set_start == -1:
        raise PatchError("NVFP4_BACKENDS_WITH_CLAMP is absent from the oracle")
    set_end = text.find("\n    }", set_start + len(set_marker))
    if set_end == -1:
        raise PatchError("NVFP4_BACKENDS_WITH_CLAMP has an unknown source shape")
    set_end += len("\n    }")
    clamp_set = text[set_start:set_end]

    add_marker = (
        "NVFP4_BACKENDS_WITH_CLAMP.add("
        "NvFp4MoeBackend.FLASHINFER_B12X)"
    )
    if (
        "NvFp4MoeBackend.FLASHINFER_B12X" not in clamp_set
        and add_marker not in text
    ):
        condition = """

    # B12x applies the clamp only for SwiGLU-OAI and only when the installed
    # FlashInfer wrapper exposes the corresponding activation parameters.
    if (
        config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        and has_flashinfer_b12x_moe_activation()
    ):
        NVFP4_BACKENDS_WITH_CLAMP.add(NvFp4MoeBackend.FLASHINFER_B12X)"""
        text = text[:set_end] + condition + text[set_end:]

    set_start = text.find(set_marker)
    set_end = text.find("\n    }", set_start + len(set_marker)) + len("\n    }")
    clamp_set = text[set_start:set_end]
    if (
        "NvFp4MoeBackend.FLASHINFER_B12X" not in clamp_set
        and add_marker not in text
    ):
        raise PatchError("NVFP4 oracle patch is incomplete")
    return text


def main() -> int:
    source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    expert_path = source_root / EXPERT_REL
    if not expert_path.exists():
        print(f"{EXPERT_REL} is absent; targeted PR #47392 patch is not applicable")
        return 0

    paths_and_patchers = (
        (source_root / FLASHINFER_UTIL_REL, patch_flashinfer_util),
        (expert_path, patch_expert),
        (source_root / ORACLE_REL, patch_oracle),
    )
    updates: list[tuple[Path, str, str]] = []
    for path, patcher in paths_and_patchers:
        if not path.exists():
            raise PatchError(f"{path} is missing while {EXPERT_REL} exists")
        original = path.read_text()
        updated = patcher(original)
        compile(updated, str(path), "exec")
        updates.append((path, original, updated))

    changed = []
    for path, original, updated in updates:
        if updated != original:
            path.write_text(updated)
            changed.append(str(path.relative_to(source_root)))

    if changed:
        print("Applied targeted vLLM PR #47392 runtime patch: " + ", ".join(changed))
    else:
        print("Targeted vLLM PR #47392 runtime behavior is already present")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PatchError as exc:
        raise SystemExit(f"targeted vLLM PR #47392 patch failed: {exc}") from exc
