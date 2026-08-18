#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCHER_PATH = PROJECT_DIR / "docker/patch_vllm_flashinfer_b12x_swigluoai.py"
SPEC = importlib.util.spec_from_file_location("b12x_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


FLASHINFER_UTIL = '''
import functools
import importlib.util


def has_flashinfer_b12x_moe() -> bool:
    return True


@functools.cache
def has_nvidia_artifactory() -> bool:
    return True


__all__ = [
    "has_flashinfer_b12x_moe",
]
'''

LEGACY_EXPERT = '''
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    has_flashinfer_b12x_moe,
)


class FlashInferB12xExperts:
    _ACTIVATION_MAP = {
        MoEActivation.SILU: "silu",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    def __init__(self, moe_config, quant_config):
        activation = moe_config.activation
        self._activation_str = self._ACTIVATION_MAP[activation]

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)

    def _ensure_wrapper(self) -> None:
        self._wrapper = B12xMoEWrapper(
            activation=self._activation_str,
        )
'''

CURRENT_GELU_EXPERT = LEGACY_EXPERT.replace(
    '        MoEActivation.SILU: "silu",\n',
    '        MoEActivation.SILU: "silu",\n'
    '        MoEActivation.GELU_TANH: "gelu_tanh",\n',
).replace(
    "        return activation in (MoEActivation.SILU, MoEActivation.RELU2_NO_MUL)\n",
    "        return activation in (\n"
    "            MoEActivation.SILU,\n"
    "            MoEActivation.GELU_TANH,\n"
    "            MoEActivation.RELU2_NO_MUL,\n"
    "        )\n",
)

ORACLE = '''
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)

logger = init_logger(__name__)


def select_nvfp4_moe_backend(config):
    NVFP4_BACKENDS_WITH_CLAMP = {
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
    }

    def _backend_supports_clamp(backend):
        if backend in NVFP4_BACKENDS_WITH_CLAMP:
            return True
        return backend == NvFp4MoeBackend.B12X

    return _backend_supports_clamp
'''


class TargetedB12xPatchTests(unittest.TestCase):
    def test_legacy_runtime_sources_are_patched_idempotently(self):
        util = PATCHER.patch_flashinfer_util(FLASHINFER_UTIL)
        expert = PATCHER.patch_expert(LEGACY_EXPERT)
        oracle = PATCHER.patch_oracle(ORACLE)

        self.assertIn("def has_flashinfer_b12x_moe_activation()", util)
        self.assertIn("inspect.signature(mod.B12xMoEWrapper)", util)
        self.assertIn("SWIGLUOAI_UNINTERLEAVE", expert)
        self.assertIn("**swiglu_kwargs", expert)
        self.assertIn("NVFP4_BACKENDS_WITH_CLAMP.add", oracle)
        self.assertIn("return backend == NvFp4MoeBackend.B12X", oracle)

        self.assertEqual(PATCHER.patch_flashinfer_util(util), util)
        self.assertEqual(PATCHER.patch_expert(expert), expert)
        self.assertEqual(PATCHER.patch_oracle(oracle), oracle)

    def test_current_gelu_expert_is_patched_without_losing_gelu(self):
        expert = PATCHER.patch_expert(CURRENT_GELU_EXPERT)

        self.assertIn('MoEActivation.GELU_TANH: "gelu_tanh"', expert)
        self.assertIn(
            "        if activation in (\n"
            "            MoEActivation.SILU,\n"
            "            MoEActivation.GELU_TANH,\n"
            "            MoEActivation.RELU2_NO_MUL,\n"
            "        ):\n",
            expert,
        )
        self.assertIn("and has_flashinfer_b12x_moe_activation()", expert)
        self.assertEqual(PATCHER.patch_expert(expert), expert)

    def test_unknown_expert_source_shape_fails(self):
        with self.assertRaises(PATCHER.PatchError):
            PATCHER.patch_expert("class FlashInferB12xExperts: pass\n")


if __name__ == "__main__":
    unittest.main()
