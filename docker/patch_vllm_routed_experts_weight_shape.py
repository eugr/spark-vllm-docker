#!/usr/bin/env python3
"""Preserve vector weight_shape metadata in RoutedExperts loading."""

import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "vllm/model_executor/layers/fused_moe/routed_experts.py"
old = '''    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = self._to_scalar(loaded_weight)
'''
new = '''    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data
        target = param_data[expert_id]

        if target.ndim > 0 and target.numel() == loaded_weight.numel():
            target.copy_(loaded_weight.reshape_as(target).to(
                device=target.device, dtype=target.dtype))
            return

        # Scalar input scales can be loaded directly and should be equal.
        param_data[expert_id] = self._to_scalar(loaded_weight)
'''

if not target.exists():
    print(f"{target} not found; skipping RoutedExperts weight_shape workaround")
else:
    text = target.read_text()
    if "target = param_data[expert_id]" in text:
        print("RoutedExperts weight_shape workaround already present; skipping")
    elif old in text:
        target.write_text(text.replace(old, new, 1))
        print("Applied RoutedExperts weight_shape workaround")
    else:
        print(
            "Known vulnerable RoutedExperts _load_single_value pattern "
            "not found; skipping"
        )
