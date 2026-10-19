# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from 2339bd39d661965b821bfbe652db8f536e0b3dcd:
# vllm/v1/worker/gpu/model_runner.py

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.compilation.wrapper import compile_model_with_stock_torch
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.distributed.aux_output_connector.worker import (
    AuxOutputWorkerConnector,
    get_aux_output_connector,
)
from vllm.distributed.parallel_state import get_dcp_group, get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    initialize_mamba_ssu_backend,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.reload.inplace import reload_weights
from vllm.model_executor.models.interfaces import requires_raw_input_tokens
from vllm.model_executor.offloader import (
    create_offloader,
    get_offloader,
    set_offloader,
)

class GPUModelRunner:
    def get_draft_model(self) -> nn.Module | None:
        speculator = self.speculator
        if not isinstance(speculator, DraftModelSpeculator):
            return None
        return speculator.model

    def reload_weights(
        self,
        weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
        weights_path: str | None = None,
        is_checkpoint_format: bool = True,
    ) -> None:
        """Reload weights in place."""
        vllm_config = self.vllm_config
        model = self.get_model()
        reload_weights(
            vllm_config, model, weights_iterator, weights_path, is_checkpoint_format
        )
        self.reset_lora_state()
        self.reset_encoder_cache()
        self.reset_mm_cache()

