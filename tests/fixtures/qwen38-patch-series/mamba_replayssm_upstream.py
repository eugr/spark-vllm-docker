# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpts from 46f36fa294559d080edc61a09e5b57826218a2bf:
# vllm/v1/worker/gpu/model_states/mamba_hybrid.py

class MambaHybridModelState(DefaultModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self._replayssm_query_start_loc: torch.Tensor | None = None
        # Pre-copy prefix-cache state (V2). The migration of each request's
        # mamba state across block boundaries runs as a fused GPU kernel reusing
        # the postprocess copy machinery, so the per-step src columns and the
        # running state_idx are kept GPU-resident.
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        self._use_flashinfer_replayssm = vllm_config.use_flashinfer_replayssm
        if self._use_flashinfer_replayssm:
            self._is_prefilling_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.bool, device=self.device
            )
        self.recoverssm = (
            RecoverSSMState() if self.cache_config.use_kda_recoverssm else None
        )
        if self._align_mode or self._use_flashinfer_replayssm:
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
            self._mamba_state_copy_funcs: MambaStateCopyFuncsByType | None = None
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_src_off_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
