# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpts from upstream b409a4f0d787fdddb75ec2e8ddb6dfcfd36235b7 (with preceding series entries).
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
        self._use_replayssm_trackers = vllm_config.use_replayssm_trackers
        if self._use_replayssm_trackers:
            self._is_prefilling_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.bool, device=self.device
            )
        self.recoverssm = (
            RecoverSSMState() if self.cache_config.use_kda_recoverssm else None
        )
        if self._align_mode or self._use_replayssm_trackers:
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
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
