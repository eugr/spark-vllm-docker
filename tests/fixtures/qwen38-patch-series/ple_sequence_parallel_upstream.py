# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpts from fe341a53dda00472c281a2d1f4bd34a74e7eae6e:
# vllm/models/qwen4_exp/nvidia/ngram_embedding.py

class Qwen4ExpNGramEmbedding:
    def configure_embedding(
        self, params_dtype, padded_vocab_size, divisor, embedding_prefix,
        embedding_quant_method, max_total_tokens, data_parallel_rank,
    ):
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        engram_config = get_current_vllm_config().engram_config
        embedding_cls = (
            Qwen4ExpPLEPinnedHostEmbedding
            if engram_config is not None and engram_config.cpu_offload
            else Qwen4ExpPLEDeviceEmbedding
        )
        # A TP reduce-scatter covers all vocabulary owners only when ETP=TP.
        self.use_reduce_scatter = (
            self.use_sequence_parallel
            and get_etp_group().world_size == get_tp_group().world_size
        )
        self.ngram_embedding = embedding_cls(
            padded_vocab_size,
            self.head_dim,
            params_dtype=params_dtype,
            padding_size=divisor,
            prefix=embedding_prefix,
            embedding_method=embedding_quant_method,
            num_ngram_heads=self.ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
            reduce_results=not self.use_reduce_scatter,
        )

    def _sp_shard_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Shard embedding token rows, optionally reducing vocabulary contributions."""
        # Each embedding element has one vocabulary owner, including FP8 bytes.
        output_dtype = embeddings.dtype
        if is_fp8(embeddings):
            embeddings = embeddings.view(torch.int8)
        if self.use_reduce_scatter:
            embeddings = sp_reduce_scatter(embeddings)
        else:
            embeddings = sp_shard(embeddings)
        return embeddings.view(output_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return full embeddings or a token shard after combining vocabulary owners."""
        embedding = self.ngram_embedding
        if embedding.supports_prefetch:
            # Prefetch contains full tokens even when hidden states are sharded.
            embeddings = embedding.finalize_prefetch(total_tokens=input_ids.shape[0])
        else:
            ngram_ids = self.compute_ngram_ids(
                input_ids, query_start_loc, ngram_context
            )
            embeddings = embedding(ngram_ids).flatten(-2)
        if not self.use_sequence_parallel:
            return embeddings
        return self._sp_shard_embeddings(embeddings)
