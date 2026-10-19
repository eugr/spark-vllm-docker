# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream lookup methods after #56273 merged, from
# 3f466c852cfd3882820af8dd7963ac775174bce4:
# vllm/models/qwen4_exp/common/ngram_embedding.py


class Qwen4ExpPLENvFp4EmbeddingMethod:
    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        ids = input_.reshape(-1)
        if not input_.is_cuda:
            rows = dequantize_to_dtype(
                F.embedding(ids, layer.weight),
                F.embedding(ids, layer.weight_scale),
                layer.weight_scale_2,
                layer.params_dtype,
                swizzle=False,
            )
            return rows.reshape(*input_.shape, layer.embedding_dim)
        output = torch.empty(
            (*input_.shape, layer.embedding_dim),
            dtype=layer.params_dtype,
            device=input_.device,
        )
        if ids.numel():
            # Device rows are already local, so every ID in range is owned.
            _lookup_nvfp4_ple_embedding_kernel[(ids.numel(),)](
                layer.weight,
                layer.weight_scale.view(torch.uint8),
                layer.weight_scale_2,
                ids,
                output,
                layer.embedding_dim,
                0,
                layer.weight.shape[0],
                BLOCK_D=triton.next_power_of_2(layer.embedding_dim),
            )
        return output

    def lookup_from_pinned(
        self,
        layer: "Qwen4ExpPLEPinnedHostEmbedding",
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        _lookup_nvfp4_ple_embedding_kernel[(ids.numel(),)](
            layer._uva_weight,
            self._uva_weight_scale.view(torch.uint8),
            layer.weight_scale_2,
            ids,
            output,
            layer.embedding_dim,
            layer.shard_indices.org_vocab_start_index,
            layer.shard_indices.org_vocab_end_index,
            BLOCK_D=triton.next_power_of_2(layer.embedding_dim),
        )

    def dequantize(
        self,
        layer: nn.Module,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del layer
        # The lookup already decoded the rows to the activation dtype.
        return embeddings.to(output_dtype)
