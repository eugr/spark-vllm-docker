# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpts from fe341a53dda00472c281a2d1f4bd34a74e7eae6e.
# common/ngram_embedding.py: PLE abstract interface (constructor omitted).

class Qwen4ExpPLEEmbedding(PLEVocabParallelEmbedding, ABC):
    supports_prefetch: ClassVar[bool] = False

    @abstractmethod
    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate storage for the complete embedding weight."""
        raise NotImplementedError

    @abstractmethod
    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        """Start an asynchronous lookup when supported."""
        raise NotImplementedError

    @abstractmethod
    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        """Return prefetched embeddings for the full input token set."""
        raise NotImplementedError


# nvidia/ngram_embedding.py: insertion point for the file-gather class.
__all__ = [
    "Qwen4ExpPLEDeviceEmbedding",
    "Qwen4ExpPLEEmbedding",
    "Qwen4ExpPLEEmbeddingMethod",
    "Qwen4ExpPLEFp8EmbeddingMethod",
    "Qwen4ExpPLEPinnedHostEmbedding",
    "Qwen4ExpPLEUnquantizedEmbeddingMethod",
    "Qwen4ExpNGramEmbedding",
]


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
