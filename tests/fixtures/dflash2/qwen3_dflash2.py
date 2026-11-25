from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(
            self.lm_head.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            raise ValueError(
                "DFlash2 requires an unquantized target LM head for candidate TopK; "
                f"got {type(self.lm_head.quant_method).__name__}."
            )

        selector = self.model.candidate_selector
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        return selector, logits
