# minimax_m2_optthink_reasoning_parser.py
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

logger = init_logger(__name__)

@ReasoningParserManager.register_module("minimax_m2_optthink")
class MiniMaxM2OptthinkReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for MiniMax M2 model that handles enable_thinking=False configs.

    MiniMax M2 models don't generate <think> start token, only </think> end
    token. All content before </think> is reasoning, content after is the
    actual response. This checks chat_template_kwargs for thinking config.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token not in model_output:
            if not self.thinking_enabled:
                # Thinking explicitly disabled \u2014 treat everything as content.
                return None, model_output
            # Thinking enabled but no </think>: output was truncated.
            return model_output, None

        # Extract reasoning content from the model output.
        reasoning, _, content = model_output.partition(self.end_token)
        final_content = content or None
        return reasoning, final_content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a delta message for streaming.
        """
        # Skip single special tokens
        if len(delta_token_ids) == 1 and (
            delta_token_ids[0] in [self.start_token_id, self.end_token_id]
        ):
            return None

        # Check if end token has already appeared in previous tokens
        if self.end_token_id in previous_token_ids:
            # We're past the reasoning phase, this is content
            return DeltaMessage(content=delta_text)

        # Check if end token is in delta tokens
        if self.end_token_id in delta_token_ids:
            # End token in delta, split reasoning and content
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning if reasoning else None,
                    content=content if content else None,
                )
            return None

        # No end token yet, all content is reasoning
        return DeltaMessage(reasoning=delta_text)
