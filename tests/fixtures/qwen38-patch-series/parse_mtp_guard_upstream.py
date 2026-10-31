# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Model Runner V1 guard excerpt at fe5fe8ba5fb9412ac6deafd192ae8b7c466b96c5.

class VllmConfig:
    def unsupported_features(self):
        unsupported = []
        # DSpark is implemented only by the V2 GPU model runner.
        if self.speculative_config:
            if self.speculative_config.method == "dspark":
                unsupported.append("dspark speculative decoding")
            elif self.speculative_config.is_xpress():
                unsupported.append("xpress speculative decoding")
            if self.speculative_config.enable_adaptive_verification:
                unsupported.append("adaptive draft verification")
            if self.speculative_config.draft_lm_head_quantization is not None:
                unsupported.append("draft lm_head quantization")

        # The attention-free Parse MTP draft requires the V2 speculator.
        if self.speculative_config:
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and "ParseAuxMTPModel" in (
                draft_config.architectures or []
            ):
                unsupported.append("Parse 3.5 MTP")

        return unsupported
