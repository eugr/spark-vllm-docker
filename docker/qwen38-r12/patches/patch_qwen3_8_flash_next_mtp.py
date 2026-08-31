#!/usr/bin/env python3
"""Patch NVIDIA's Qwen3.8-Flash-Next MTP class with a packed draft head."""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "QWEN_DRAFT_HEAD_MXFP4_V1"


def patch(path: Path) -> None:
    source = path.read_text()
    if MARKER in source:
        return

    import_anchor = "from vllm.sequence import IntermediateTensors\n"
    helper_import = (
        "from vllm.model_executor.models.qwen_draft_head_mxfp4 import (\n"
        "    PackedMxfp4DraftHead,\n"
        "    draft_head_mxfp4_artifact,\n"
        "    draft_head_mxfp4_enabled,\n"
        ")  # QWEN_DRAFT_HEAD_MXFP4_V1\n"
    )
    if import_anchor not in source:
        raise ValueError("MTP import anchor not found")
    source = source.replace(import_anchor, helper_import + import_anchor, 1)

    config_anchor = (
        "        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config\n"
        "        self.vllm_config = vllm_config\n"
    )
    config_replacement = (
        "        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config\n"
        "        self.vllm_config = vllm_config\n"
        "        self._draft_head_mxfp4_enabled = draft_head_mxfp4_enabled()\n"
        "        self.has_own_lm_head = self._draft_head_mxfp4_enabled\n"
        "        if (\n"
        "            self._draft_head_mxfp4_enabled\n"
        "            and vllm_config.parallel_config.tensor_parallel_size != 1\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"packed MXFP4 draft head currently requires tensor_parallel_size=1\"\n"
        "            )\n"
    )
    if config_anchor not in source:
        raise ValueError("MTP config anchor not found")
    source = source.replace(config_anchor, config_replacement, 1)

    head_anchor = (
        "        if get_pp_group().is_last_rank:\n"
        "            if config.tie_word_embeddings:\n"
        "                self.lm_head = self.model.embed_tokens\n"
        "            else:\n"
        "                self.lm_head = ParallelLMHead(\n"
        "                    config.vocab_size,\n"
        "                    config.hidden_size,\n"
        "                    prefix=maybe_prefix(prefix, \"lm_head\"),\n"
        "                )\n"
        "        else:\n"
        "            self.lm_head = PPMissingLayer()\n"
    )
    head_replacement = (
        "        if get_pp_group().is_last_rank:\n"
        "            if self._draft_head_mxfp4_enabled:\n"
        "                self.lm_head = PackedMxfp4DraftHead(\n"
        "                    draft_head_mxfp4_artifact(),\n"
        "                    vocab_size=config.vocab_size,\n"
        "                    hidden_size=config.hidden_size,\n"
        "                )\n"
        "            elif config.tie_word_embeddings:\n"
        "                self.lm_head = self.model.embed_tokens\n"
        "            else:\n"
        "                self.lm_head = ParallelLMHead(\n"
        "                    config.vocab_size,\n"
        "                    config.hidden_size,\n"
        "                    prefix=maybe_prefix(prefix, \"lm_head\"),\n"
        "                )\n"
        "        else:\n"
        "            self.lm_head = PPMissingLayer()\n"
    )
    if head_anchor not in source:
        raise ValueError("MTP head anchor not found")
    source = source.replace(head_anchor, head_replacement, 1)

    logits_anchor = (
        "    def compute_logits(\n"
        "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
        "    ) -> torch.Tensor | None:\n"
        "        return self.logits_processor(self.lm_head, hidden_states)\n"
    )
    logits_replacement = (
        "    def compute_logits(\n"
        "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
        "    ) -> torch.Tensor | None:\n"
        "        if self._draft_head_mxfp4_enabled:\n"
        "            return self.lm_head(hidden_states)\n"
        "        return self.logits_processor(self.lm_head, hidden_states)\n"
    )
    if logits_anchor not in source:
        raise ValueError("MTP logits anchor not found")
    source = source.replace(logits_anchor, logits_replacement, 1)

    remap_anchor = (
        "                remapped_name = _remap_mtp_weight_name(name)\n"
        "                if remapped_name is not None:\n"
        "                    yield remapped_name, weight\n"
    )
    remap_replacement = (
        "                remapped_name = _remap_mtp_weight_name(name)\n"
        "                if (\n"
        "                    self._draft_head_mxfp4_enabled\n"
        "                    and remapped_name == \"lm_head.weight\"\n"
        "                ):\n"
        "                    continue\n"
        "                if remapped_name is not None:\n"
        "                    yield remapped_name, weight\n"
    )
    if remap_anchor not in source:
        raise ValueError("MTP weight-remap anchor not found")
    source = source.replace(remap_anchor, remap_replacement, 1)
    path.write_text(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    patch(args.path)


if __name__ == "__main__":
    main()
