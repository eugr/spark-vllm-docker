"""Adapt the pinned vLLM build to the Inco AI Inkling 2 Mini drafter."""

from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
import sys


PATCHES = {
    "v1/worker/gpu/spec_decode/eagle/eagle3_utils.py": (
        (
            "            # Add 1 to convert DFlash's aux layer id semantics\n",
            """            target = spec_config.target_model_config
            if (
                spec_config.method == "dflash"
                and target is not None
                and target.hf_text_config.model_type == "inkling2_loop_text"
            ):
                # Inkling captures raw loop visits and appends the final norm.
                layer_ids = tuple(dflash_config.get("target_layer_ids") or ())
                if layer_ids != (5, 23, 41, 48):
                    raise ValueError(
                        "Inkling 2 Mini DFlash2 requires capture visits [5, 23, 41, 48]"
                    )
                return layer_ids
            # Add 1 to convert DFlash's aux layer id semantics
""",
        ),
    ),
    "model_executor/models/qwen3_dflash2.py": (
        (
            '        draft_config = self.config.dflash_config\n        self.input_embedding_scale = float(',
            '        draft_config = self.config.dflash_config\n'
            '        self.input_embedding_norm: nn.Module | None = None\n'
            '        self.input_embedding_scale = float(',
        ),
        (
            "        return super().embed_input_ids(input_ids) * self.input_embedding_scale",
            """        embeds = super().embed_input_ids(input_ids)
        if self.input_embedding_norm is not None:
            embeds = self.input_embedding_norm(embeds)
        return embeds * self.input_embedding_scale""",
        ),
        (
            "        self.top_k = top_k\n        self.predecessor_codebook",
            "        self.top_k = top_k\n        self.post_softcap: float | None = None\n"
            "        self.predecessor_codebook",
        ),
        (
            "        hidden = self.hidden_projection(hidden_states)\n        return _score_edges(",
            "        hidden = self.hidden_projection(hidden_states)\n        scores = _score_edges(",
        ),
        (
            "            self.top_k,\n        )\n\n\nclass DFlash2Qwen3Model",
            """            self.top_k,
        )
        if self.post_softcap is not None:
            scores = self.post_softcap * torch.tanh(scores / self.post_softcap)
        return scores


class DFlash2Qwen3Model""",
        ),
        (
            """        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        # Candidates index the selector codebooks, which have the draft's
        # vocabulary; the target's vocabulary can be larger.
        self.candidate_logits_processor = LogitsProcessor(
            min(vllm_config.model_config.get_vocab_size(), self.config.vocab_size),""",
            """        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        spec_config = vllm_config.speculative_config
        assert spec_config is not None and spec_config.target_model_config is not None
        target_text = spec_config.target_model_config.hf_text_config
        vocab_size = vllm_config.model_config.get_vocab_size()
        if target_text.model_type == "inkling2_loop_text":
            # Inkling stores padded LM-head rows that cannot be sampled.
            vocab_size = getattr(target_text, "unpadded_vocab_size", None) or vocab_size
        if softcap > 0 and draft_config.get("selector_softcap_mode") == "after_correction":
            self.model.candidate_selector.post_softcap = softcap
            softcap = 0.0
        # Candidates index the selector codebooks, which have the draft's
        # vocabulary; the target's vocabulary can be larger.
        self.candidate_logits_processor = LogitsProcessor(
            min(vocab_size, self.config.vocab_size),""",
        ),
    ),
    "v1/worker/gpu/spec_decode/dflash/utils.py": (
        (
            "    maybe_share_target_embed(dflash_model, draft_inner, target_inner)\n",
            """    maybe_share_target_embed(dflash_model, draft_inner, target_inner)
    if (
        vllm_config.model_config.hf_text_config.model_type == "inkling2_loop_text"
        and hasattr(draft_inner, "input_embedding_norm")
    ):
        # The draft uses the target's normalized embedding for both the
        # anchor token and masked slots, as in the training runtime.
        if draft_inner.embed_tokens is not target_inner.embed_tokens:
            raise ValueError("Inkling 2 Mini DFlash2 requires shared input embeddings")
        draft_inner.input_embedding_norm = target_inner.embed_norm
""",
        ),
    ),
}


def apply(root: Path) -> None:
    staged = {}
    for relative, replacements in PATCHES.items():
        path = root / relative
        original = updated = path.read_text()
        for before, after in replacements:
            if updated.count(after) == 1:
                continue
            if updated.count(before) != 1:
                raise ValueError(f"Unexpected source in {relative}: {before.splitlines()[0]}")
            updated = updated.replace(before, after, 1)
        compile(updated, str(path), "exec")
        if original != updated:
            staged[path] = updated

    for path, updated in staged.items():
        temporary = path.with_suffix(".py.inkling2-dflash2-tmp")
        temporary.write_text(updated)
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)

    result = "Adapter applied." if staged else "Adapter already applied."
    print(f"[inkling2-dflash2] {result}", flush=True)


if __name__ == "__main__":
    try:
        installed = version("vllm")
        if installed != "0.1.dev25855+g7cb5dae22.cu130":
            raise ValueError(f"Unsupported vLLM build: {installed}")
        spec = find_spec("vllm")
        if spec is None or spec.origin is None:
            raise ValueError("Could not locate the vLLM package")
        apply(Path(spec.origin).parent)
    except (OSError, ValueError, SyntaxError) as error:
        sys.exit(f"[inkling2-dflash2] {error}")
