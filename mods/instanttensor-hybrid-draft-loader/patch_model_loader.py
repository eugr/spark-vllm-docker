#!/usr/bin/env python3
"""Add an opt-in hybrid loader policy for speculative draft models."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "# spark-vllm mod: instanttensor-hybrid-draft-loader v1"

GET_MODEL_ANCHOR = """
def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    loader = get_model_loader(load_config or vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
"""

GET_MODEL_REPLACEMENT = f"""
{MARKER}
def _instanttensor_draft_load_config(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    load_config: LoadConfig | None,
) -> LoadConfig:
    \"\"\"Resolve the loader for one model without changing the target loader.\"\"\"
    import os

    from vllm.config import replace

    mode = os.environ.get("INSTANTTENSOR_DRAFT_LOADER", "auto").strip().lower()
    allowed_modes = ("auto", "safetensors", "instanttensor")
    if mode not in allowed_modes:
        raise ValueError(
            "INSTANTTENSOR_DRAFT_LOADER must be one of "
            f"{{', '.join(allowed_modes)}}; got {{mode!r}}"
        )

    effective = load_config or vllm_config.load_config
    load_format = getattr(effective.load_format, "value", effective.load_format)
    if mode == "instanttensor" or str(load_format).lower() != "instanttensor":
        return effective

    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None or model_config is not draft_model_config:
        return effective

    if mode == "auto":
        target_model_config = (
            getattr(speculative_config, "target_model_config", None)
            or vllm_config.model_config
        )
        draft_source = (
            getattr(draft_model_config, "model", None),
            getattr(draft_model_config, "revision", None),
        )
        target_source = (
            getattr(target_model_config, "model", None),
            getattr(target_model_config, "revision", None),
        )
        if draft_source != target_source:
            return effective

    logger.info_once(
        "Hybrid draft loading: using lazy safetensors for speculative draft "
        "weights while preserving InstantTensor for the target model "
        "(INSTANTTENSOR_DRAFT_LOADER=%s).",
        mode,
    )
    return replace(
        effective,
        load_format="safetensors",
        safetensors_load_strategy="lazy",
    )


def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    if model_config is None:
        model_config = vllm_config.model_config
    resolved_load_config = _instanttensor_draft_load_config(
        vllm_config, model_config, load_config
    )
    loader = get_model_loader(resolved_load_config)
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )
"""


def validate_shape(text: str, *, patched: bool) -> None:
    tree = ast.parse(text)
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = {"get_model", "get_model_loader"}
    if patched:
        required.add("_instanttensor_draft_load_config")
    missing = sorted(required - functions)
    if missing:
        raise ValueError(
            "unexpected model-loader module; missing " + ", ".join(missing)
        )


def patched_text(text: str) -> str:
    validate_shape(text, patched=MARKER in text)
    if MARKER in text:
        if text.count(MARKER) != 1:
            raise ValueError("hybrid-loader marker occurs more than once")
        if "resolved_load_config = _instanttensor_draft_load_config(" not in text:
            raise ValueError("mod marker exists but get_model does not use the helper")
        compile(text, "<patched model_loader/__init__.py>", "exec")
        return text

    count = text.count(GET_MODEL_ANCHOR)
    if count != 1:
        raise ValueError(
            f"expected exactly one supported get_model function; found {count}"
        )
    patched = text.replace(GET_MODEL_ANCHOR, GET_MODEL_REPLACEMENT, 1)
    validate_shape(patched, patched=True)
    compile(patched, "<patched model_loader/__init__.py>", "exec")
    return patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="validate compatibility without writing"
    )
    args = parser.parse_args()

    if not args.target.is_file():
        print(
            "[instanttensor-hybrid-draft-loader ERROR] "
            f"target not found: {args.target}",
            file=sys.stderr,
        )
        return 1

    original = args.target.read_text()
    try:
        patched = patched_text(original)
    except (SyntaxError, ValueError) as exc:
        print(
            "[instanttensor-hybrid-draft-loader ERROR] "
            f"refusing to patch {args.target}: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.check:
        state = "already patched" if patched == original else "compatible"
        print(f"[instanttensor-hybrid-draft-loader] {args.target} is {state}.")
        return 0

    if patched == original:
        print(
            "[instanttensor-hybrid-draft-loader] "
            "Model loader is already patched; skipping."
        )
        return 0

    temporary = args.target.with_suffix(
        args.target.suffix + ".instanttensor-draft-mod.tmp"
    )
    temporary.write_text(patched)
    temporary.replace(args.target)
    print(f"[instanttensor-hybrid-draft-loader] Patched {args.target}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
