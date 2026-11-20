#!/usr/bin/env python3
"""Exclude the BF16 MTP layers in nvidia/GLM-5.3-Flash-NVFP4 from quantization.

The checkpoint stores its MTP layer in BF16 without weight scales but omits it
from quantization_config.ignore. vLLM consequently allocates packed NVFP4
experts and fails when loading BF16 weights that are twice as wide.

Safetensors headers inspected on 2026-09-22:
    Layer 45 (MTP): 864 BF16 expert tensors, no weight_scale tensors.
    Layer 44 (target): 864 U8 + 864 F8_E4M3 + 1728 F32 tensors.

In a 2x DGX Spark TP2 setup, the destination shape was (4096, 512), while
BF16 source weights went from (4096, 2048) to (4096, 1024) after sharding.
Sharding was correct; the mismatch came from packed versus unpacked weights.

MTP indices are derived from the model configuration:
    range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)
For this checkpoint, range(45, 46) yields layer 45. The helper checks for nextn
layers rather than model_type, which may differ in the drafter configuration.

Set moe_backend to auto in speculative-config: Marlin cannot run the
unquantized drafter MoE. The patch fails if its source anchor does not match.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

MARKER = "GLM53-NVIDIA-MTP-UNQUANT"
REL = "model_executor/layers/quantization/modelopt.py"


class PatchError(RuntimeError):
    pass


ANCHOR = '''    def is_layer_excluded(self, prefix: str) -> bool:
        """
        Check if a layer should be excluded from quantization.

        Handles both exact matching (for fused layers) and ModelOpt wildcard matching.

        The ModelOpt exclude_modules list is a list of wildcards.
        """
        if len(self.exclude_modules) == 0:
            return False
'''

REPLACEMENT = f'''    def _{MARKER.lower().replace("-", "_")}_mtp_layers(self):
        """Indices of a glm5_next model's MTP layers, derived from the config.

        NVIDIA's checkpoint ships them in BF16 without scales but does not list
        them in its `ignore`. Without this, the drafter is built as NVFP4 and
        loading its experts fails with a packing mismatch (512 vs 1024).
        """
        cached = getattr(self, "_glm53_mtp_layer_cache", None)
        if cached is not None:
            return cached
        idx = ()
        reason = "?"
        try:
            from vllm.config import get_current_vllm_config_or_none

            vc = get_current_vllm_config_or_none()
            hf = getattr(getattr(vc, "model_config", None), "hf_config", None)
            if vc is None:
                reason = "no current VllmConfig"
            elif hf is None:
                reason = "no hf_config"
            else:
                txt = getattr(hf, "text_config", hf)
                n = int(getattr(txt, "num_hidden_layers", 0) or 0)
                k = int(getattr(txt, "num_nextn_predict_layers", 0) or 0)
                # Check for nextn layers rather than the model name:
                # the drafter hf_config may not declare the target model_type.
                if n > 0 and k > 0:
                    idx = tuple(range(n, n + k))
                    reason = "derived"
                else:
                    reason = (
                        f"no nextn layers (model_type={{getattr(hf, 'model_type', '?')}} "
                        f"num_hidden_layers={{n}} num_nextn_predict_layers={{k}})"
                    )
        except Exception as exc:  # Never fail startup because of this helper.
            reason = f"exception: {{exc!r}}"
            idx = ()
        # Always log the result, including when no MTP layers are found.
        # info_once deduplicates by arguments, so they must be hashable.
        # Convert the list to a string to avoid failing startup.
        logger.info_once(
            "[{MARKER}] MTP layers treated as NOT quantized: %s (%s)",
            str(list(idx)),
            str(reason),
        )
        self._glm53_mtp_layer_cache = idx
        return idx

    def is_layer_excluded(self, prefix: str) -> bool:
        """
        Check if a layer should be excluded from quantization.

        Handles both exact matching (for fused layers) and ModelOpt wildcard matching.

        The ModelOpt exclude_modules list is a list of wildcards.
        """
        # {MARKER}: check BEFORE the empty exclude_modules early return.
        # The checkpoint may omit its unquantized MTP layers from that list.
        for _mtp_layer in self._{MARKER.lower().replace("-", "_")}_mtp_layers():
            if f".layers.{{_mtp_layer}}." in prefix or prefix.endswith(
                f".layers.{{_mtp_layer}}"
            ):
                logger.debug_once(
                    "[{MARKER}] excluida de cuantizacion: %s", prefix
                )
                return True
        if len(self.exclude_modules) == 0:
            return False
'''


def patch(text: str) -> str:
    if MARKER in text:
        return text
    n = text.count(ANCHOR)
    if n != 1:
        raise PatchError(
            f"ANCHOR FAILED: expected exactly 1 occurrence of is_layer_excluded's "
            f"head, found {n}. Re-derive the anchor before building."
        )
    text = text.replace(ANCHOR, REPLACEMENT, 1)
    if "logger" not in text:
        raise PatchError("modelopt.py has no module logger")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = Path(args.vllm_root) / REL
    if not path.exists():
        raise PatchError(f"{REL} missing under {args.vllm_root}")
    original = path.read_text()
    if args.check:
        if MARKER in original:
            print(f"[{MARKER}] check: already patched")
            return 0
        patch(original)
        print(f"[{MARKER}] check: ready (anchor matches, nothing written)")
        return 0
    updated = patch(original)
    ast.parse(updated)
    if updated == original:
        print(f"[{MARKER}] already present (idempotent no-op)")
        return 0
    path.write_text(updated)
    print(f"[{MARKER}] applied: {REL}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PatchError as exc:
        print(f"[{MARKER}] FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
