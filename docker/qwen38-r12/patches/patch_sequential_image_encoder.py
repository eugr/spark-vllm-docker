#!/usr/bin/env python3
"""Build-bundle patch for the exact pinned EncoderRunner; disabled path unchanged.

Main installer must separately install modules/image_encoder.py as
/usr/local/lib/python3.12/dist-packages/qwen38_image_encoder.py, run this script,
and explicitly enable QWEN38_SEQUENTIAL_IMAGE_ENCODER=1. No installation here.
"""

import argparse
import hashlib
from pathlib import Path


MARKER = "QWEN38_SEQUENTIAL_IMAGE_ENCODER_V1"
PINNED_SHA256 = "7e61056d1ee942b9ea7f7e5207050362581084fec174e74676c54534e6c5f07b"
DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/mm/encoder_runner.py"
)
OLD_METHOD = '''    @torch.inference_mode()
    def execute_mm_encoder(
        self, mm_kwargs: list[tuple[str, MultiModalKwargsItem]]
    ) -> list[torch.Tensor]:
        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs, device=self.device, pin_memory=PIN_MEMORY
        ):
            cg_manager = self.cudagraph_manager
            cudagraph_output = (
                cg_manager.execute(mm_kwargs_batch)
                if cg_manager is not None
                and cg_manager.is_captured()
                and cg_manager.supports_modality(modality)
                else None
            )
            batch_outputs = (
                cudagraph_output
                if cudagraph_output is not None
                else self.model.embed_multimodal(**mm_kwargs_batch)
            )
            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)
        return encoder_outputs
'''
INSERTION = f'''        # {MARKER}: split original CPU items before native transfer.
        from qwen38_image_encoder import execute_mm_encoder, sequential_enabled

        if sequential_enabled():
            return execute_mm_encoder(
                self, mm_kwargs,
                group_and_batch_mm_kwargs=group_and_batch_mm_kwargs,
                sanity_check_mm_encoder_outputs=sanity_check_mm_encoder_outputs,
                pin_memory=PIN_MEMORY,
            )

'''
NEW_METHOD = OLD_METHOD.replace(
    "        encoder_outputs: list[torch.Tensor] = []\n",
    INSERTION + "        encoder_outputs: list[torch.Tensor] = []\n", 1,
)


def patch_source(source):
    if MARKER in source:
        if source.count(MARKER) != 1 or source.count(NEW_METHOD) != 1:
            raise ValueError("Partial or modified sequential image encoder patch")
        original = source.replace(NEW_METHOD, OLD_METHOD, 1)
        if hashlib.sha256(original.encode()).hexdigest() != PINNED_SHA256:
            raise ValueError("Patched EncoderRunner differs from the exact pinned source")
        return source
    if hashlib.sha256(source.encode()).hexdigest() != PINNED_SHA256:
        raise ValueError("EncoderRunner SHA256 differs from the exact pinned source; refusing to patch")
    if source.count(OLD_METHOD) != 1:
        raise ValueError("Expected exact pinned execute_mm_encoder method once")
    result = source.replace(OLD_METHOD, NEW_METHOD, 1)
    compile(result, "encoder_runner.py", "exec")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args()
    source = args.target.read_bytes().decode("utf-8")
    result = patch_source(source)
    if not args.check and result != source:
        # Refuse an edit that raced our read. No unrelated source is replaced.
        if args.target.read_bytes().decode("utf-8") != source:
            raise RuntimeError("EncoderRunner changed while preparing patch")
        args.target.write_bytes(result.encode("utf-8"))
    print("validated" if args.check else "patched/already applied", args.target)


if __name__ == "__main__":
    main()
