#!/usr/bin/env python3
"""Route unsupported DeepSeek-V4 expert counts through vLLM's Torch router."""

from pathlib import Path
import py_compile


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/router/fused_topk_bias_router.py"
)
SUPPORTED_EXPERT_COUNTS = "(16, 32, 64, 128, 192, 256, 320, 384, 512)"

OLD = """    if current_platform.is_xpu():
        return _topk_softplus_sqrt_torch("""

NEW = f"""    # The CUDA extension is instantiated only for fixed expert counts.
    # REAP K160 is valid but nonstandard, so use vLLM's equivalent Torch path.
    if (
        current_platform.is_xpu()
        or gating_output.shape[-1] not in {SUPPORTED_EXPERT_COUNTS}
    ):
        return _topk_softplus_sqrt_torch("""


def main() -> None:
    text = TARGET.read_text()
    if NEW not in text:
        if text.count(OLD) != 1:
            raise SystemExit(f"PATCH_TARGET_MISMATCH count={{text.count(OLD)}}")
        TARGET.write_text(text.replace(OLD, NEW, 1))
    py_compile.compile(str(TARGET), doraise=True)
    print("K160_ROUTER_PATCH_OK", flush=True)


if __name__ == "__main__":
    main()
