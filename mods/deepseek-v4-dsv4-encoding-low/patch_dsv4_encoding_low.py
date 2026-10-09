#!/usr/bin/env python3
"""Install REAP's own DeepSeek-V4 encoder and enable reasoning_effort "low".

Ports the DSPARK_ENCODING_FILE branch of MiaAI-Lab's docker-compose prelude
into the recipe mod system:

  1. Copy the checkpoint's encoding_dsv4.py over vLLM's bundled
     tokenizers/deepseek_v4_encoding.py.
  2. Patch tokenizers/deepseek_v4.py so an unrecognised reasoning_effort maps
     to "low" instead of the stock "high" (leaving off/high/max/xhigh intact).

Both steps are idempotent. The encoding file is auto-discovered in the HF cache
so this works regardless of the container's cache mount path.
"""

from pathlib import Path
import glob
import py_compile
import shutil

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm/tokenizers")
ENCODING_DST = DIST / "deepseek_v4_encoding.py"
DSV4 = DIST / "deepseek_v4.py"

# Snapshot layout: .../models--0xSero--DeepSeek-V4-Flash-0731-REAP/snapshots/<rev>/encoding/encoding_dsv4.py
ENCODING_GLOBS = [
    "/root/.cache/huggingface/hub/models--0xSero--DeepSeek-V4-Flash-0731-REAP/snapshots/*/encoding/encoding_dsv4.py",
    "/cache/huggingface/hub/models--0xSero--DeepSeek-V4-Flash-0731-REAP/snapshots/*/encoding/encoding_dsv4.py",
]

OLD = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"'''

NEW = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            elif reasoning_effort == "high":
                reasoning_effort = "high"
            else:
                reasoning_effort = "low"'''


def install_encoding() -> None:
    for pattern in ENCODING_GLOBS:
        matches = sorted(glob.glob(pattern))
        if matches:
            src = matches[-1]
            shutil.copyfile(src, ENCODING_DST)
            print(f"DSV4_ENCODING_INSTALLED {src}", flush=True)
            return
    print("DSV4_ENCODING_SKIP encoding_dsv4.py not found in HF cache", flush=True)


def patch_reasoning_low() -> None:
    text = DSV4.read_text()
    if NEW in text:
        print("DSV4_LOW_PATCH_OK already applied", flush=True)
    elif OLD in text:
        DSV4.write_text(text.replace(OLD, NEW, 1))
        print("DSV4_LOW_PATCH_OK", flush=True)
    else:
        print("DSV4_LOW_PATCH_SKIP mapping block not found (image changed?)", flush=True)
    py_compile.compile(str(DSV4), doraise=True)


def main() -> None:
    install_encoding()
    patch_reasoning_low()


if __name__ == "__main__":
    main()
