#!/usr/bin/env python3
"""Patch pinned ImageMediaIO; helper runs inside load_bytes before return.

Main integration installs modules/image_resize.py as qwen38_image_resize.py,
runs this patch, and injects the opt-in image media flag on the Qwen route.
"""

import argparse
import ast
from pathlib import Path

MARKER = "QWEN38_EARLY_NATIVE_IMAGE_RESIZE_V1"
OLD_INIT = '    def __init__(self, image_mode: str | None = "RGB", **kwargs) -> None:\n'
NEW_INIT = '''    def __init__(
        self, image_mode: str | None = "RGB", *,
        qwen38_early_resize: bool = False, **kwargs,
    ) -> None:
'''
OLD_CONFIG = '        self.rgba_background_color = rgba_bg\n'
NEW_CONFIG = OLD_CONFIG + '''
        # QWEN38_EARLY_NATIVE_IMAGE_RESIZE_V1
        if type(qwen38_early_resize) is not bool:
            raise ValueError("qwen38_early_resize must be a boolean")
        if qwen38_early_resize and (
            image_mode != "RGB" or rgba_bg != (255, 255, 255)
        ):
            raise ValueError("Qwen early resize requires RGB and white background")
        self.qwen38_early_resize = qwen38_early_resize
'''
OLD_RETURN = '        return MediaWithBytes(converted, data, io_config)\n'
NEW_RETURN = '''        if self.qwen38_early_resize:
            source = converted
            try:
                from qwen38_image_resize import early_resize

                converted, io_config = early_resize(source, io_config)
            finally:
                source.close()
                if image is not source:
                    image.close()
        return MediaWithBytes(converted, data, io_config)
'''


def patch_source(source):
    replacements = ((OLD_INIT, NEW_INIT), (OLD_CONFIG, NEW_CONFIG),
                    (OLD_RETURN, NEW_RETURN))
    if MARKER in source:
        if all(source.count(new) == 1 for _, new in replacements):
            ast.parse(source)
            return source
        raise ValueError("Partial or changed Qwen early-resize patch; refusing")
    for old, _ in replacements:
        if source.count(old) != 1:
            raise ValueError("Unexpected pinned ImageMediaIO source; refusing")
    for old, new in replacements:
        source = source.replace(old, new, 1)
    ast.parse(source)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path(
        "/usr/local/lib/python3.12/dist-packages/vllm/multimodal/media/image.py"))
    args = parser.parse_args()
    before = args.path.read_text()
    after = patch_source(before)
    if after != before:
        args.path.write_text(after)


if __name__ == "__main__":
    main()
