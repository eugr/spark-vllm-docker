"""CPU tests: python3 -B tests/test_qwen38_image_resize.py -v

Native tests require the matching release image; no model/tokenizer downloads.
The loader source is patched in memory, never on disk. Run under the CPU lab's
4-GiB cap/120-second deadline. Missing native dependencies are explicit skips.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
from importlib import util
from io import BytesIO
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docker/qwen38-r12"


def load_module(name, path):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patcher = load_module("resize_patcher", BUNDLE / "patches/patch_early_image_resize.py")
helper = load_module("resize_helper", BUNDLE / "modules/image_resize.py")


class PatchTests(unittest.TestCase):
    def fixture(self):
        return ("class ImageMediaIO:\n" + patcher.OLD_INIT
                + "        rgba_bg = (255, 255, 255)\n" + patcher.OLD_CONFIG
                + "    def load_bytes(self, data):\n" + patcher.OLD_RETURN)

    def test_patch_is_parseable_and_idempotent(self):
        result = patcher.patch_source(self.fixture())
        ast.parse(result)
        self.assertEqual(patcher.patch_source(result), result)
        self.assertIn("qwen38_early_resize: bool = False", result)
        self.assertLess(result.index("early_resize(source"), result.index(patcher.OLD_RETURN))

    def test_unknown_duplicate_or_partial_source_refused(self):
        for source in ("", self.fixture() * 2,
                       self.fixture().replace(patcher.OLD_CONFIG, ""),
                       self.fixture() + "\n# " + patcher.MARKER):
            with self.subTest(source=source[:30]), self.assertRaises(ValueError):
                patcher.patch_source(source)


NATIVE = all(util.find_spec(name) is not None
             for name in ("torch", "torchvision", "transformers", "PIL", "vllm"))


@unittest.skipUnless(NATIVE, "requires matching release Torch/HF/Pillow/vLLM")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from PIL import Image
        from vllm.multimodal.media import image as native
        from vllm.multimodal.hasher import MultiModalHasher

        torch.set_num_threads(2)
        cls.torch, cls.Image, cls.hasher = torch, Image, MultiModalHasher
        ns = {"__name__": native.__name__, "__package__": native.__package__}
        exec(compile(patcher.patch_source(Path(native.__file__).read_text()),
                     "<patched ImageMediaIO>", "exec"), ns)
        cls.Loader = ns["ImageMediaIO"]
        cls.module_patch = patch.dict(sys.modules, {"qwen38_image_resize": helper})
        cls.module_patch.start()
        cls.addClassCleanup(cls.module_patch.stop)
        cls.kwargs = dict(min_pixels=65536, max_pixels=2097152,
                          size={"shortest_edge": 65536, "longest_edge": 2097152},
                          patch_size=16, merge_size=2, temporal_patch_size=2,
                          do_resize=True, return_tensors="pt", device="cpu",
                          disable_grouping=True)

    def encoded(self, mode="RGB", size=(997, 613), orientation=None):
        # Nonuniform deterministic synthetic image, including varying alpha.
        import numpy as np
        w, h = size
        x = np.arange(w, dtype=np.uint16)[None, :]
        y = np.arange(h, dtype=np.uint16)[:, None]
        pixels = np.empty((h, w, len(mode)), dtype=np.uint8)
        pixels[..., 0] = (x * 13 + y * 7) % 256
        pixels[..., 1] = ((x // 7 + y // 11) % 2) * 255
        pixels[..., 2] = (x * 3 + y * 19) % 256
        if mode == "RGBA":
            pixels[..., 3] = (x * 5 + y * 3) % 256
        with self.Image.fromarray(pixels) as im, BytesIO() as output:
            save_kwargs = {}
            if orientation is not None:
                exif = self.Image.Exif()
                exif[274] = orientation
                save_kwargs["exif"] = exif
            im.save(output, format="PNG", **save_kwargs)
            return output.getvalue()

    def test_exact_native_outputs_rgb_rgba_exif_and_geometry(self):
        cases = (("RGB", (4096, 4096), None), ("RGBA", (2049, 1025), None),
                 ("RGB", (997, 613), None), ("RGB", (1025, 4095), 6),
                 ("RGB", (17, 31), None), ("RGB", (1440, 1440), None))
        for mode, size, orientation in cases:
            with self.subTest(mode=mode, size=size, orientation=orientation):
                data = self.encoded(mode, size, orientation)
                baseline = self.Loader().load_bytes(data)
                early = self.Loader(qwen38_early_resize=True).load_bytes(data)
                try:
                    self.assertIs(early.original_bytes, data)
                    self.assertIn(helper.NAMESPACE, early.io_config)
                    self.assertIsNone(early.media.getexif().get(274))
                    with self.torch.inference_mode():
                        left = helper._processor()(images=[baseline.media], **self.kwargs)
                        right = helper._processor()(images=[early.media], **self.kwargs)
                    for key in ("pixel_values", "image_grid_thw"):
                        self.assertEqual(left[key].dtype, right[key].dtype)
                        self.assertEqual(left[key].device.type, "cpu")
                        self.assertEqual(right[key].device.type, "cpu")
                        self.assertTrue(self.torch.equal(left[key], right[key]), key)
                    self.assertEqual(left["pixel_values"].dtype, self.torch.float32)
                    self.assertEqual(left["image_grid_thw"].dtype, self.torch.int64)
                    del left, right
                finally:
                    baseline.media.close()
                    early.media.close()

    def test_flag_strict_and_disabled_path_unchanged(self):
        for flag in ("true", 1, None):
            with self.assertRaises(ValueError):
                self.Loader(qwen38_early_resize=flag)
        for kwargs in ({"image_mode": None}, {"rgba_background_color": [0, 0, 0]}):
            with self.assertRaises(ValueError):
                self.Loader(qwen38_early_resize=True, **kwargs)
        data = self.encoded()
        with patch.object(helper, "early_resize", side_effect=AssertionError("called")):
            result = self.Loader().load_bytes(data)
        self.assertEqual(result.media.size, (997, 613))
        self.assertIsNone(result.io_config)
        result.media.close()

    def test_white_compositing_before_resize_and_hash_namespace(self):
        with self.Image.new("RGBA", (256, 256), (250, 0, 0, 0)) as im, BytesIO() as buf:
            im.save(buf, format="PNG")
            data = buf.getvalue()
        baseline = self.Loader().load_bytes(data)
        first = self.Loader(qwen38_early_resize=True).load_bytes(data)
        second = self.Loader(qwen38_early_resize=True).load_bytes(data)
        try:
            self.assertEqual(first.media.getpixel((0, 0)), (255, 255, 255))
            for key, value in baseline.io_config.items():
                self.assertEqual(first.io_config[key], value)
            hash_item = lambda item: self.hasher.hash_kwargs("sha256", image=item)
            self.assertEqual(hash_item(first), hash_item(second))
            self.assertNotEqual(hash_item(baseline), hash_item(first))
        finally:
            for item in (baseline, first, second):
                item.media.close()

    def test_sources_closed_before_future_result_and_on_failure(self):
        data = self.encoded()
        held = []
        resize = helper.early_resize

        def track(source, config):
            held.append(source)
            return resize(source, config)

        with patch.object(helper, "early_resize", side_effect=track):
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(self.Loader(qwen38_early_resize=True).load_bytes, data).result()
        self.assertNotEqual(result.media.size, (997, 613))
        with self.assertRaises(ValueError):
            held[0].getpixel((0, 0))
        result.media.close()

        def fail(source, config):
            held.append(source)
            raise RuntimeError("injected resize failure")

        with patch.object(helper, "early_resize", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                self.Loader(qwen38_early_resize=True).load_bytes(data)
        with self.assertRaises(ValueError):
            held[-1].getpixel((0, 0))

    def test_helper_keeps_input_and_metadata_unchanged(self):
        original = {"image_mode": "RGB"}
        with self.Image.new("RGB", (256, 256)) as image:
            reduced, config = helper.early_resize(image, original)
            try:
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
                self.assertEqual(original, {"image_mode": "RGB"})
                self.assertIsNot(config, original)
                self.assertIsNot(reduced, image)
            finally:
                reduced.close()


if __name__ == "__main__":
    unittest.main()
