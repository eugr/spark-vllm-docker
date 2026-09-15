#!/usr/bin/env python3
"""CPU-only checks for #53007's fallback and the primary-size optimization."""

import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
PATCHER_PATH = PROJECT_DIR / "docker/patch_vllm_swa_block_size.py"
SPEC = importlib.util.spec_from_file_location("swa_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# Relevant excerpts from vllm-project/vllm@86aca6619 attention.py.
# The surrounding model and tensor objects are replaced with CPU stand-ins.
# MultipleOf is supplied below instead of importing vLLM.
UPSTREAM = '''
def _largest_kernel_block_within(
    attn_backend, per_token_bytes, page_budget, fallback,
):
    sizes = attn_backend.get_supported_kernel_block_sizes()
    max_block_size = page_budget // per_token_bytes
    candidates = [s for s in sizes if isinstance(s, int)]
    candidates.extend(
        max(s.base, max_block_size // s.base * s.base)
        for s in sizes
        if isinstance(s, MultipleOf)
    )
    if not candidates:
        return fallback
    smallest = min(candidates)
    fitting = [b for b in candidates if b * per_token_bytes <= page_budget]
    return max(fitting) if fitting else smallest


class Attention:
    def get_kv_cache_spec(self, vllm_config):
        block_size = vllm_config.cache_config.block_size
        if self.sliding_window is not None:
            shared_page = vllm_config.cache_config.skip_page_size_padded
            sw_per_token = self.attn_backend.customize_spec(
                SlidingWindowSpec(
                    block_size=1,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    head_size_v=self.head_size_v,
                    dtype=self.kv_cache_torch_dtype,
                    kv_quant_mode=1,
                    sliding_window=self.sliding_window,
                )
            ).real_page_size_bytes
            page_budget = shared_page or sw_per_token * block_size
            sw_block_size = _largest_kernel_block_within(
                self.attn_backend, sw_per_token, page_budget, block_size
            )
            return SlidingWindowSpec(
                block_size=sw_block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=1,
                sliding_window=self.sliding_window,
                page_size_padded=shared_page,
            )
        return "full attention unchanged"
'''


@dataclass(frozen=True)
class MultipleOf:
    base: int


class SlidingWindowSpec(SimpleNamespace):
    @property
    def real_page_size_bytes(self):
        return self.block_size * self.num_kv_heads * (self.head_size + self.head_size_v)


def select_spec(source, sizes, primary=1648, shared_page=None, kv_heads=8, window=2048):
    namespace = {"MultipleOf": MultipleOf, "SlidingWindowSpec": SlidingWindowSpec}
    exec(compile(source, "attention_fixture.py", "exec"), namespace)
    layer = namespace["Attention"]()
    layer.attn_backend = SimpleNamespace(
        get_supported_kernel_block_sizes=lambda: sizes,
        customize_spec=lambda spec: spec,
        is_mla=lambda: False,
    )
    layer.attn_type = 2
    layer.kv_cache_dtype = "fp8"
    layer.num_kv_heads = kv_heads
    layer.head_size = layer.head_size_v = 128
    layer.kv_cache_torch_dtype = "fp8"
    layer.sliding_window = window
    config = SimpleNamespace(cache_config=SimpleNamespace(
        block_size=primary, skip_page_size_padded=shared_page,
    ))
    return layer.get_kv_cache_spec(config)


def qwen_cache_geometry(spec):
    # The other hybrid groups impose a 1648 * 2048-byte physical page.
    common_page = 1648 * 2 * 4 * 256
    natural_page = spec.real_page_size_bytes
    block_size = spec.block_size
    if common_page % natural_page == 0:
        block_size *= common_page // natural_page
    # Otherwise vLLM pads the bytes but leaves the token count unchanged.
    pool_bytes = 5 * common_page
    draft_blocks = math.ceil((2048 - 1 + 2 * 16384) / block_size) + 1
    target_blocks = 4 * math.ceil(262144 / 1648)
    mamba_blocks = 10 * (2 + 8)
    required_gib = (draft_blocks + target_blocks + mamba_blocks) * pool_bytes / 2**30
    return block_size, draft_blocks, required_gib


class SWABlockFallbackTests(unittest.TestCase):
    def setUp(self):
        self.patched, _ = PATCHER.patch_source(UPSTREAM)

    def test_qwen_dflash_capacity_regression(self):
        before = select_spec(UPSTREAM, [16, 32, 64])
        after = select_spec(self.patched, [16, 32, 64])
        self.assertEqual(before.block_size, 64)
        self.assertEqual(after.block_size, 16)
        self.assertEqual(qwen_cache_geometry(before), (64, 545, 20.195770263671875))
        self.assertEqual(qwen_cache_geometry(after), (1648, 23, 11.991729736328125))

    def test_kimi_primary_size_optimization_is_preserved(self):
        # #53007's original case: 1152 B/token target, 1024 B/token SWA,
        # primary 1536. Scaling a 16-token SWA page would instead yield 1728.
        before = select_spec(UPSTREAM, [MultipleOf(16)], primary=1536, kv_heads=4)
        after = select_spec(self.patched, [MultipleOf(16)], primary=1536, kv_heads=4)
        self.assertEqual(vars(after), vars(before))
        self.assertEqual(after.block_size, 1536)
        self.assertEqual(math.lcm(1536, after.block_size), 1536)
        old_block = 16 * (1536 * 1152 // (16 * 1024))
        self.assertEqual(math.lcm(1536, old_block), 13824)

    def test_supported_discrete_primary_is_preserved(self):
        for primary in [16, 32, 64]:
            with self.subTest(primary=primary):
                spec = select_spec(self.patched, [16, 32, 64], primary=primary)
                self.assertEqual(spec.block_size, primary)

    def test_multipleof_fallback_uses_base_not_rounded_candidate(self):
        spec = select_spec(self.patched, [MultipleOf(16)], primary=1650)
        self.assertEqual(spec.block_size, 16)

    def test_mixed_backend_declarations(self):
        for primary, expected in [(1536, 1536), (1650, 16)]:
            with self.subTest(primary=primary):
                spec = select_spec(self.patched, [64, MultipleOf(16)], primary=primary)
                self.assertEqual(spec.block_size, expected)

    def test_explicit_skip_quant_padding_keeps_largest_fit(self):
        for sizes in [[16, 32, 64], [MultipleOf(16)], [64, MultipleOf(16)]]:
            for page in [16 * 2048, 48 * 2048, 1648 * 2048]:
                with self.subTest(sizes=sizes, page=page):
                    before = select_spec(UPSTREAM, sizes, shared_page=page)
                    after = select_spec(self.patched, sizes, shared_page=page)
                    self.assertEqual(vars(before), vars(after))

    def test_empty_backend_declaration_keeps_fallback(self):
        self.assertEqual(select_spec(self.patched, []).block_size, 1648)

    def test_full_attention_is_untouched(self):
        self.assertEqual(select_spec(self.patched, [16, 32, 64], window=None),
                         "full attention unchanged")

    def test_patch_is_idempotent(self):
        patched_again, message = PATCHER.patch_source(self.patched)
        self.assertEqual(patched_again, self.patched)
        self.assertIn("already present", message)

    def test_pre_pr_and_missing_selector_are_skipped(self):
        legacy = UPSTREAM.replace(
            "            page_budget = shared_page or sw_per_token * block_size\n", ""
        ).replace(
            "self.attn_backend, sw_per_token, page_budget, block_size",
            "self.attn_backend, sw_per_token, shared_page, block_size",
        )
        for source in [legacy, "class Attention: pass\n"]:
            with self.subTest(source=source):
                self.assertEqual(PATCHER.patch_source(source)[0], source)

    def test_partial_or_changed_sources_are_rejected(self):
        for source in [
            UPSTREAM.replace("page_budget = shared_page or", "page_budget = other or"),
            UPSTREAM + UPSTREAM,
            self.patched.replace("if shared_page is None", "if shared_page"),
        ]:
            with self.subTest(source=source):
                with self.assertRaises(PATCHER.PatchError):
                    PATCHER.patch_source(source)

    def test_cli_and_failed_patch_leave_files_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / PATCHER.TARGET_REL
            command = [sys.executable, str(PATCHER_PATH), str(root)]
            missing = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 0, missing.stderr)
            self.assertIn("not applicable", missing.stdout)
            target.parent.mkdir(parents=True)
            target.write_text(UPSTREAM)
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(target.read_text(), self.patched)
            unknown = UPSTREAM.replace("page_budget = shared_page or", "page_budget = other or")
            target.write_text(unknown)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_text(), unknown)


if __name__ == "__main__":
    unittest.main()
