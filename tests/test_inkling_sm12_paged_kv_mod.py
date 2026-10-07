#!/usr/bin/env python3
"""Offline regressions for the Inkling SM12 paged-KV mod's source patcher."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods/inkling-sm12-paged-kv"
SPEC = importlib.util.spec_from_file_location("inkling_patch", MOD / "patch_inkling.py")
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# Trimmed stand-ins for vllm/models/inkling/nvidia/ops/fa4_rel_attention.py.
# The anchored lines are byte-for-byte upstream; everything else is filler.
HEAD = '''from functools import cache

from vllm.platforms import current_platform


@cache
def _use_sheared_bias() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major in (10, 11)


@cache
def _get_score_mod(rel_extent):
    from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK

    return SeqlenInfoQK, rel_extent


def inkling_fa4_num_splits(*, is_local):
    capability = current_platform.get_device_capability()
    if capability is not None and capability.major == 9:
        return 1
    return 1 if is_local else 32
'''
# Dispatch block of upstream 653ebb52d (0.26.1); current main has the same lines.
DISPATCH = '''    if _use_sheared_bias():
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func

        bias_kwargs = {"rel_bias": rel_logits}
    else:
        from vllm.vllm_flash_attn.cute import (
            flash_attn_varlen_func as cute_flash_attn_varlen_func,
        )

        flash_attn_varlen_func = cute_flash_attn_varlen_func
        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }
    return flash_attn_varlen_func(q, **bias_kwargs)
'''
# The older dispatch block the patcher also accepts.
LEGACY_DISPATCH = '''    if _use_sheared_bias():
        from vllm.third_party.tml_fa4 import flash_attn_varlen_func

        bias_kwargs = {"rel_bias": rel_logits}
    else:
        from vllm.vllm_flash_attn.cute import flash_attn_varlen_func

        bias_kwargs = {
            "score_mod": _get_score_mod(rel_extent),
            "aux_tensors": [rel_logits],
        }
    return flash_attn_varlen_func(q, **bias_kwargs)
'''
FUNCTION_ENTRY = "\n\ndef inkling_fa4_rel_attention(q, rel_logits, rel_extent):\n"
# Since vllm-project/vllm#49315 the same body is a method of the JIT kernel class.
CLASS_ENTRY = (
    "\n\nclass InklingFA4RelAttentionKernel:\n"
    "    @staticmethod\n"
    "    def kernel(q, rel_logits, rel_extent):\n"
)


def function_layout(dispatch=DISPATCH):
    return HEAD + FUNCTION_ENTRY + dispatch


def class_layout(dispatch=DISPATCH):
    return HEAD + CLASS_ENTRY + textwrap.indent(dispatch, "    ")


def changed_lines(before, after):
    """Lines the patcher added, ignoring indentation."""
    old = [line.strip() for line in before.splitlines()]
    return [line.strip() for line in after.splitlines() if line.strip() not in old]


class InklingPatcherTests(unittest.TestCase):
    def assert_patched(self, source):
        patched = PATCHER.patched_text(source)
        self.assertIn(PATCHER.MARKER, patched)
        self.assertIn("def _use_sm12_paged_kv() -> bool:", patched)
        self.assertIn("vllm.third_party.inkling_sm120_fa4_adapter", patched)
        self.assertIn("vllm.third_party.inkling_sm120_fa4.seqlen_info", patched)
        self.assertIn("capability.major in (9, 12)", patched)
        # Other architectures keep the pinned kernel.
        self.assertIn("from vllm.vllm_flash_attn.cute import", patched)
        compile(patched, "fa4_rel_attention.py", "exec")
        return patched

    def test_function_layout_is_patched(self):
        self.assert_patched(function_layout())

    def test_legacy_function_layout_is_patched(self):
        self.assert_patched(function_layout(LEGACY_DISPATCH))

    def test_kernel_class_layout_is_patched(self):
        patched = self.assert_patched(class_layout())
        # The adapter import must stay inside the method's else branch.
        self.assertIn(
            "            if _use_sm12_paged_kv():\n"
            "                from vllm.third_party.inkling_sm120_fa4_adapter import (\n",
            patched,
        )

    def test_both_layouts_receive_the_same_edits(self):
        function_edits = changed_lines(function_layout(), PATCHER.patched_text(function_layout()))
        class_edits = changed_lines(class_layout(), PATCHER.patched_text(class_layout()))
        self.assertEqual(function_edits, class_edits)

    def test_repeat_application_is_a_no_op(self):
        for source in (function_layout(), class_layout()):
            patched = PATCHER.patched_text(source)
            self.assertEqual(PATCHER.patched_text(patched), patched)

    def test_unknown_layouts_are_refused(self):
        without_entry_point = HEAD
        class_without_kernel = HEAD + CLASS_ENTRY.replace("def kernel(", "def run(") + textwrap.indent(DISPATCH, "    ")
        unknown_dispatch = function_layout(DISPATCH.replace("cute_flash_attn_varlen_func", "renamed_func"))
        for source in (without_entry_point, class_without_kernel, unknown_dispatch):
            with self.assertRaises(ValueError):
                PATCHER.patched_text(source)

    def test_command_line_checks_patches_and_refuses(self):
        script = str(MOD / "patch_inkling.py")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "fa4_rel_attention.py"
            target.write_text(class_layout())

            check = subprocess.run([sys.executable, script, "--check", str(target)], capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertIn("is compatible", check.stdout)
            self.assertEqual(target.read_text(), class_layout())

            apply = subprocess.run([sys.executable, script, str(target)], capture_output=True, text=True)
            self.assertEqual(apply.returncode, 0, apply.stderr)
            self.assertIn(PATCHER.MARKER, target.read_text())

            again = subprocess.run([sys.executable, script, "--check", str(target)], capture_output=True, text=True)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn("already patched", again.stdout)

            unknown = Path(tmp) / "unknown.py"
            unknown.write_text(HEAD)
            refused = subprocess.run([sys.executable, script, str(unknown)], capture_output=True, text=True)
            self.assertEqual(refused.returncode, 1)
            self.assertIn("refusing to patch", refused.stderr)
            self.assertEqual(unknown.read_text(), HEAD)


if __name__ == "__main__":
    unittest.main()
