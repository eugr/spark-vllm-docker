#!/usr/bin/env python3
"""Offline regressions for quantized Qwen3-Next router checkpoints."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "mods/fix-qwen3-next-autoround"
SPEC = importlib.util.spec_from_file_location("autoround_patch", MOD / "patch_qwen3_next.py")
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# The router calls mirror upstream 0748d3bd57 (last successful nightly) and
# c961121519 (failed nightly, following the GateLinear conversion in #58234).
OLD_GATE = '''ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )'''
NEW_GATE = '''GateLinear(
            config.hidden_size,
            config.num_experts,
            prefix=f"{prefix}.gate",
        )'''
TEMPLATE = '''# Upstream commentary can drift — keep it byte-for-byte.
class Qwen3NextSparseMoeBlock:
    def __init__(self, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.gate = GATE_CALL
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size, 1, bias=False, quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

class UnrelatedModel:
    def __init__(self):
        self.gate = ReplicatedLinear(128, 16, quant_config=None)
'''


def source(gate=NEW_GATE):
    return TEMPLATE.replace("GATE_CALL", gate)


class ReplicatedLinear:
    def __init__(self, input_size, output_size, bias=False, quant_config=None, prefix=""):
        self.prefix = prefix
        self.quant_config = quant_config
        self.parameters = {"weight"} if quant_config is None else {"qweight", "qzeros", "scales"}


class GateLinear(ReplicatedLinear):
    pass


class AutoRoundModTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.package = Path(self.temp.name) / "vllm"
        self.target = self.package / "model_executor/models/qwen3_next.py"
        self.target.parent.mkdir(parents=True)

    def run_mod(self, discover=False):
        env = os.environ.copy()
        env.pop("VLLM_PACKAGE_ROOT", None)
        if discover:
            # Package discovery must not execute vLLM's initialization code.
            (self.package / "__init__.py").write_text('raise RuntimeError("vLLM imported")\n')
            env["PYTHONPATH"] = self.temp.name
        else:
            env["VLLM_PACKAGE_ROOT"] = str(self.package)
        return subprocess.run(
            ["bash", str(MOD / "run.sh")], env=env, text=True, capture_output=True,
            cwd=self.temp.name,
        )

    def instantiate(self, text, quant_config):
        namespace = {"ReplicatedLinear": ReplicatedLinear, "GateLinear": GateLinear}
        exec(compile(text, "<fixture>", "exec"), namespace)
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(hidden_size=2048, num_experts=512)
            ),
            quant_config=quant_config,
        )
        return namespace["Qwen3NextSparseMoeBlock"](config, prefix="layers.0.mlp")

    def test_checkpoint_router_parameters_and_shared_gate(self):
        for gate, constructor in ((OLD_GATE, ReplicatedLinear), (NEW_GATE, GateLinear)):
            with self.subTest(constructor=constructor.__name__):
                text = source(gate)
                config = object()
                self.assertNotIn("qweight", self.instantiate(text, config).gate.parameters)
                self.target.write_text(text)
                result = self.run_mod()
                self.assertEqual(result.returncode, 0, result.stderr)
                patched = self.target.read_text()
                model = self.instantiate(patched, config)
                self.assertIs(type(model.gate), constructor)
                self.assertIs(model.gate.quant_config, config)
                checkpoint_keys = {"layers.0.mlp.gate." + name for name in ("qweight", "qzeros", "scales")}
                parameter_keys = {model.gate.prefix + "." + name for name in model.gate.parameters}
                self.assertTrue(checkpoint_keys <= parameter_keys)
                self.assertEqual(model.shared_expert_gate.parameters, {"weight"})
                self.assertEqual(self.instantiate(patched, None).gate.parameters, {"weight"})

                # Only the selected router call can change.
                before = ast.parse(text)
                after = ast.parse(patched)
                before.body[0].body[0].body[2].value = after.body[0].body[0].body[2].value
                self.assertEqual(ast.dump(before), ast.dump(after))
                self.assertTrue(patched.startswith(text.splitlines(keepends=True)[0]))
                self.assertEqual(self.run_mod().returncode, 0)
                self.assertEqual(self.target.read_text(), patched)

    def test_package_discovery_does_not_import_vllm(self):
        self.target.write_text(source())
        result = self.run_mod(discover=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Patched router gate quantization", result.stdout)

    def test_already_compatible_router_is_unchanged(self):
        for gate in (OLD_GATE, NEW_GATE):
            with self.subTest(gate=gate):
                compatible = PATCHER.patched_text(source(gate))
                self.target.write_text(compatible)
                result = self.run_mod()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("already uses", result.stdout)
                self.assertEqual(self.target.read_text(), compatible)

    def test_source_formatting_drift(self):
        gate = '''GateLinear(config.hidden_size, config.num_experts,
            # Quantized router checkpoint; café is intentionally non-ASCII.
            prefix = f'{prefix}.gate')'''
        patched = PATCHER.patched_text(source(gate))
        self.assertIn("# Quantized router checkpoint; café", patched)
        self.assertIsNotNone(self.instantiate(patched, object()).gate.quant_config)

    def test_unsupported_sources_fail_without_writing(self):
        cases = {
            "missing class": source().replace("Qwen3NextSparseMoeBlock", "RenamedBlock"),
            "missing gate": source().replace("self.gate = GateLinear", "self.router = GateLinear"),
            "unknown constructor": source().replace("self.gate = GateLinear", "self.gate = CustomLinear"),
            "changed dimensions": source().replace("config.num_experts", "config.num_experts + 1"),
            "changed prefix": source().replace('f"{prefix}.gate"', 'f"{prefix}.router"'),
            "changed config binding": source().replace("quant_config = vllm_config.quant_config", "quant_config = None"),
            "dynamic config": source(OLD_GATE.replace("quant_config=None", "quant_config=choose_config()")),
            "keyword expansion": source(NEW_GATE.replace("config.num_experts,", "config.num_experts, **options,")),
            "duplicate gate": source().replace("        self.shared_expert_gate =", "        self.gate = GateLinear(config.hidden_size, config.num_experts)\n        self.shared_expert_gate ="),
            "duplicate config": source(OLD_GATE.replace("quant_config=None,", "quant_config=None, quant_config=None,")),
            "invalid Python": source() + "\ninvalid syntax!\n",
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                self.target.write_text(text)
                result = self.run_mod()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ERROR: cannot enable quantized router gate", result.stderr)
                self.assertEqual(self.target.read_text(), text)

    def test_missing_source_stops_mod(self):
        result = self.run_mod()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR: cannot enable quantized router gate", result.stderr)


if __name__ == "__main__":
    unittest.main()
