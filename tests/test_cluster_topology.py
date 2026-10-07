#!/usr/bin/env python3
"""Ring discovery and distribution tests; all hosts and network tools are mocked."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cluster_topology import Topology, discover, validate_launch

NODES = [f"192.0.2.{n}" for n in range(1, 5)]
# Physical order deliberately differs from sorting management addresses.
ORDER = [NODES[0], NODES[2], NODES[1], NODES[3]]
LINKS = []
for i, (a, b) in enumerate(zip(ORDER, ORDER[1:] + ORDER[:1])):
    for rail in range(2):
        LINKS.append({a: f"10.20.{i * 2 + rail}.1", b: f"10.20.{i * 2 + rail}.2"})


def config(nodes=ORDER, links=LINKS):
    return (f"CLUSTER_NODES={','.join(nodes)}\nLOCAL_IP={NODES[0]}\n"
            "ETH_IF=management0\nIB_IF=roce0,roce1,roce2,roce3\n"
            f"CLUSTER_LINKS='{json.dumps(links, separators=(',', ':'))}'\n")


def larger_ring(count):
    nodes = [f"192.0.2.{i}" for i in range(1, count + 1)]
    links = [{a: f"10.40.{2 * i + rail}.1", b: f"10.40.{2 * i + rail}.2"}
             for i, (a, b) in enumerate(zip(nodes, nodes[1:] + nodes[:1]))
             for rail in range(2)]
    return nodes, links


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.topology = Topology(NODES, LINKS)

    def test_ring_order_collapses_twin_rails(self):
        self.assertTrue(self.topology.ring)
        self.assertEqual(self.topology.rank_order(NODES[0]), ORDER)
        self.assertEqual(len(self.topology.adj[NODES[0]]), 2)

    def test_copy_paths_cover_opposite_node(self):
        paths = self.topology.paths(NODES[0])
        self.assertEqual(paths[NODES[1]], [(NODES[2], "10.20.0.2"), (NODES[1], "10.20.2.2")])
        self.assertEqual(len(paths[NODES[3]]), 1)

    def test_launch_order_and_trim_validation(self):
        self.assertEqual(validate_launch(self.topology, ORDER, "vllm serve test --tensor-parallel-size 4")["NCCL_ALGO"], "Ring")
        self.assertEqual(validate_launch(self.topology, ORDER, "vllm serve test --tensor-parallel-size=2"), {})
        self.assertEqual(validate_launch(self.topology, ORDER, "vllm serve test -tp 1"), {})
        for nodes, command in ((NODES, ""), (ORDER, "vllm serve test -tp 3"),
                               (ORDER, "vllm serve test -tp 2 -pp 2"),
                               (ORDER, "vllm serve test -tp 8")):
            with self.subTest(nodes=nodes, command=command), self.assertRaises(ValueError):
                validate_launch(self.topology, nodes, command)
        with self.assertRaisesRegex(ValueError, "native backend"):
            validate_launch(self.topology, ORDER, ray=True)
        with self.assertRaisesRegex(ValueError, "NCCL_ALGO=Ring"):
            validate_launch(self.topology, ORDER, env={"NCCL_ALGO": "Tree"})

    def test_invalid_and_disconnected_links(self):
        for links in (LINKS[:2], LINKS + [LINKS[0]], [{NODES[0]: "10.0.0.1"}],
                      [{NODES[0]: "10.0.0.1", NODES[1]: "10.0.0.1"}],
                      [{NODES[0]: "$(bad)", NODES[1]: "10.0.0.2"}]):
            with self.subTest(links=links), self.assertRaises(ValueError):
                Topology(NODES, links)

    def test_complete_and_legacy(self):
        triangle = [{a: f"10.30.0.{i + 1}", b: f"10.30.0.{j + 1}"}
                    for i, a in enumerate(NODES[:3]) for j, b in enumerate(NODES[:3]) if i < j]
        topology = Topology(NODES[:3], triangle)
        self.assertTrue(topology.complete)
        self.assertFalse(topology.ring)
        self.assertEqual(validate_launch(topology, NODES[:3]), {})
        self.assertIsNone(Topology.from_env({"CLUSTER_NODES": ",".join(NODES)}))

    def test_multihop_ssh_config_is_parseable_and_temporary(self):
        nodes = [f"192.0.2.{i}" for i in range(1, 7)]
        links = [{a: f"10.40.{i}.1", b: f"10.40.{i}.2"}
                 for i, (a, b) in enumerate(zip(nodes, nodes[1:] + nodes[:1]))]
        topology = Topology(nodes, links)
        with topology.ssh_config(nodes[0]) as path:
            result = subprocess.run(["/usr/bin/ssh", "-G", "-F", str(path), nodes[3]], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("hostname 10.40.2.2\n", result.stdout)
            self.assertIn("proxyjump 192.0.2.2,192.0.2.3\n", result.stdout.replace("[", "").replace("]", ""))
            self.assertIn("hostkeyalias 192.0.2.4\n", result.stdout)
            self.assertNotIn("    User ", path.read_text())
        self.assertFalse(path.exists())

    def test_discovery_checks_remote_rails_in_both_directions(self):
        records = {node: [] for node in NODES}
        for i, link in enumerate(LINKS):
            for node, address in link.items():
                records[node].append({"interface": f"roce{i}", "address": address, "prefix": 24})
        def probe(node, head, command, **kwargs):
            return subprocess.CompletedProcess(command, 0, json.dumps(records[node]) if command[0] == "python3" else "", "")
        with patch("cluster_topology.on_node", side_effect=probe) as mock:
            topology = discover(NODES, NODES[0])
        self.assertEqual(topology.nodes, ORDER)
        pings = [call for call in mock.call_args_list if call.args[2][0] == "ping"]
        self.assertEqual(len(pings), 16)
        with patch("cluster_topology.on_node", return_value=subprocess.CompletedProcess([], 1, "", "")):
            with self.assertRaisesRegex(ValueError, "Could not inspect"):
                discover(NODES, NODES[0])

    def test_larger_ring_discovery_and_launch_validation(self):
        for count in (6, 8):
            with self.subTest(count=count):
                nodes, links = larger_ring(count)
                records = {node: [] for node in nodes}
                for i, link in enumerate(links):
                    for node, address in link.items():
                        records[node].append({"interface": f"roce{i}", "address": address, "prefix": 24})
                def probe(node, head, command, **kwargs):
                    return subprocess.CompletedProcess(command, 0, json.dumps(records[node]))
                # Management discovery order must not determine physical rank order.
                with patch("cluster_topology.on_node", side_effect=probe) as mock:
                    topology = discover(nodes[:1] + nodes[:0:-1], nodes[0])
                self.assertEqual(topology.nodes, nodes)
                self.assertEqual(len(topology.links), 2 * count)
                pings = [call for call in mock.call_args_list if call.args[2][0] == "ping"]
                self.assertEqual(len(pings), 4 * count)
                paths = topology.paths(nodes[0])
                self.assertEqual([len(paths[node]) for node in nodes],
                                 [min(i, count - i) for i in range(count)])
                self.assertEqual(validate_launch(topology, nodes, f"vllm serve test -tp {count}")["NCCL_ALGO"], "Ring")
                with self.assertRaisesRegex(ValueError, "break the ring"):
                    validate_launch(topology, nodes, f"vllm serve test -tp {count - 1}")


MOCK = r'''#!/usr/bin/env python3
import json, os, pathlib, shlex, sys
tool = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
record = {"tool": tool, "args": args}
if tool == "ssh" and "-F" in args:
    path = args[args.index("-F") + 1]
    record["config"] = pathlib.Path(path).read_text()
if tool == "rsync" and "-e" in args:
    remote = shlex.split(args[args.index("-e") + 1])
    record["config"] = pathlib.Path(remote[remote.index("-F") + 1]).read_text()
if tool == "ssh" and args[-1] == "docker load":
    record["bytes"] = len(sys.stdin.buffer.read())
with open(os.environ["TOPOLOGY_TEST_LOG"], "a") as log:
    log.write(json.dumps(record) + "\n")
if tool == "docker":
    if args[:2] == ["image", "inspect"]:
        if "--format" in args:
            print("sha256:current")
        else:
            print(pathlib.Path(os.environ["TOPOLOGY_TEST_IMAGE"]).read_text())
    elif args[0] == "save":
        pathlib.Path(args[args.index("-o") + 1]).write_bytes(b"mock image stream")
elif tool == "ssh" and "docker image inspect" in args[-1]:
    if "--format" in args[-1]:
        print("sha256:old" if "-F" in args else "sha256:current")
    else:
        image = json.loads(pathlib.Path(os.environ["TOPOLOGY_TEST_IMAGE"]).read_text())
        image[0]["Config"]["Cmd"] = ["different-runtime"]
        print(json.dumps(image))
elif tool == "ssh" and args[-1].startswith("docker ps"):
    sys.exit(1)
if os.environ.get("TOPOLOGY_FAIL_COPY") and (tool == "rsync" or (tool == "ssh" and args[-1] == "docker load")):
    sys.exit(1)
'''


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="topology-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("autodiscover.sh", "build-and-copy.sh", "hf-download.sh", "hf-cache.py", "cluster_topology.py", "launch-cluster.sh"):
            shutil.copy2(ROOT / name, self.root / name)
        (self.root / "docker").mkdir()
        shutil.copy2(ROOT / "docker/image_identity.py", self.root / "docker/image_identity.py")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for tool in ("ssh", "docker", "rsync", "uvx", "sleep"):
            path = self.bin / tool
            path.write_text(MOCK)
            path.chmod(0o755)
        self.config = self.root / "custom.env"
        self.config.write_text(config())
        self.log = self.root / "calls.jsonl"
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
                    "TOPOLOGY_TEST_LOG": str(self.log), "HF_HOME": str(self.root / "cache"), "USER": "fixture",
                    "TOPOLOGY_TEST_IMAGE": str(ROOT / "tests/fixtures/image-identity/classic.json")}
        for key in list(self.env):
            if key.startswith("DOTENV_") or key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
                del self.env[key]
        (self.root / "cache/hub/models--org--model").mkdir(parents=True)

    def run_script(self, script, *args, ok=True, input=None):
        result = subprocess.run(["bash", str(self.root / script), *args], cwd=self.root,
                                env=self.env, capture_output=True, text=True, input=input, timeout=30)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def assert_route(self, call):
        self.assertIn("Host 192.0.2.2\n    HostName 10.20.2.2\n", call["config"])
        self.assertIn("    ProxyJump 192.0.2.3\n", call["config"])

    def test_image_inspection_and_stream_use_same_routes(self):
        self.run_script("build-and-copy.sh", "--config", str(self.config), "--no-build", "-c", "--copy-parallel")
        calls = [call for call in self.calls() if call["tool"] == "ssh"]
        self.assertEqual(len(calls), 9)
        for call in calls:
            self.assert_route(call)
        self.assertEqual({call["args"][-2] for call in calls}, set(ORDER[1:]))
        self.assertEqual([call["bytes"] for call in calls if "bytes" in call], [17] * 3)

    def test_model_permissions_and_rsync_use_same_routes(self):
        self.run_script("hf-download.sh", "--config", str(self.config), "org/model", "-c", "--copy-parallel")
        calls = [call for call in self.calls() if call["tool"] in ("ssh", "rsync")]
        self.assertEqual(len(calls), 6)
        for call in calls:
            self.assert_route(call)
        destinations = {call["args"][-1].split(":", 1)[0] for call in calls if call["tool"] == "rsync"}
        self.assertEqual(destinations, set(ORDER[1:]))

    def test_larger_ring_image_and_model_distribution(self):
        for count in (6, 8):
            for parallel in (False, True):
                with self.subTest(count=count, parallel=parallel):
                    nodes, links = larger_ring(count)
                    self.config.write_text(config(nodes, links))
                    self.log.unlink(missing_ok=True)
                    options = ["--copy-parallel"] if parallel else []
                    self.run_script("build-and-copy.sh", "--config", str(self.config), "--no-build", "-c", *options)
                    self.run_script("hf-download.sh", "--config", str(self.config), "org/model", "-c", *options)
                    calls = self.calls()
                    inspections = [call for call in calls if call["tool"] == "ssh" and "docker image inspect" in call["args"][-1]]
                    streams = [call for call in calls if "bytes" in call]
                    permissions = [call for call in calls if call["tool"] == "ssh" and call not in inspections + streams]
                    copies = [call for call in calls if call["tool"] == "rsync"]
                    for group in (inspections, streams, permissions, copies):
                        self.assertEqual(len(group), (count - 1) * (2 if group is inspections else 1))
                        destinations = {call["args"][-1].split(":", 1)[0] if call["tool"] == "rsync"
                                        else call["args"][-2] for call in group}
                        self.assertEqual(destinations, set(nodes[1:]))
                    self.assertEqual([call["bytes"] for call in streams], [17] * (count - 1))
                    # Both distribution scripts must use the same entire routing table.
                    configs = {call["config"] for call in inspections + streams + permissions + copies}
                    self.assertEqual(len(configs), 1)
                    routing = self.root / "routing.conf"
                    routing.write_text(configs.pop())
                    # Check every route using OpenSSH itself, in both directions.
                    for i, node in enumerate(nodes[1:], 1):
                        forward = i <= count // 2
                        edge, endpoint = (i - 1, 2) if forward else (i, 1)
                        jumps = nodes[1:i] if forward else nodes[:i:-1]
                        result = subprocess.run(["/usr/bin/ssh", "-G", "-F", str(routing), node], capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn(f"hostname 10.40.{2 * edge}.{endpoint}\n", result.stdout)
                        output = result.stdout.replace("[", "").replace("]", "")
                        if jumps:
                            self.assertIn(f"proxyjump {','.join(jumps)}\n", output)
                        else:
                            self.assertNotIn("proxyjump ", output)

    def test_explicit_copy_subset_and_failure(self):
        self.run_script("hf-download.sh", "--config", str(self.config), "org/model", "-c", NODES[1])
        self.assertEqual(len([call for call in self.calls() if call["tool"] == "rsync"]), 1)
        self.env["TOPOLOGY_FAIL_COPY"] = "1"
        self.run_script("build-and-copy.sh", "--config", str(self.config), "--no-build", "-c", "--copy-parallel", ok=False)

    def test_launch_generates_ring_env_for_every_rank(self):
        self.run_script("launch-cluster.sh", "--config", str(self.config), "--no-cache-dirs", "-d",
                        "exec", "vllm", "serve", "test", "--tensor-parallel-size", "4")
        commands = [" ".join(call["args"]) for call in self.calls()]
        runs = [command for command in commands if "docker run" in command or command.startswith("run ")]
        self.assertEqual(len(runs), 4)
        for command in runs:
            self.assertIn("NCCL_ALGO=Ring", command)
            self.assertIn("NCCL_IB_SUBNET_AWARE_ROUTING=1", command)

    def test_invalid_launch_never_starts_or_stops_containers(self):
        for order, tp in ((NODES, "4"), (ORDER, "3")):
            self.config.write_text(config(order))
            result = self.run_script("launch-cluster.sh", "--config", str(self.config), "--no-cache-dirs",
                                     "exec", "vllm", "serve", "test", "--tensor-parallel-size", tp, ok=False)
            self.assertIn("ring", result.stdout + result.stderr)
            self.assertEqual(self.calls(), [])

    def test_discovery_saves_one_config_and_rejects_broken_selection(self):
        script = self.root / "save.sh"
        script.write_text('''#!/bin/bash
set -e
CONFIG_FILE="$1"; FORCE_DISCOVER=true
source "$(dirname "$0")/autodiscover.sh"
LOCAL_IP=192.0.2.1; ETH_IF=management0; IB_IF=roce0; MESH_MODE=true
NODES_ARG=192.0.2.1,192.0.2.3,192.0.2.2,192.0.2.4
PEER_NODES=(192.0.2.3 192.0.2.2 192.0.2.4)
COPY_PEER_NODES=("${PEER_NODES[@]}")
save_config
''')
        original = self.config.read_text()
        self.run_script("save.sh", str(self.config), input="y\ny\ny\ny\nn\n", ok=False)
        self.assertEqual(self.config.read_text(), original)
        self.run_script("save.sh", str(self.config), input="y\ny\ny\ny\ny\n")
        saved = self.config.read_text()
        self.assertIn("CONTAINER_NCCL_ALGO=Ring\n", saved)
        self.assertIn("CLUSTER_LINKS=", saved)
        self.assertNotIn("COPY_HOSTS=", saved)
        self.assertNotIn("SSH_USER=", saved)

    def test_legacy_two_node_triangle_and_switch_copy_selection(self):
        script = self.root / "legacy.sh"
        script.write_text('''#!/bin/bash
set -e
CONFIG_FILE=/dev/null
source "$(dirname "$0")/autodiscover.sh"
LOCAL_IP=192.0.2.1; ETH_IF=management0
PEER_NODES=(192.0.2.2)
MESH_MODE=false
detect_copy_hosts
[[ "${COPY_PEER_NODES[*]}" == 192.0.2.2 ]]
# A switch can expose arbitrarily many directly reachable peers.
PEER_NODES=(192.0.2.2 192.0.2.3 192.0.2.4)
detect_copy_hosts
[[ "${COPY_PEER_NODES[*]}" == '192.0.2.2 192.0.2.3 192.0.2.4' ]]
MESH_MODE=true
PEER_NODES=(192.0.2.2 192.0.2.3)
ip() { echo '1: fixture inet 10.50.0.1/24'; }
_scan_subnet_for_gb10() { printf '10.50.0.2\\n10.50.0.3\\n' >> "$3"; }
ssh() { echo "${@: -2:1}"; }
# A triangle must never use graph discovery.
detect_topology() { echo 'unexpected graph discovery' >&2; return 99; }
detect_copy_hosts
[[ "${COPY_PEER_NODES[*]}" == '10.50.0.2 10.50.0.3' ]]
[[ -z "${DOTENV_CLUSTER_LINKS:-}" ]]
# Four active NICs on a complete network also retain the legacy scan.
PEER_NODES=(192.0.2.2 192.0.2.3 192.0.2.4)
detect_topology() { return 0; }
_scan_subnet_for_gb10() { printf '10.50.0.2\\n10.50.0.3\\n10.50.0.4\\n' >> "$3"; }
detect_copy_hosts
[[ "${COPY_PEER_NODES[*]}" == '10.50.0.2 10.50.0.3 10.50.0.4' ]]
[[ -z "${DOTENV_CLUSTER_LINKS:-}" ]]
''')
        self.run_script("legacy.sh")


class RecipeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("recipe_runner", ROOT / "run-recipe.py")
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)

    def test_config_forwarded_to_distribution_scripts(self):
        with tempfile.NamedTemporaryFile(suffix=".env") as file:
            self.runner.ENV_FILE = Path(file.name)
            with patch.object(self.runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
                self.runner.build_image("test", ORDER[1:], copy_only=True)
                self.runner.download_model("org/model", ORDER[1:])
            for call in run.call_args_list:
                argv = call.args[0]
                self.assertEqual(argv[argv.index("--config") + 1], file.name)
                self.assertEqual(argv[argv.index("--copy-to") + 1], ",".join(ORDER[1:]))
            self.assertIn("--no-build", run.call_args_list[0].args[0])

    def test_dry_run_rejects_bad_order_without_network(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env") as file:
            file.write(config(NODES)); file.flush()
            result = subprocess.run([sys.executable, str(ROOT / "run-recipe.py"),
                                     "recipes/qwen3.8-27b-nvfp4-dflash2.yaml", "--dry-run", "--tp", "4",
                                     "--config", file.name, "-n", ",".join(NODES)],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("physical ring order", result.stdout)

    def test_cached_model_is_distributed_to_every_ring_worker(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env") as file:
            file.write(config()); file.flush()
            argv = ["run-recipe.py", "recipes/qwen3.8-27b-nvfp4-dflash2.yaml", "--download-only",
                    "--config", file.name, "--tp", "4"]
            with patch.object(sys, "argv", argv), patch.object(self.runner, "check_model_exists", return_value=True), \
                    patch.object(self.runner, "download_model", return_value=True) as download:
                self.assertEqual(self.runner.main(), 0)
            self.assertEqual(download.call_args.args[1], ORDER[1:])

    def test_existing_image_is_synced_without_building_for_ring(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env") as file:
            file.write(config()); file.flush()
            argv = ["run-recipe.py", "recipes/qwen3.8-27b-nvfp4-dflash2.yaml", "--build-only",
                    "--config", file.name, "--tp", "4"]
            with patch.object(sys, "argv", argv), patch.object(self.runner, "check_image_exists", return_value=True), \
                    patch.object(self.runner, "build_image", return_value=True) as build:
                self.assertEqual(self.runner.main(), 0)
            self.assertEqual(build.call_args.args[1], ORDER[1:])
            self.assertTrue(build.call_args.kwargs["copy_only"])


if __name__ == "__main__":
    unittest.main()
