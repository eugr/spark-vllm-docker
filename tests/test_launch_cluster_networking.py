#!/usr/bin/env python3
"""Check launcher networking with fake Docker/SSH and synthetic configuration."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_CONFIG = """ETH_IF=cluster0
IB_IF=roce0,roce1
LOCAL_IP=192.0.2.1
CLUSTER_NODES=192.0.2.1,192.0.2.2
"""


class LauncherNetworkingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="launcher-networking-")
        self.addCleanup(self.tmp.cleanup)
        self.fixture = Path(self.tmp.name)
        for name in ("launch-cluster.sh", "autodiscover.sh"):
            shutil.copy2(ROOT / name, self.fixture / name)
        self.bin = self.fixture / "bin"
        self.bin.mkdir()
        self.config = self.fixture / "test.env"
        self.log = self.fixture / "commands.jsonl"
        self.env = {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "HOME": os.environ["HOME"],
            "NETWORK_TEST_LOG": str(self.log),
        }
        self.write_tool("docker", """#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ['NETWORK_TEST_LOG'], 'a') as log:
    log.write(json.dumps({'node': os.environ.get('NETWORK_TEST_NODE', '192.0.2.1'),
                         'args': args}) + '\\n')
if args[:2] == ['image', 'inspect']:
    print('sha256:test')
elif args[0] not in ('ps', 'run', 'exec', 'stop'):
    raise SystemExit('Unexpected Docker command: ' + repr(args))
""")
        self.write_tool("ssh", """#!/usr/bin/env python3
import os, subprocess, sys
args = sys.argv[1:]
while args[0] == '-o':
    args = args[2:]
node, command = args
assert node == '192.0.2.2', node
assert command == 'true' or command.startswith('docker '), command
env = dict(os.environ, NETWORK_TEST_NODE=node)
raise SystemExit(subprocess.run(['bash', '-c', command], env=env).returncode)
""")
        self.write_tool("sleep", "#!/bin/bash\nexit 0\n")
        for name in ("ip", "ibdev2netdev", "nc", "scp"):
            self.write_tool(name, "#!/bin/bash\necho 'Unexpected network discovery or copy' >&2\nexit 99\n")

    def write_tool(self, name, source):
        path = self.bin / name
        path.write_text(source)
        path.chmod(0o755)

    def launch(self, *flags, config="", action="exec"):
        self.config.write_text(config)
        self.log.write_text("")
        command = ["bash", str(self.fixture / "launch-cluster.sh"),
                   "--config", str(self.config), "--no-cache-dirs", "-d",
                   *flags, action]
        if action == "exec":
            command += ["vllm", "serve", "test-model", "--host", "0.0.0.0"]
        result = subprocess.run(command, cwd=self.fixture, env=self.env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.output = result.stdout
        self.assertEqual(self.config.read_text(), config)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [call for call in calls if call["args"][0] == "run"]

    @staticmethod
    def container_env(call):
        # Docker uses the last value when a variable is supplied repeatedly.
        args = call["args"]
        return dict(args[i + 1].split("=", 1)
                    for i, arg in enumerate(args) if arg == "-e")

    def assert_solo(self, calls, interface="lo"):
        self.assertEqual(len(calls), 1)
        self.assertIn("Head Node: 127.0.0.1\n", self.output)
        self.assertIn("Starting Head Node on 127.0.0.1...\n", self.output)
        env = self.container_env(calls[0])
        self.assertEqual(env["VLLM_HOST_IP"], "127.0.0.1")
        self.assertEqual(env["NCCL_IB_DISABLE"], "1")
        for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME"):
            self.assertEqual(env[key], interface)
        for key in ("NCCL_IB_HCA", "UCX_NET_DEVICES", "MN_IF_NAME",
                    "OMPI_MCA_btl_tcp_if_include", "RAY_NODE_IP_ADDRESS",
                    "RAY_OVERRIDE_NODE_IP_ADDRESS"):
            self.assertNotIn(key, env)

    def test_solo_host_and_bridge_with_and_without_saved_cluster(self):
        for config in ("", CLUSTER_CONFIG):
            for publish in (False, True):
                with self.subTest(saved_cluster=bool(config), publish=publish):
                    flags = ["--solo"]
                    if publish:
                        flags += ["-p", "8000:8000"]
                    calls = self.launch(*flags, config=config)
                    self.assert_solo(calls)
                    args = calls[0]["args"]
                    if publish:
                        self.assertNotIn("--network", args)
                        self.assertEqual(args[args.index("-p") + 1], "8000:8000")
                    else:
                        self.assertEqual(args[args.index("--network") + 1], "host")

    def test_solo_start_uses_local_defaults(self):
        self.assert_solo(self.launch("--solo", config=CLUSTER_CONFIG, action="start"))

    def test_solo_retains_mesh_container_settings_without_using_cluster_ip(self):
        config = (CLUSTER_CONFIG + "CONTAINER_NCCL_IB_MERGE_NICS=0\n"
                  "CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING=1\n"
                  "CONTAINER_NCCL_NET_PLUGIN=none\n")
        calls = self.launch("--solo", config=config)
        self.assert_solo(calls)
        env = self.container_env(calls[0])
        self.assertEqual(env["NCCL_IB_MERGE_NICS"], "0")
        self.assertEqual(env["NCCL_IB_SUBNET_AWARE_ROUTING"], "1")
        self.assertEqual(env["NCCL_NET_PLUGIN"], "none")

    def test_implicit_solo_uses_local_defaults(self):
        self.assert_solo(self.launch("--nodes", "192.0.2.1", config=CLUSTER_CONFIG))

    def test_explicit_interface_overrides_solo_default(self):
        self.assert_solo(self.launch("--solo", "--eth-if", "custom0",
                                    config=CLUSTER_CONFIG), interface="custom0")

    def test_cli_environment_overrides_solo_defaults(self):
        overrides = {
            "VLLM_HOST_IP": "192.0.2.10",
            "NCCL_SOCKET_IFNAME": "socket0",
            "GLOO_SOCKET_IFNAME": "gloo0",
            "TP_SOCKET_IFNAME": "pipe0",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": "custom_roce",
        }
        flags = ["--solo"]
        for key, value in overrides.items():
            flags += ["-e", f"{key}={value}"]
        calls = self.launch(*flags, config=CLUSTER_CONFIG)
        self.assertEqual(len(calls), 1)
        env = self.container_env(calls[0])
        for key, value in overrides.items():
            self.assertEqual(env[key], value)

    def test_container_config_overrides_solo_defaults(self):
        config = (CLUSTER_CONFIG + "CONTAINER_GLOO_SOCKET_IFNAME=custom0\n"
                  "CONTAINER_VLLM_HOST_IP=192.0.2.10\nCONTAINER_NCCL_IB_DISABLE=0\n")
        calls = self.launch("--solo", config=config)
        self.assertEqual(len(calls), 1)
        env = self.container_env(calls[0])
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "custom0")
        self.assertEqual(env["VLLM_HOST_IP"], "192.0.2.10")
        self.assertEqual(env["NCCL_IB_DISABLE"], "0")

    def test_cluster_keeps_per_node_addresses_and_rdma(self):
        for ray in (False, True):
            with self.subTest(ray=ray):
                calls = self.launch(*(["--ray"] if ray else []), config=CLUSTER_CONFIG)
                self.assertEqual(len(calls), 2)
                self.assertIn("Head Node: 192.0.2.1\n", self.output)
                self.assertIn("Starting Head Node on 192.0.2.1...\n", self.output)
                self.assertEqual({call["node"] for call in calls},
                                 {"192.0.2.1", "192.0.2.2"})
                for call in calls:
                    env = self.container_env(call)
                    for key in ("VLLM_HOST_IP", "RAY_NODE_IP_ADDRESS",
                                "RAY_OVERRIDE_NODE_IP_ADDRESS"):
                        self.assertEqual(env[key], call["node"])
                    for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
                                "TP_SOCKET_IFNAME", "UCX_NET_DEVICES",
                                "MN_IF_NAME", "OMPI_MCA_btl_tcp_if_include"):
                        self.assertEqual(env[key], "cluster0")
                    self.assertEqual(env["NCCL_IB_HCA"], "roce0,roce1")
                    self.assertEqual(env["NCCL_IB_DISABLE"], "0")


if __name__ == "__main__":
    unittest.main()
