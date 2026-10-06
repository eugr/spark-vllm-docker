#!/usr/bin/env python3
"""Offline setup/restore tests: no sudo, SSH connections, or live networking."""

import base64
import contextlib
import importlib.util
import io
import ipaddress
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zlib

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup_cluster_node as worker
from cluster_topology import Topology, validate_launch

spec = importlib.util.spec_from_file_location("setup_cluster", ROOT / "setup-cluster.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
NODES = [f"192.0.2.{11 + i}" for i in range(8)]


def inventory(node, ports=(0, 1), interfaces=None):
    links = [{"ifname": "management0", "flags": ["UP", "LOWER_UP"], "mtu": 1500,
              "addr_info": [{"family": "inet", "local": node, "prefixlen": 24}]}]
    for port, names in worker.PORTS.items():
        for name in names:
            addresses = []
            if interfaces and name in interfaces:
                address, prefix = interfaces[name].split("/")
                addresses = [{"family": "inet", "local": address, "prefixlen": int(prefix)}]
            links.append({"ifname": name, "flags": ["UP", "LOWER_UP"] if port in ports else [],
                          "mtu": 1500, "addr_info": addresses})
    for index, link in enumerate(links):
        link["ifindex"] = index + 1
        link["address"] = f"02:00:00:00:00:{index:02x}"
    return {"ports": list(ports), "links": links, "routes": [], "host_keys": ["ssh-ed25519 AAAAtest"]}


def route_attribute(kind, payload):
    length = len(payload) + 4
    return struct.pack("=HH", length, kind) + payload + bytes(-length % 4)


def route_message(attributes, family=2):
    # Linux RTM_NEWROUTE with a non-default table, protocol, metric, and flags.
    header = struct.pack("=IH HII", 28 + len(attributes), 24, 2, 123, 456)
    return header + struct.pack("=8BI", family, 24 if family == 2 else 64, 0, 0, 0, 4, 0, 1, 0) + attributes


def route_dump(index, family=2):
    attributes = route_attribute(4, struct.pack("=I", index))
    attributes += route_attribute(15, struct.pack("=I", 1234))
    attributes += route_attribute(8, route_attribute(2, struct.pack("=I", 4096)))
    attributes += route_attribute(5, ipaddress.ip_address("10.30.0.1" if family == 2 else "fd00::1").packed)
    return struct.pack("=I", 0x45311224) + route_message(attributes, family)


class RouteDumpTests(unittest.TestCase):
    def test_ipv4_and_ipv6_only_device_references_change(self):
        for family in (2, 10):
            with self.subTest(family=family):
                before = route_dump(2, family)
                self.assertEqual(worker.remap_route_dump(before, {2: 19}), route_dump(19, family))
                self.assertEqual(worker.remap_route_dump(before, {2: 2}), before)

    def test_swapped_iif_oif_and_multiple_messages_do_not_cascade(self):
        def message(iif, oif):
            return route_message(route_attribute(3, struct.pack("=I", iif)) +
                                 route_attribute(4, struct.pack("=I", oif)))
        header = struct.pack("=I", 0x45311224)
        before = header + message(2, 3) + message(3, 2)
        self.assertEqual(worker.remap_route_dump(before, {2: 3, 3: 2}),
                         header + message(3, 2) + message(2, 3))

    def test_multipath_preserves_gateways_weights_and_flags(self):
        def dump(a, b):
            gateway = route_attribute(5, ipaddress.ip_address("10.30.0.1").packed)
            hops = b"".join(struct.pack("=HBBI", 8+len(gateway), 4, weight, index) + gateway
                            for index, weight in ((a, 0), (b, 2)))
            return struct.pack("=I", 0x45311224) + route_message(route_attribute(9, hops))
        self.assertEqual(worker.remap_route_dump(dump(2, 3), {2: 3, 3: 2}), dump(3, 2))

    def test_padding_and_unspecified_device_are_preserved(self):
        attrs = route_attribute(4, struct.pack("=I", 0)) + route_attribute(20, b"\x01")
        # Nonzero padding should be retained too; it is not an interface ID.
        data = struct.pack("=I", 0x45311224) + route_message(attrs[:-3] + b"abc")
        self.assertEqual(worker.remap_route_dump(data, {}), data)
        empty = struct.pack("=I", 0x45311224)
        self.assertEqual(worker.remap_route_dump(empty, {2: 19}), empty)

    def test_unverified_interface_is_never_assumed_to_be_unchanged(self):
        with self.assertRaisesRegex(ValueError, "unverified interface index 99"):
            worker.remap_route_dump(route_dump(99), {2: 19})

    def test_unsupported_embedded_device_or_nexthop_ids_fail_closed(self):
        for kind in (22, 30, 33, 0x4004):
            data = struct.pack("=I", 0x45311224) + route_message(route_attribute(kind, struct.pack("=I", 2)))
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "manual reconciliation"):
                worker.remap_route_dump(data, {2: 19})

    def test_malformed_streams_fail_before_restore(self):
        header = struct.pack("=I", 0x45311224)
        invalid = [b"", b"bad!", header+b"short", route_dump(2)[:-1],
                   header+route_message(b"\x01"),
                   header+route_message(struct.pack("=HH", 0, 4)),
                   header+route_message(route_attribute(4, b"\x01")),
                   header+route_message(route_attribute(9, b"\x01")),
                   header+route_message(route_attribute(9, struct.pack("=HBBI", 4, 0, 0, 2))),
                   header+route_message(b"", family=17)]
        for offset, value in ((4, 9999), (4, 16), (8, 25)):
            data = bytearray(route_dump(2))
            struct.pack_into("=I" if offset == 4 else "=H", data, offset, value)
            invalid.append(data)
        for index, data in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                worker.remap_route_dump(data, {2: 19})


class PlanningTests(unittest.TestCase):
    def test_mtu_default_custom_values_and_invalid_arguments(self):
        self.assertIsNone(setup.parser().parse_args([]).mtu)
        for topology, count in (("direct", 2), ("switch", 4), ("ring", 4)):
            nodes = NODES[:count]
            ports = {} if topology == "ring" else dict.fromkeys(nodes, 0)
            default = setup.make_plan(nodes, topology, ports, "10.20.0.0/16")
            self.assertEqual({p["mtu"] for p in default.values()}, {9000})
            for mtu in (68, 1500, 4096, 9216, 65535):
                with self.subTest(topology=topology, mtu=mtu):
                    args = setup.parser().parse_args(["--mtu", str(mtu)])
                    plan = setup.make_plan(nodes, topology, ports, "10.20.0.0/16", args.mtu)
                    for request in plan.values():
                        network = yaml.safe_load(worker.network_files(request, {})[worker.NETPLAN])["network"]
                        self.assertEqual({p["mtu"] for p in network["ethernets"].values()}, {mtu})
        for value in ("0", "67", "65536", "-1", "1500.5", "auto"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                setup.parser().parse_args(["--mtu", value])
        for value in (None, True, 1500.0, 67, 65536):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "MTU"):
                worker.requested_mtu({"mtu": value})

    def test_saved_modes_reject_mtu_override_before_connecting(self):
        for mode in ("--restore", "--save-env", "--doctor"):
            with self.subTest(mode=mode), patch.object(setup.os, "getuid", return_value=1000), \
                 patch.object(setup.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="fixture")), \
                 patch.object(setup, "check_head") as check_head:
                with self.assertRaisesRegex(ValueError, "omit nodes and networking options"):
                    setup.main([mode, "--mtu", "1500"])
                check_head.assert_not_called()

    def test_env_uses_cx7_for_uniform_direct_and_switch_ports(self):
        for count in (2, 3, 8):
            nodes = NODES[:count]
            for port in (0, 1):
                ports = dict.fromkeys(nodes, port)
                plan = setup.make_plan(nodes, "switch", ports, "10.40.0.0/16")
                values = setup.cluster_env(nodes, "switch", ports, plan, {})
                expected = [f"10.40.0.{11+i}" for i in range(count)]
                self.assertEqual(values["CLUSTER_NODES"].split(","), expected)
                self.assertEqual(values["COPY_HOSTS"].split(","), expected[1:])
                self.assertEqual(values["LOCAL_IP"], expected[0])
                self.assertEqual(values["ETH_IF"], worker.PORTS[port][0])
                self.assertEqual(values["IB_IF"], f"rocep1s0f{port},roceP2p1s0f{port}")
                self.assertNotIn("CLUSTER_LINKS", values)

    def test_env_ring_is_accepted_by_launcher_topology_and_routes_every_worker(self):
        for count in (3, 4, 6, 8):
            nodes = [NODES[0]] + NODES[1:count][::-1]
            plan = setup.make_plan(nodes, "ring", {}, "192.168.0.0/16")
            info = {n: inventory(n) for n in nodes}
            values = setup.cluster_env(nodes, "ring", {}, plan, info)
            graph = Topology.from_env(values)
            self.assertEqual(graph.nodes, nodes)
            self.assertEqual(len(graph.links), 2*count)
            self.assertEqual(set(graph.paths(nodes[0])), set(nodes))
            self.assertEqual(values["ETH_IF"], "management0")
            self.assertNotIn("COPY_HOSTS", values)
            defaults = validate_launch(graph, nodes, f"vllm serve fixture -tp {count}")
            for key, value in defaults.items():
                self.assertEqual(values["CONTAINER_"+key], value)
            self.assertEqual(values.get("CONTAINER_NCCL_ALGO"), "Ring" if count >= 4 else None)

    def test_env_cross_port_uses_management_coordination_and_cx7_copy_routes(self):
        nodes = NODES[:2]
        ports = dict(zip(nodes, (0, 1)))
        plan = setup.make_plan(nodes, "direct", ports, "10.20.0.0/16")
        info = {n: inventory(n) for n in nodes}
        values = setup.cluster_env(nodes, "direct", ports, plan, info)
        self.assertEqual(values["CLUSTER_NODES"], ",".join(nodes))
        self.assertEqual(values["ETH_IF"], "management0")
        self.assertEqual(len(values["IB_IF"].split(",")), 4)
        self.assertEqual(Topology.from_env(values).paths(nodes[0])[nodes[1]], [(nodes[1], "10.20.0.12")])
        info[nodes[1]]["links"][0]["ifname"] = "other0"
        with self.assertRaisesRegex(ValueError, "same management interface"):
            setup.cluster_env(nodes, "direct", ports, plan, info)

    def test_generated_env_round_trips_through_actual_repository_readers(self):
        nodes = NODES[:8]
        values = setup.cluster_env(nodes, "ring", {}, setup.make_plan(nodes, "ring", {}, "10.20.0.0/16"),
                                   {n: inventory(n) for n in nodes})
        spec = importlib.util.spec_from_file_location("env_fixture_runner", ROOT / "run-recipe.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster.env"
            path.write_bytes(worker.merge_env(b"MASTER_PORT=12345\n", values))
            with patch.object(runner, "ENV_FILE", path):
                self.assertEqual(runner.load_env_file(), {"MASTER_PORT": "12345", **values})
            # Sourcing autodiscover only loads the explicit fixture and defines
            # functions. It does not run discovery or read the real .env.
            code = 'CONFIG_FILE="$1"; source "$2"; exec "$3" -c \'import os,json; print(json.dumps({k[7:]:v for k,v in os.environ.items() if k.startswith("DOTENV_")}))\''
            env = {k: v for k, v in os.environ.items() if not k.startswith("DOTENV_")}
            result = subprocess.run(["bash", "-c", code, "fixture", str(path), str(ROOT / "autodiscover.sh"),
                                     sys.executable], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {"MASTER_PORT": "12345", **values})

    def lldp_ring(self, order):
        info = {node: inventory(node) for node in order}
        for index, node in enumerate(order):
            for link in info[node]["links"]:
                link["address"] = f"02:00:00:00:{index:02x}:{link['ifindex']:02x}"
        for index, node in enumerate(order):
            info[node]["lldp"] = {}
            for port, names in worker.PORTS.items():
                peer = order[(index + (1 if port == 0 else -1)) % len(order)]
                remote = [a["address"] for a in info[peer]["links"] if a["ifname"] in worker.PORTS[1-port]]
                local = {a["ifname"]: a["address"] for a in info[node]["links"]}
                for rail, name in enumerate(names):
                    info[node]["lldp"][name] = remote + [local[names[1-rail]]]
        return info

    def test_lldp_detects_physical_ring_without_configured_subnets(self):
        order = [NODES[0], NODES[2], NODES[3], NODES[1]]
        info = self.lldp_ring(order)
        self.assertEqual(setup.choose_topology(NODES[:4], info, "auto", None, False)[:2], ("ring", order))
        with self.assertRaisesRegex(ValueError, "disagrees with LLDP"):
            setup.choose_topology(NODES[:4], info, "ring", None, False)
        self.assertEqual(setup.choose_topology(order, info, "ring", None, False)[1], order)

    def test_lldp_rejects_missing_conflicting_unknown_or_nonreciprocal_peers(self):
        for kind in ("missing", "conflicting", "unknown", "nonreciprocal"):
            with self.subTest(kind=kind):
                info = self.lldp_ring(NODES[:4])
                ports = info[NODES[0]]["lldp"]
                if kind == "missing":
                    ports[worker.PORTS[0][1]] = []
                elif kind == "conflicting":
                    ports[worker.PORTS[0][1]] = ports[worker.PORTS[1][1]]
                elif kind == "unknown":
                    ports[worker.PORTS[0][0]].append("02:ff:ff:ff:ff:ff")
                else:
                    for name in worker.PORTS[1]:
                        info[NODES[1]]["lldp"][name] = [a["address"] for a in info[NODES[3]]["links"]
                                                            if a["ifname"] in worker.PORTS[0]]
                self.assertIsNone(setup.detect_lldp_ring(NODES[:4], info))

    def test_netplan_file_argument_is_an_etc_netplan_yaml_destination(self):
        path = Path("/etc/netplan/40-cx7.yaml")
        self.assertEqual(setup.parser().parse_args(["--netplan-file", str(path)]).netplan_file, path)
        for value in ("40-cx7.yaml", "/tmp/cx7.yaml", "/etc/passwd", "/etc/netplan/../secret.yaml",
                      "/etc/netplan/nested/cx7.yaml", "/etc/netplan/cx7.yml"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.netplan_path(value)

    def test_ip_arguments(self):
        self.assertEqual(setup.parse_nodes([NODES[0] + "," + NODES[1], NODES[2]]), NODES[:3])
        for values in ([NODES[0]], [NODES[0], NODES[0]], ["127.0.0.1", NODES[1]],
                       ["host;touch /tmp/x", NODES[0]], ["::1", NODES[0]], ["224.0.0.1", NODES[0]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                setup.parse_nodes(values)

    def test_dual_cross_port_and_switch(self):
        for count in (2, 3, 8):
            nodes = NODES[:count]
            ports = {n: i % 2 for i, n in enumerate(nodes)}
            plan = setup.make_plan(nodes, "direct" if count == 2 else "switch", ports, "10.20.0.0/16")
            for i, node in enumerate(nodes):
                interfaces = plan[node]["interfaces"]
                self.assertEqual(interfaces[worker.PORTS[ports[node]][0]], f"10.20.0.{11 + i}/24")
                self.assertEqual(interfaces[worker.PORTS[ports[node]][1]], f"10.20.1.{11 + i}/24")
                self.assertEqual(len(plan[node]["ping_targets"]), 2 * (count - 1))
                self.assertEqual(len(plan[node]["ssh_targets"]), 3 * (count - 1))

    def test_ring_rails_closure_and_neighbor_checks(self):
        for count in (3, 4, 6, 8):
            nodes = NODES[:count]
            plan = setup.make_plan(nodes, "ring", {}, "10.20.0.0/16")
            for i, node in enumerate(nodes):
                self.assertEqual(len(plan[node]["interfaces"]), 4)
                self.assertEqual(len(plan[node]["ping_targets"]), 4)
                self.assertEqual(plan[node]["interfaces"][worker.PORTS[0][0]], f"10.20.{2*i}.{11+i}/24")
            self.assertEqual(plan[nodes[0]]["interfaces"][worker.PORTS[1][1]], f"10.20.{2*count-1}.11/24")
            self.assertNotIn("10.20.2.13", plan[nodes[0]]["ssh_targets"] if count > 3 else [])

    def test_pool_validation(self):
        for pool in ("10.20.0.0/24", "10.20.0.0/25", "10.20.0.1/16", "224.0.0.0/16", "0.0.0.0/0"):
            with self.subTest(pool=pool), self.assertRaises(ValueError):
                setup.make_plan(NODES[:4], "ring", {}, pool)
        with self.assertRaisesRegex(ValueError, "management"):
            setup.make_plan(["10.20.0.2", "10.20.0.3"], "direct", {"10.20.0.2": 0, "10.20.0.3": 0}, "10.20.0.0/16")

    def test_auto_single_port(self):
        for count, expected in ((2, "direct"), (3, "switch")):
            nodes = NODES[:count]
            info = {n: inventory(n, ports=(i % 2,)) for i, n in enumerate(nodes)}
            topology, order, ports = setup.choose_topology(nodes, info, "auto", None, False)
            self.assertEqual(topology, expected)
            self.assertEqual(order, nodes)
            self.assertEqual(ports, {n: i % 2 for i, n in enumerate(nodes)})

    def test_ring_autodetection_uses_both_rails_and_cable_order(self):
        order = [NODES[0], NODES[2], NODES[3], NODES[1]]
        plan = setup.make_plan(order, "ring", {}, "10.20.0.0/16")
        info = {n: inventory(n, interfaces=plan[n]["interfaces"]) for n in order}
        topology, detected, _ = setup.choose_topology(NODES[:4], info, "auto", None, False)
        self.assertEqual((topology, detected), ("ring", order))
        # Missing/contradictory second rail must never yield a guessed ring.
        info[NODES[2]]["links"][2]["addr_info"] = []
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            setup.choose_topology(NODES[:4], info, "auto", None, False)

    def test_explicit_topology_and_ambiguous_ports(self):
        info = {n: inventory(n) for n in NODES[:4]}
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            setup.choose_topology(NODES[:4], info, "auto", None, False)
        self.assertEqual(setup.choose_topology(NODES[:4], info, "ring", None, False)[0], "ring")
        with self.assertRaisesRegex(ValueError, "three nodes"):
            setup.choose_topology(NODES[:4], info, "mesh", None, False)
        with self.assertRaisesRegex(ValueError, "Select one"):
            setup.choose_topology(NODES[:4], info, "switch", None, False)
        self.assertEqual(setup.choose_topology(NODES[:4], info, "switch", "0,1,0,1", False)[2],
                         dict(zip(NODES, (0, 1, 0, 1))))
        with patch("builtins.input", return_value="ring"):
            self.assertEqual(setup.choose_topology(NODES[:4], info, "auto", None, True)[0], "ring")

    def test_open_chain_not_detected_as_ring(self):
        plan = setup.make_plan(NODES[:4], "ring", {}, "10.20.0.0/16")
        info = {n: inventory(n, interfaces=plan[n]["interfaces"]) for n in NODES[:4]}
        for link in info[NODES[0]]["links"]:
            if link["ifname"] in worker.PORTS[1]:
                link["addr_info"] = []
        self.assertIsNone(setup.detect_ring(NODES[:4], info))

    def test_explicit_topology_can_bootstrap_administratively_down_ports(self):
        info = {n: inventory(n, ports=()) for n in NODES[:4]}
        self.assertEqual(setup.choose_topology(NODES[:4], info, "ring", None, False)[0], "ring")
        self.assertEqual(setup.choose_topology(NODES[:2], info, "direct", "1", False)[2],
                         {n: 1 for n in NODES[:2]})
        # An up interface with no carrier really has a link problem.
        info[NODES[0]]["links"][1]["flags"] = ["UP"]
        with self.assertRaisesRegex(ValueError, "no carrier"):
            setup.choose_topology(NODES[:4], info, "ring", None, False)


class CommandTests(unittest.TestCase):
    def test_inventory_records_ipv6_state_even_without_addresses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = root / "host.pub"
            key.write_text("ssh-ed25519 AAAA fixture\n")
            for name in worker.PORTS[0] + worker.PORTS[1]:
                (root / name).mkdir()
                (root / name / "disable_ipv6").write_text("1" if name in worker.PORTS[1] else "0")
            info = inventory(NODES[0])
            account = SimpleNamespace(pw_dir=str(root), pw_uid=os.getuid())
            def command(argv, **kwargs):
                return subprocess.CompletedProcess(argv, 0, json.dumps(info["routes"] if "route" in argv else info["links"]), "")
            with patch.object(worker, "IPV6_CONF", root), patch.object(worker.shutil, "which", return_value="fixture"), \
                 patch.object(worker.pwd, "getpwnam", return_value=account), patch.object(Path, "glob", return_value=[key]), \
                 patch.object(worker, "lldp_neighbors", return_value={}), patch.object(worker, "run", side_effect=command):
                links = worker.inventory("fixture")["links"]
            for link in links:
                if link["ifname"] == "management0":
                    self.assertNotIn("ipv6_disabled", link)
                else:
                    self.assertEqual(link["ipv6_disabled"], int(link["ifname"] in worker.PORTS[1]))

    def test_run_passes_seekable_input_to_child(self):
        with tempfile.TemporaryFile() as stream:
            stream.write(b"route fixture")
            stream.seek(0)
            child = "import sys; assert sys.stdin.buffer.read()==b'route fixture'; sys.stdin.seek(0); print('ok')"
            result = worker.run([sys.executable, "-c", child], binary=True, stdin=stream)
        self.assertEqual(result.stdout.strip(), b"ok")

    def test_ip_error_is_actionable_and_netplan_output_stays_private(self):
        with patch.object(worker.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 1, b"", b"Failed to restore: ftell: Illegal seek")):
            with self.assertRaisesRegex(ValueError, "ip -4 route restore.*Illegal seek"):
                worker.run(["ip", "-4", "route", "restore"], binary=True)
        with patch.object(worker.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 1, "", "sensitive netplan diagnostic")):
            with self.assertRaisesRegex(ValueError, r"^netplan failed \(exit 1\)$"):
                worker.run(["netplan", "apply"])

    def test_optional_lldp_inventory_filters_management_and_handles_single_entry(self):
        port = {"port": {"id": {"type": "mac", "value": "02:AA:BB:CC:DD:EE"}}}
        entries = [{"management0": port}, {worker.PORTS[0][0]: port}]
        with patch.object(worker.shutil, "which", return_value="/usr/bin/lldpctl"):
            for shape in (entries, entries[1]):
                result = subprocess.CompletedProcess([], 0, json.dumps({"lldp": {"interface": shape}}), "")
                with patch.object(worker, "run", return_value=result):
                    self.assertEqual(worker.lldp_neighbors(), {worker.PORTS[0][0]: ["02:aa:bb:cc:dd:ee"]})
            with patch.object(worker, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
                self.assertEqual(worker.lldp_neighbors(), {})
        with patch.object(worker.shutil, "which", return_value=None):
            self.assertEqual(worker.lldp_neighbors(), {})


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="spark-setup-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        self.state = self.root / "state"
        self.state.mkdir(mode=0o700)
        self.ipv6 = self.root / "ipv6"
        for name in ["all", "default", "management0", *worker.PORTS[0], *worker.PORTS[1]]:
            (self.ipv6 / name).mkdir(parents=True)
            (self.ipv6 / name / "disable_ipv6").write_text("0\n")
        self.dirs = tuple(self.root / path for path in ("lib/netplan", "etc/netplan", "run/netplan"))
        for directory in self.dirs:
            directory.mkdir(parents=True)
        self.netplan = self.dirs[1] / worker.NETPLAN.name
        self.account = SimpleNamespace(pw_name="fixture", pw_uid=os.getuid(), pw_gid=os.getgid(), pw_dir=str(self.home))
        self.plan = setup.make_plan(NODES[:4], "ring", {}, "10.20.0.0/16")[NODES[0]]
        self.request = {**self.plan, "transaction": "test-transaction", "user": "fixture"}
        self.info = inventory(NODES[0])
        self.calls = []
        for name, value in (("STATE", self.state), ("NETPLAN", self.netplan), ("IPV6_CONF", self.ipv6)):
            patcher = patch.object(worker, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, kwargs in (("netplan_directories", {"return_value": self.dirs}),
                             ("inventory", {"side_effect": lambda user: self.info}),
                             ("run", {"side_effect": self.fake_run})):
            patcher = patch.object(worker, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Tests run unprivileged. Root-owned simulated files have normalized
        # ownership; user SSH files retain their real uid/gid.
        original_write, original_snapshot = worker.atomic_write, worker.snapshot
        def write(path, data, mode=0o600, uid=0, gid=0):
            original_write(path, data, mode, os.getuid(), os.getgid())
        def snapshot(path):
            value = original_snapshot(path)
            if value and not path.is_relative_to(self.home):
                value.update(uid=0, gid=0)
            return value
        for name, effect in (("atomic_write", write), ("snapshot", snapshot)):
            patcher = patch.object(worker, name, side_effect=effect)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.old = self.dirs[1] / "40-cx7.yaml"
        self.original = ("# original comment\nnetwork:\n  version: 2\n  ethernets:\n"
                         "    management0:\n      dhcp4: true\n"
                         "    enp1s0f0np0:\n      addresses: [10.30.0.11/24]\n")
        self.old.write_text(self.original)
        self.old.chmod(0o640)

    def fake_run(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[0] == "ssh-keygen":
            path = Path(argv[argv.index("-f") + 1])
            path.write_text("PRIVATE-KEY-NEVER-RETURNED")
            Path(str(path) + ".pub").write_text("ssh-ed25519 AAAA fixture\n")
        output = json.dumps(self.info["routes"] if "route" in argv else self.info["links"]) if argv[:2] == ["ip", "-j"] else ""
        return subprocess.CompletedProcess(argv, 0, output, "")

    def test_static_routes_are_saved_per_family_interface_and_protocol(self):
        records = [{"dev": worker.PORTS[0][0], "protocol": "static"},
                   {"dev": worker.PORTS[0][0], "protocol": "kernel"},
                   {"dev": worker.PORTS[0][0], "protocol": "dhcp"},
                   {"dev": "management0", "protocol": "static"}]
        def command(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, b"route-stream" if "save" in argv else json.dumps(records), "")
        with patch.object(worker, "run", side_effect=command) as calls:
            saved = worker.save_routes(self.plan["interfaces"])
        self.assertEqual(len(saved), 2)
        self.assertEqual([entry["family"] for entry in saved], ["-4", "-6"])
        self.assertTrue(all(base64.b64decode(entry["data"]) == b"route-stream" for entry in saved))
        saves = [call for call in calls.call_args_list if "save" in call.args[0]]
        self.assertTrue(all(call.args[0][-4:] == ["dev", worker.PORTS[0][0], "proto", "static"] for call in saves))

    def prepare(self):
        result = worker.prepare(self.request, self.account)
        self.assertEqual(result, {"public_key": "ssh-ed25519 AAAA fixture"})
        return worker.Journal(json.loads((self.state / "journal.json").read_text()))

    def test_exact_migration_preserves_management_and_sets_guide_fields(self):
        files = worker.network_files(self.request, worker.preflight(self.request, self.info))
        old = yaml.safe_load(files[self.old])
        self.assertEqual(old["network"]["ethernets"], {"management0": {"dhcp4": True}})
        new = yaml.safe_load(files[self.netplan])
        for name, settings in new["network"]["ethernets"].items():
            self.assertEqual(settings, {"dhcp4": False, "dhcp6": False, "link-local": [],
                                       "mtu": 9000, "addresses": [self.plan["interfaces"][name]]})

    def test_custom_mtu_is_journaled_applied_and_original_runtime_restored(self):
        self.request["mtu"] = 4096
        journal = self.prepare()
        self.assertEqual(journal.state["mtu"], 4096)
        worker.apply_network(self.request, journal)
        network = yaml.safe_load(self.netplan.read_bytes())["network"]
        self.assertEqual({s["mtu"] for s in network["ethernets"].values()}, {4096})
        self.info = inventory(NODES[0], interfaces=self.plan["interfaces"])
        for link in self.info["links"]:
            if link["ifname"] in self.plan["interfaces"]:
                link["mtu"] = 4096
        self.calls.clear()
        worker.restore(journal)
        restored = [c for c in self.calls if c[:3] == ["ip", "link", "set"]]
        self.assertEqual(len(restored), 4)
        self.assertTrue(all(c[-3:] == ["mtu", "1500", "up"] for c in restored))
        self.assertTrue(all(c[4] in self.plan["interfaces"] for c in restored))
        self.assertEqual(self.old.read_text(), self.original)
        self.assertFalse(self.netplan.exists())

    def test_management_mac_match_is_preserved(self):
        document = yaml.safe_load(self.original)
        definition = {"match": {"macaddress": "02:00:00:00:00:00"}, "set-name": "management0", "dhcp4": True}
        document["network"]["ethernets"]["management0"] = definition
        self.old.write_text(yaml.safe_dump(document))
        changed = worker.preflight(self.request, self.info)
        self.assertEqual(yaml.safe_load(changed[self.old])["network"]["ethernets"]["management0"], definition)
        document["network"]["ethernets"]["management0"]["match"]["macaddress"] = self.info["links"][1]["address"]
        self.old.write_text(yaml.safe_dump(document))
        with self.assertRaisesRegex(ValueError, "Ambiguous CX7"):
            worker.preflight(self.request, self.info)

    def test_validate_uses_combined_temporary_tree(self):
        def validate(argv, **kwargs):
            self.assertEqual(argv[:3], ["netplan", "generate", "--root-dir"])
            generated = Path(argv[3]) / "etc/netplan" / self.netplan.name
            self.assertEqual(yaml.safe_load(generated.read_text())["network"]["version"], 2)
            self.assertTrue((generated.parent / self.old.name).exists())
            return subprocess.CompletedProcess(argv, 0, "", "")
        with patch.object(worker, "run", side_effect=validate):
            worker.validate_netplan(worker.network_files(self.request, worker.preflight(self.request, self.info)))
        self.assertFalse(self.netplan.exists())
        self.assertEqual(self.old.read_text(), self.original)

    def test_ambiguous_match_bond_and_runtime_definitions_refused(self):
        for definition in (
            {"ethernets": {"cx": {"match": {"name": "en*"}}}},
            {"ethernets": {"cx": {"match": {"driver": "mlx5_core"}}}},
            {"bonds": {"bond0": {"interfaces": ["enp1s0f0np0"]}}},
            {"vlans": {"vlan0": {"link": "enp1s0f0np0", "id": 12}}},
        ):
            with self.subTest(definition=definition):
                self.old.write_text(yaml.safe_dump({"network": {"version": 2, **definition}}))
                with self.assertRaises(ValueError):
                    worker.netplan_changes(self.plan["interfaces"])
        self.old.unlink()
        for directory in (self.dirs[0], self.dirs[2]):
            path = directory / "vendor.yaml"
            path.write_text(self.original)
            with self.assertRaisesRegex(ValueError, "vendor/runtime"):
                worker.netplan_changes(self.plan["interfaces"])
            path.unlink()

    def test_symlink_refused_and_unrelated_target_preserved(self):
        self.netplan.symlink_to(self.old)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.netplan_changes(self.plan["interfaces"])
        self.netplan.unlink()
        self.netplan.write_text("network:\n  version: 2\n  ethernets:\n    management0:\n      dhcp4: true\n")
        files = worker.network_files(self.request, worker.preflight(self.request, self.info))
        self.assertEqual(yaml.safe_load(files[self.netplan])["network"]["ethernets"]["management0"], {"dhcp4": True})

    def test_custom_existing_netplan_preserves_management_and_restores_original(self):
        request = {**self.request, "netplan_file": str(self.old)}
        worker.prepare(request, self.account)
        journal = worker.Journal(json.loads((self.state / "journal.json").read_text()))
        self.assertEqual(journal.state["netplan_file"], str(self.old))
        worker.apply_network(request, journal)
        self.assertFalse(self.netplan.exists())
        ethernets = yaml.safe_load(self.old.read_text())["network"]["ethernets"]
        self.assertEqual(ethernets["management0"], {"dhcp4": True})
        self.assertEqual(ethernets["enp1s0f0np0"]["addresses"], [self.plan["interfaces"]["enp1s0f0np0"]])
        worker.restore(journal)
        self.assertEqual(self.old.read_text(), self.original)
        self.assertEqual(self.old.stat().st_mode & 0o777, 0o640)

    def test_custom_new_netplan_is_removed_by_restore(self):
        target = self.old.parent / "75-cluster.yaml"
        request = {**self.request, "netplan_file": str(target)}
        worker.prepare(request, self.account)
        journal = worker.Journal(json.loads((self.state / "journal.json").read_text()))
        worker.apply_network(request, journal)
        self.assertTrue(target.exists())
        self.assertFalse(self.netplan.exists())
        worker.restore(journal)
        self.assertFalse(target.exists())
        self.assertEqual(self.old.read_text(), self.original)

    def test_custom_target_cannot_be_shadowed_or_symlinked(self):
        target = self.old.parent / "custom.yaml"
        request = {**self.request, "netplan_file": str(target)}
        target.symlink_to(self.old)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.preflight(request, self.info)
        target.unlink()
        (self.dirs[2] / target.name).write_text("network:\n  version: 2\n")
        with self.assertRaisesRegex(ValueError, "shadow"):
            worker.preflight(request, self.info)

    def test_network_collisions_and_management_nic_refused(self):
        self.info["routes"] = [{"dst": "10.20.0.0/16", "dev": "docker0"}]
        with self.assertRaisesRegex(ValueError, "route"):
            worker.preflight(self.request, self.info)
        self.info["routes"] = []
        self.info["links"][0]["addr_info"].append({"family": "inet", "local": "10.20.0.42", "prefixlen": 24})
        with self.assertRaisesRegex(ValueError, "overlaps"):
            worker.preflight(self.request, self.info)
        self.info = inventory(NODES[0])
        self.info["links"][1]["addr_info"] = self.info["links"][0]["addr_info"]
        with self.assertRaisesRegex(ValueError, "management IP"):
            worker.preflight(self.request, self.info)

    def test_wrong_management_endpoint_refused(self):
        self.info = inventory(NODES[1])
        with self.assertRaisesRegex(ValueError, "uniquely"):
            worker.preflight(self.request, self.info)

    def test_existing_journal_refused_before_starting_new_setup(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "existing setup journal"):
            worker.preflight(self.plan, self.info)

    def test_existing_netplan_target_is_restored(self):
        original = b"# prior config\nnetwork:\n  version: 2\n  ethernets:\n    enp1s0f1np1:\n      mtu: 1500\n"
        self.netplan.write_bytes(original)
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        self.assertNotEqual(self.netplan.read_bytes(), original)
        worker.restore(journal)
        self.assertEqual(self.netplan.read_bytes(), original)

    def test_route_restore_remaps_rebooted_device_ids_using_seekable_streams(self):
        journal = self.prepare()
        journal.state["routes"] = [{"family": "-4", "data": base64.b64encode(route_dump(2)).decode()}]
        before = json.dumps(journal.state, sort_keys=True)
        for record in self.info["links"]:
            record["ifindex"] += 10
        worker.check_restore(journal)
        streams = []
        def command(argv, **kwargs):
            if argv == ["ip", "-4", "route", "restore"]:
                stream = kwargs["stdin"]
                self.assertNotIn("input", kwargs)
                self.assertTrue(stream.seekable())
                self.assertEqual(stream.read(), route_dump(12))
                stream.seek(0)
                self.assertEqual(stream.read(), route_dump(12))
                streams.append(stream)
            return self.fake_run(argv, **kwargs)
        with patch.object(worker, "run", side_effect=command) as calls:
            worker.restore_runtime(journal)
        restore = [call for call in calls.call_args_list if call.args[0] == ["ip", "-4", "route", "restore"]]
        self.assertEqual(len(restore), 1)
        self.assertTrue(streams[0].closed)
        self.assertEqual(json.dumps(journal.state, sort_keys=True), before)

    def test_restore_rechecks_device_indexes_after_netplan_apply(self):
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        journal.state["routes"] = [{"family": "-6", "data": base64.b64encode(route_dump(2, 10)).decode()}]
        def command(argv, **kwargs):
            if argv == ["netplan", "apply"]:
                for item in self.info["links"]:
                    item["ifindex"] += 20
            if argv == ["ip", "-6", "route", "restore"]:
                self.assertEqual(kwargs["stdin"].read(), route_dump(22, 10))
            return self.fake_run(argv, **kwargs)
        with patch.object(worker, "run", side_effect=command):
            worker.restore(journal)
        self.assertIn(["ip", "-6", "route", "restore"], self.calls)
        self.assertEqual(self.old.read_text(), self.original)

    def test_restore_rejects_missing_or_replaced_devices_before_file_changes(self):
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        after = self.netplan.read_bytes()
        original = json.loads(json.dumps(self.info))
        for change in ("missing", "mac", "missing_mac", "unknown_route_device"):
            self.info = json.loads(json.dumps(original))
            for link in self.info["links"]:
                link["ifindex"] += 10
            if change == "missing":
                self.info["links"].pop(1)
            elif change == "mac":
                self.info["links"][1]["address"] = "02:ff:ff:ff:ff:ff"
            elif change == "missing_mac":
                self.info["links"][1].pop("address")
            else:
                journal.state["routes"] = [{"family": "-4", "data": base64.b64encode(route_dump(99)).decode()}]
            self.calls.clear()
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.restore(journal)
            self.assertEqual(self.netplan.read_bytes(), after)
            self.assertNotIn(["netplan", "apply"], self.calls)
            self.assertFalse(any(c[:3] == ["ip", "address", "del"] for c in self.calls))

    def test_corrupt_route_dump_is_rejected_by_restore_preflight(self):
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        after = self.netplan.read_bytes()
        for data in (base64.b64encode(route_dump(2)[:-1]).decode(), "not-base64!"):
            journal.state["routes"] = [{"family": "-4", "data": data}]
            self.calls.clear()
            with self.assertRaises(ValueError):
                worker.restore(journal)
            self.assertEqual(self.netplan.read_bytes(), after)
            self.assertNotIn(["netplan", "apply"], self.calls)

    def test_setup_restore_bytes_permissions_and_private_keys(self):
        ssh = self.home / ".ssh"
        ssh.mkdir(mode=0o700)
        config = ssh / "config"
        config.write_text("# existing\nHost *\n  ConnectTimeout 7\n")
        auth = ssh / "authorized_keys"
        auth.write_text("ssh-ed25519 OLD keep-me")
        originals = {path: path.read_bytes() for path in (self.old, config, auth)}
        journal = self.prepare()
        aliases = [{"addresses": [NODES[0], "10.20.0.11"], "host_keys": ["ssh-ed25519 AAAA"]}]
        worker.install_ssh({**self.request, "aliases": aliases, "public_keys": ["ssh-ed25519 NEW test"]}, self.account, journal)
        self.assertTrue(config.read_text().startswith("# spark-vllm"))
        self.assertIn("Host *\n# existing", config.read_text())
        parsed = subprocess.run(["ssh", "-G", "-F", str(config), NODES[0]], capture_output=True, text=True)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.assertIn("user fixture\n", parsed.stdout)
        worker.apply_network(self.request, journal)
        self.assertEqual(self.netplan.stat().st_mode & 0o777, 0o600)
        self.assertIn(["netplan", "apply"], self.calls)
        worker.restore(journal)
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)
        self.assertEqual(self.old.stat().st_mode & 0o777, 0o640)
        self.assertFalse(self.netplan.exists())
        self.assertFalse((ssh / "spark-vllm-cluster").exists())
        self.assertFalse((self.state / "journal.json").exists())

    def test_restore_conflict_is_checked_before_any_file_changes(self):
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        new = self.netplan.read_bytes()
        self.old.write_text("# later edit\n" + self.old.read_text())
        with self.assertRaisesRegex(ValueError, "Changed since setup"):
            worker.restore(journal)
        self.assertEqual(self.netplan.read_bytes(), new)
        self.assertTrue((self.state / "journal.json").exists())

    def test_write_ahead_journal_handles_failed_write_and_retry(self):
        journal = self.prepare()
        before = self.old.read_bytes()
        original_write = worker.atomic_write
        def fail(path, *args, **kwargs):
            if path == self.old:
                raise OSError("simulated disk failure")
            return original_write(path, *args, **kwargs)
        with patch.object(worker, "atomic_write", side_effect=fail), self.assertRaises(OSError):
            journal.write(self.old, b"changed")
        self.assertEqual(self.old.read_bytes(), before)
        worker.restore(journal)
        self.assertFalse((self.state / "journal.json").exists())

    def test_restore_network_failure_keeps_ssh_and_can_retry(self):
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        def fail(argv, **kwargs):
            if argv == ["netplan", "apply"]:
                raise ValueError("apply failed")
            return self.fake_run(argv, **kwargs)
        with patch.object(worker, "run", side_effect=fail), self.assertRaisesRegex(ValueError, "apply failed"):
            worker.restore(journal)
        self.assertTrue((self.home / ".ssh/spark-vllm-cluster/id_ed25519").exists())
        worker.restore(journal)
        self.assertFalse((self.state / "journal.json").exists())

    def test_restores_runtime_mtu_and_removes_only_added_addresses(self):
        journal = self.prepare()
        self.info = inventory(NODES[0], interfaces=self.plan["interfaces"])
        worker.restore_runtime(journal)
        deletes = [c for c in self.calls if c[:3] == ["ip", "address", "del"]]
        self.assertEqual(len(deletes), 4)
        self.assertTrue(all(c[3] in self.plan["interfaces"].values() for c in deletes))
        mtu = [c for c in self.calls if c[:3] == ["ip", "link", "set"]]
        self.assertEqual(len(mtu), 4)
        self.assertTrue(all(c[-3:] == ["mtu", "1500", "up"] for c in mtu))

    def test_verification_uses_plain_user_ssh_without_existing_masters(self):
        worker.verify(self.request, self.account)
        ssh = [c for c in self.calls if c[0] == "ssh"]
        self.assertEqual(len(ssh), len(self.plan["ssh_targets"]))
        self.assertTrue(all("BatchMode=yes" in c and "ControlPath=none" in c for c in ssh))
        ping = [c for c in self.calls if c[0] == "ping"]
        self.assertEqual(len(ping), 4)
        self.assertTrue(all("8972" in c and "do" in c for c in ping))
        def reject_ssh(argv, **kwargs):
            return subprocess.CompletedProcess(argv, int(argv[0] == "ssh"), "", "Permission denied (publickey).")
        with patch.object(worker, "run", side_effect=reject_ssh):
            with self.assertRaisesRegex(ValueError, "authentication rejected"):
                worker.verify(self.request, self.account)

    def test_link_failure_is_reported_before_ssh(self):
        with patch.object(worker, "run", return_value=subprocess.CompletedProcess([], 1, "", "")) as calls:
            with self.assertRaisesRegex(ValueError, "CX7 reachability failed"):
                worker.verify(self.request, self.account)
        self.assertEqual(calls.call_args.args[0][0], "ping")

    def test_verification_sizes_unfragmented_packets_for_selected_mtu(self):
        for mtu in (None, 68, 1500, 4096, 9216, 65535):
            with self.subTest(mtu=mtu):
                request = {k: v for k, v in self.request.items() if k != "mtu"}
                if mtu is not None:
                    request["mtu"] = mtu
                expected = 9000 if mtu is None else mtu
                self.calls.clear()
                worker.verify(request, self.account)
                pings = [c for c in self.calls if c[0] == "ping"]
                self.assertEqual(len(pings), 4)
                self.assertTrue(all(c[c.index("-s")+1] == str(expected-28) for c in pings))
                self.assertTrue(all(c[c.index("-M")+1] == "do" for c in pings))
                with patch.object(worker, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
                    with self.assertRaisesRegex(ValueError, f"MTU {expected}"):
                        worker.verify(request, self.account)

    def test_missing_address_restores_scope_and_noprefixroute(self):
        name = worker.PORTS[1][0]
        for link in self.info["links"]:
            if link["ifname"] == name:
                link["addr_info"] = [{"family": "inet", "local": "169.254.1.2", "prefixlen": 16,
                                      "scope": "link", "noprefixroute": True}]
        journal = self.prepare()
        self.info = inventory(NODES[0])
        worker.restore_runtime(journal)
        self.assertIn(["ip", "address", "replace", "169.254.1.2/16", "dev", name,
                       "scope", "link", "noprefixroute"], self.calls)

    def test_legacy_restore_reenables_ipv6_after_mtu_before_restoring_addresses(self):
        name = worker.PORTS[1][0]
        addresses = [{"family": "inet6", "local": "fe80::1234", "prefixlen": 64, "scope": "link"},
                     {"family": "inet6", "local": "fd00::1234", "prefixlen": 64, "scope": "global"},
                     {"family": "inet6", "local": "fd00::5678", "prefixlen": 64, "scope": "global", "dynamic": True}]
        old = next(link for link in self.info["links"] if link["ifname"] == name)
        old["addr_info"] = addresses
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        flag = self.ipv6 / name / "disable_ipv6"
        flag.write_text("1\n")
        for other in ("all", "default", "management0"):
            (self.ipv6 / other / "disable_ipv6").write_text("1\n")
        self.info = inventory(NODES[0])
        live = next(link for link in self.info["links"] if link["ifname"] == name)
        live["mtu"] = 1024  # IPv6 cannot be enabled until the old MTU is restored.
        def command(argv, **kwargs):
            if argv[:5] == ["ip", "link", "set", "dev", name]:
                live["mtu"] = int(argv[6])
            if argv[:3] == ["ip", "address", "replace"]:
                self.assertEqual(flag.read_text().strip(), "0")
                self.assertEqual(live["mtu"], 1500)
            return self.fake_run(argv, **kwargs)
        with patch.object(worker, "run", side_effect=command):
            worker.restore(journal)
        for address in addresses[:2]:
            self.assertIn(["ip", "address", "replace", address["local"]+"/64", "dev", name, "scope", address["scope"]], self.calls)
        self.assertFalse(any("fd00::5678/64" in c for c in self.calls))
        self.assertFalse((self.state / "journal.json").exists())
        for other in ("all", "default", "management0"):
            self.assertEqual((self.ipv6 / other / "disable_ipv6").read_text(), "1\n")

    def test_regenerated_ipv6_link_local_address_is_not_replaced(self):
        name = worker.PORTS[1][0]
        old = next(link for link in self.info["links"] if link["ifname"] == name)
        old["addr_info"] = [{"family": "inet6", "local": "fe80::1234", "prefixlen": 64, "scope": "link"}]
        journal = self.prepare()
        self.info = inventory(NODES[0])
        flag = self.ipv6 / name / "disable_ipv6"
        flag.write_text("1\n")
        def command(argv, **kwargs):
            if argv == ["ip", "-j", "address", "show", "dev", name]:
                self.assertEqual(flag.read_text().strip(), "0")
                return subprocess.CompletedProcess(argv, 0, json.dumps([old]), "")
            return self.fake_run(argv, **kwargs)
        with patch.object(worker, "run", side_effect=command):
            worker.restore_runtime(journal)
        self.assertFalse(any("fe80::1234/64" in c for c in self.calls))

    def test_saved_ipv6_state_restores_enabled_and_disabled_addressless_interfaces(self):
        enabled, disabled = worker.PORTS[0]
        for link in self.info["links"]:
            if link["ifname"] in (enabled, disabled):
                link["ipv6_disabled"] = int(link["ifname"] == disabled)
        journal = self.prepare()
        (self.ipv6 / enabled / "disable_ipv6").write_text("1\n")
        (self.ipv6 / disabled / "disable_ipv6").write_text("0\n")
        worker.restore_runtime(journal)
        self.assertEqual((self.ipv6 / enabled / "disable_ipv6").read_text(), "0\n")
        self.assertEqual((self.ipv6 / disabled / "disable_ipv6").read_text(), "1\n")
        # Already restored flags are left alone on retry.
        with patch.object(Path, "write_text", side_effect=AssertionError("unexpected sysctl write")):
            worker.restore_runtime(journal)

    def test_legacy_addressless_ipv6_state_is_not_guessed(self):
        name = worker.PORTS[0][0]
        (self.ipv6 / name / "disable_ipv6").write_text("1\n")
        journal = self.prepare()
        worker.restore_runtime(journal)
        self.assertEqual((self.ipv6 / name / "disable_ipv6").read_text(), "1\n")

    def test_unavailable_ipv6_keeps_journal_and_ssh_for_retry(self):
        name = worker.PORTS[0][0]
        link = next(link for link in self.info["links"] if link["ifname"] == name)
        link["ipv6_disabled"] = 0
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        flag = self.ipv6 / name / "disable_ipv6"
        flag.unlink()
        with self.assertRaisesRegex(ValueError, "IPv6 support is unavailable"):
            worker.restore(journal)
        self.assertTrue((self.state / "journal.json").exists())
        self.assertTrue((self.home / ".ssh/spark-vllm-cluster/id_ed25519").exists())
        flag.write_text("1\n")
        worker.restore(journal)
        self.assertFalse((self.state / "journal.json").exists())

    def env_request(self):
        return {"env_file": str(self.home / ".env"),
                "env_values": setup.cluster_env(NODES[:4], "ring", {},
                                                setup.make_plan(NODES[:4], "ring", {}, "10.20.0.0/16"),
                                                {n: inventory(n) for n in NODES[:4]})}

    def test_env_preserves_unrelated_bytes_and_restores_original_permissions(self):
        path = self.home / ".env"
        original = (b"# custom settings\r\nCONTAINER_HF_TOKEN='fixture-token'\r\nMASTER_PORT=12345\r\n"
                    b"CLUSTER_NODES=old\n CLUSTER_NODES =duplicate\nCOPY_HOSTS=old\n")
        path.write_bytes(original)
        path.chmod(0o640)
        request = self.env_request()
        _, before, _ = worker.env_contents(request, self.account)
        request["env_before"] = worker.env_fingerprint(before)
        journal = self.prepare()
        worker.install_env(request, self.account, journal)
        self.assertTrue(path.read_bytes().startswith(original.split(b"CLUSTER_NODES")[0]))
        self.assertNotIn(b"COPY_HOSTS=", path.read_bytes())
        self.assertEqual(path.read_bytes().count(b"CLUSTER_NODES="), 1)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        # Saving an already-generated file again keeps the original backup.
        _, current, _ = worker.env_contents(request, self.account)
        worker.install_env({**request, "env_before": worker.env_fingerprint(current)}, self.account, journal)
        self.assertEqual(sum(e["path"] == str(path) for e in journal.state["files"]), 1)
        worker.restore(journal)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_env_new_file_removed_and_later_edits_protected_by_restore(self):
        request = self.env_request()
        request["env_before"] = worker.env_fingerprint(None)
        journal = self.prepare()
        worker.install_env(request, self.account, journal)
        path = Path(request["env_file"])
        saved = path.read_bytes()
        path.write_bytes(saved + b"MASTER_PORT=12345\n")
        with self.assertRaisesRegex(ValueError, "Changed since setup"):
            worker.restore(journal)
        self.assertTrue((self.state / "journal.json").exists())
        path.write_bytes(saved)
        worker.restore(journal)
        self.assertFalse(path.exists())

    def test_env_concurrent_edit_and_unsafe_paths_refused(self):
        request = self.env_request()
        request["env_before"] = worker.env_fingerprint(None)
        journal = self.prepare()
        path = Path(request["env_file"])
        path.write_text("MASTER_PORT=12345\n")
        with self.assertRaisesRegex(ValueError, "changed during setup"):
            worker.install_env(request, self.account, journal)
        self.assertEqual(path.read_text(), "MASTER_PORT=12345\n")
        self.assertFalse(any(e["path"] == str(path) for e in journal.state["files"]))
        path.unlink()
        path.symlink_to(self.old)
        with self.assertRaisesRegex(ValueError, "symlink"):
            worker.env_contents(request, self.account)

    def test_env_replaces_stale_ring_settings_and_never_executes_values(self):
        values = {"CLUSTER_NODES": ",".join(NODES[:2]), "LOCAL_IP": NODES[0]}
        original = (b"CONTAINER_NCCL_ALGO=Ring\nCLUSTER_LINKS='[]'\n"
                    b"CONTAINER_HF_TOKEN='$(touch should-not-execute)'\n")
        result = worker.merge_env(original, values)
        self.assertNotIn(b"CLUSTER_LINKS=", result)
        self.assertNotIn(b"CONTAINER_NCCL_ALGO=", result)
        self.assertIn(b"CONTAINER_HF_TOKEN='$(touch should-not-execute)'\n", result)
        self.assertEqual(worker.merge_env(result, values), result)
        for bad in (b"sensitive-invalid-line", b"SECRET='multiline\nsecret\n'"):
            with self.assertRaises(ValueError) as error:
                worker.merge_env(bad, values)
            self.assertNotIn("secret", str(error.exception).lower())

    def doctor_fixture(self):
        journal = self.prepare()
        worker.apply_network(self.request, journal)
        self.info = inventory(NODES[0], interfaces=self.plan["interfaces"])
        for item in self.info["links"]:
            if item["ifname"] in self.plan["interfaces"]:
                item["mtu"] = self.request["mtu"]
        self.info["routes"] = [{"dev": name, "dst": str(ipaddress.ip_interface(cidr).network)}
                               for name, cidr in self.plan["interfaces"].items()]
        for name, value in (("docker_membership", {"exists": True, "gid": 1234, "member": True}),
                            ("docker_access", None)):
            patcher = patch.object(worker, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return journal

    def test_doctor_healthy_report_keeps_private_data_local(self):
        journal = self.doctor_fixture()
        before = (self.state / "journal.json").read_bytes()
        report = worker.doctor_report(self.account, journal)
        self.assertEqual(report["issues"], [])
        self.assertNotIn("PRIVATE-KEY", json.dumps(report))
        self.assertNotIn("after_data", json.dumps(report))
        self.assertEqual((self.state / "journal.json").read_bytes(), before)

    def test_doctor_uses_saved_custom_mtu_and_repairs_drift(self):
        self.request["mtu"] = 1500
        journal = self.doctor_fixture()
        report = worker.doctor_report(self.account, journal)
        self.assertEqual(report["mtu"], 1500)
        self.assertEqual(report["issues"], [])
        self.info["links"][1]["mtu"] = 9000
        report = worker.doctor_report(self.account, journal)
        self.assertTrue(report["network_needed"])
        self.assertTrue(any("MTU 1500" in i["message"] and i["repairable"] for i in report["issues"]))
        saved = self.netplan.read_bytes()
        # Also exercise reconstruction without an after-image, at the saved MTU.
        for entry in journal.state["files"]:
            entry.pop("after_data")
        self.netplan.unlink()
        report = worker.doctor_report(self.account, journal)
        worker.repair_node({**self.request, "fingerprint": report["fingerprint"]}, self.account, journal)
        self.assertEqual(self.netplan.read_bytes(), saved)
        self.assertIn(["netplan", "apply"], self.calls)

    def test_doctor_repairs_missing_file_and_permissions_without_changing_restore_backup(self):
        journal = self.doctor_fixture()
        saved = self.netplan.read_bytes()
        self.netplan.unlink()
        key = self.home / ".ssh/spark-vllm-cluster/id_ed25519"
        key.chmod(0o644)
        report = worker.doctor_report(self.account, journal)
        self.assertEqual(set(report["repair_files"]), {str(self.netplan), str(key)})
        worker.repair_node({**self.request, "fingerprint": report["fingerprint"]}, self.account, journal)
        self.assertEqual(self.netplan.read_bytes(), saved)
        self.assertEqual(key.stat().st_mode & 0o777, 0o600)
        self.assertEqual(worker.doctor_report(self.account, journal)["issues"], [])
        worker.restore(journal)
        self.assertEqual(self.old.read_text(), self.original)
        self.assertFalse(self.netplan.exists())

    def test_doctor_preserves_later_file_edits_and_detects_concurrent_changes(self):
        journal = self.doctor_fixture()
        self.netplan.write_bytes(self.netplan.read_bytes() + b"# later user comment\n")
        report = worker.doctor_report(self.account, journal)
        self.assertTrue(any(not i["repairable"] and i["kind"] == "network" for i in report["issues"]))
        self.calls.clear()
        worker.repair_node({**self.request, "fingerprint": report["fingerprint"]}, self.account, journal)
        self.assertTrue(self.netplan.read_bytes().endswith(b"# later user comment\n"))
        self.assertNotIn(["netplan", "apply"], self.calls)
        self.netplan.write_bytes(self.netplan.read_bytes() + b"# another edit\n")
        with self.assertRaisesRegex(ValueError, "changed after doctor inspection"):
            worker.repair_node({**self.request, "fingerprint": report["fingerprint"]}, self.account, journal)

    def test_doctor_finds_runtime_and_cable_issues(self):
        journal = self.doctor_fixture()
        self.info["links"][1]["addr_info"] = []
        self.info["links"][1]["mtu"] = 1500
        report = worker.doctor_report(self.account, journal)
        self.assertTrue(report["network_needed"])
        worker.repair_node({**self.request, "fingerprint": report["fingerprint"]}, self.account, journal)
        self.assertIn(["netplan", "apply"], self.calls)
        self.info["links"][1]["flags"] = ["UP"]
        report = worker.doctor_report(self.account, journal)
        self.assertTrue(any("carrier" in i["message"] and not i["repairable"] for i in report["issues"]))

    def test_doctor_legacy_netplan_reconstruction_is_checked_against_after_hash(self):
        journal = self.doctor_fixture()
        journal.state.pop("mtu")
        self.assertEqual(worker.doctor_report(self.account, journal)["issues"], [])
        for entry in journal.state["files"]:
            entry.pop("after_data")
        saved = self.netplan.read_bytes()
        self.netplan.unlink()
        report = worker.doctor_report(self.account, journal)
        self.assertIn(str(self.netplan), report["repair_files"])
        worker.repair_node({**self.request, "fingerprint": report["fingerprint"]}, self.account, journal)
        self.assertEqual(self.netplan.read_bytes(), saved)
        self.assertTrue(all("after_data" in e for e in journal.state["files"]))
        key = self.home / ".ssh/spark-vllm-cluster/id_ed25519"
        for entry in journal.state["files"]:
            if entry["path"] == str(key):
                entry.pop("after_data")
        key.unlink()
        report = worker.doctor_report(self.account, journal)
        self.assertTrue(any(str(key) in i["message"] and not i["repairable"] for i in report["issues"]))

    def test_doctor_missing_docker_membership_is_repairable(self):
        journal = self.doctor_fixture()
        with patch.object(worker, "docker_preflight", return_value={"exists": True, "gid": 1234, "member": False}), \
             patch.object(worker, "ensure_docker", return_value={"added": True}) as ensure:
            report = worker.doctor_report(self.account, journal)
            self.assertTrue(any(i["kind"] == "docker" and i["repairable"] for i in report["issues"]))
            result = worker.repair_node({**self.request, "fingerprint": report["fingerprint"]}, self.account, journal)
            self.assertTrue(result["docker_added"])
            ensure.assert_called_once_with(self.account, journal)


class DockerGroupTests(unittest.TestCase):
    def setUp(self):
        self.account = SimpleNamespace(pw_name="fixture", pw_gid=100, pw_uid=501)
        self.membership = {"exists": True, "gid": 999, "member": False}
        self.other_members = []
        self.journal = worker.Journal({"user": "fixture"})
        self.commands = []
        for target, kwargs in (("docker_membership", {"side_effect": lambda account: dict(self.membership)}),
                               ("run", {"side_effect": self.command})):
            patcher = patch.object(worker, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        for target, kwargs in (("save", {}),):
            patcher = patch.object(self.journal, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        for owner, target, kwargs in (
            (worker.shutil, "which", {"return_value": "/fixture/command"}),
            (worker.pwd, "getpwnam", {"return_value": self.account}),
            (worker.pwd, "getpwall", {"return_value": []}),
            (worker.grp, "getgrnam", {"side_effect": lambda name: SimpleNamespace(gr_gid=self.membership["gid"],
                                                                                  gr_mem=self.other_members)}),
        ):
            patcher = patch.object(owner, target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def command(self, argv, **kwargs):
        self.assertIn("docker", self.journal.state)  # Write-ahead intent exists.
        self.journal.save.assert_called()
        self.commands.append(argv)
        if argv[0] == "usermod":
            self.assertEqual(argv, ["usermod", "-a", "-G", "docker", "fixture"])
            self.membership["member"] = True
        elif argv[0] == "gpasswd":
            self.assertEqual(argv, ["gpasswd", "-d", "fixture", "docker"])
            self.membership["member"] = False
        elif argv[0] == "groupadd":
            self.membership.update(exists=True, gid=999)
        elif argv[0] == "groupdel":
            self.membership.update(exists=False, gid=None)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def test_membership_appends_idempotently_and_restore_removes_only_own_addition(self):
        self.assertEqual(worker.ensure_docker(self.account, self.journal), {"added": True})
        self.assertEqual(worker.ensure_docker(self.account, self.journal), {"added": False})
        worker.restore_docker(self.journal)
        worker.restore_docker(self.journal)
        self.assertEqual([c[0] for c in self.commands], ["usermod", "gpasswd"])

    def test_existing_membership_is_never_revoked(self):
        self.membership["member"] = True
        worker.ensure_docker(self.account, self.journal)
        worker.restore_docker(self.journal)
        self.assertEqual(self.commands, [])

    def test_missing_group_is_created_and_removed_only_when_unused(self):
        self.membership.update(exists=False, gid=None)
        worker.ensure_docker(self.account, self.journal)
        worker.restore_docker(self.journal)
        self.assertEqual([c[0] for c in self.commands], ["groupadd", "usermod", "gpasswd", "groupdel"])

    def test_group_adopted_by_another_user_is_preserved(self):
        self.membership.update(exists=False, gid=None)
        worker.ensure_docker(self.account, self.journal)
        self.other_members.append("other-user")
        worker.restore_docker(self.journal)
        self.assertNotIn("groupdel", [c[0] for c in self.commands])

    def test_replaced_group_is_not_modified(self):
        worker.ensure_docker(self.account, self.journal)
        self.membership["gid"] = 888
        with self.assertRaisesRegex(ValueError, "identity changed"):
            worker.restore_docker(self.journal)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            worker.ensure_docker(self.account, self.journal)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name).resolve() / "manifest.json"
        self.args = SimpleNamespace(topology="ring", ports=None, yes=True, dry_run=False,
                                    subnet_pool="10.20.0.0/16", state_file=self.path, netplan_file=None, mtu=None,
                                    env_file=self.path.parent / ".env", no_env=False)
        self.transport = Mock()
        self.transport.call.side_effect = self.call
        self.failed = None

    def call(self, node, action, **request):
        if (node, action) == self.failed:
            raise ValueError("injected failure")
        if action == "inspect":
            return inventory(node)
        if action == "preflight":
            return {"netplan_files": [str(worker.NETPLAN)]}
        if action == "preflight-env":
            return {"env_before": worker.env_fingerprint(None)}
        if action == "prepare":
            return {"public_key": "ssh-ed25519 PUBLIC"}
        if action == "docker":
            return {"added": True}
        return {}

    def run_setup(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            setup.setup(self.args, NODES[:4], self.transport, "fixture")

    def test_dry_run_has_no_mutations_or_manifest(self):
        self.args.dry_run = True
        self.run_setup()
        self.assertFalse(self.path.exists())
        self.assertEqual({call.args[1] for call in self.transport.call.call_args_list}, {"inspect", "preflight", "preflight-env"})

    def test_env_saved_on_head_only_after_all_verification(self):
        self.run_setup()
        calls = self.transport.call.call_args_list
        self.assertEqual(calls[-1].args, (NODES[0], "env"))
        self.assertEqual([c.args for c in calls[-5:-1]], [(n, "verify") for n in NODES[:4]])
        self.assertEqual(json.loads(self.path.read_text())["env_file"], str(self.args.env_file))

    def test_no_env_skips_read_and_write(self):
        self.args.no_env = True
        self.run_setup()
        self.assertFalse(any(c.args[1] in ("env", "preflight-env") for c in self.transport.call.call_args_list))

    def test_setup_ensures_docker_membership_on_every_node_including_head(self):
        self.run_setup()
        calls = self.transport.call.call_args_list
        self.assertEqual([c.args[0] for c in calls if c.args[1] == "docker"], NODES[:4])
        self.assertLess(max(i for i, c in enumerate(calls) if c.args[1] == "prepare"),
                        min(i for i, c in enumerate(calls) if c.args[1] == "docker"))

    def doctor_transport(self):
        plan = setup.make_plan(NODES[:4], "ring", {}, "10.20.0.0/16")
        reports = {n: {"interfaces": plan[n]["interfaces"],
                       "inventory": inventory(n, interfaces=plan[n]["interfaces"]),
                       "issues": [{"kind": "docker", "message": "Missing Docker membership", "repairable": True}],
                       "fingerprint": "fixture"} for n in NODES[:4]}
        self.args.no_env = True
        def call(node, action, **request):
            if action == "doctor-inspect":
                return json.loads(json.dumps(reports[node]))
            if action == "doctor-repair":
                reports[node]["issues"] = []
                return {"docker_added": True}
            if action == "doctor-env":
                return {"needed": True}
            return self.call(node, action, **request)
        self.transport.call.side_effect = call
        return reports

    def run_doctor(self, errors=None):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            setup.doctor(self.args, NODES[:4], {"transaction": "fixture"}, self.transport, errors)

    def test_doctor_repairs_workers_then_head_and_rechecks(self):
        self.doctor_transport()
        self.run_doctor()
        calls = self.transport.call.call_args_list
        self.assertEqual([c.args[0] for c in calls if c.args[1] == "doctor-repair"], NODES[:4][::-1])
        self.assertEqual(sum(c.args[1] == "verify" for c in calls), 8)
        self.assertEqual(sum(c.args[1] == "doctor-inspect" for c in calls), 8)
        self.assertFalse(any(c.args[1] in ("restore", "prepare", "network") for c in calls))

    def test_doctor_passes_saved_mtu_to_repairs_and_verification(self):
        reports = self.doctor_transport()
        for report in reports.values():
            report["mtu"] = 1500
        self.run_doctor()
        calls = [c for c in self.transport.call.call_args_list if c.args[1] in ("doctor-repair", "verify")]
        self.assertEqual(len(calls), 12)
        self.assertTrue(all(c.kwargs["mtu"] == 1500 for c in calls))

    def test_doctor_dry_run_checks_without_mutating(self):
        self.doctor_transport()
        self.args.dry_run = True
        with self.assertRaisesRegex(ValueError, "dry run made no persistent changes"):
            self.run_doctor()
        self.assertEqual({c.args[1] for c in self.transport.call.call_args_list}, {"doctor-inspect", "verify"})

    def test_doctor_manual_and_unreachable_issues_are_not_reported_as_success(self):
        reports = self.doctor_transport()
        for report in reports.values():
            report["issues"] = [{"kind": "file", "message": "Later user edits", "repairable": False}]
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.run_doctor()
        self.assertFalse(any(c.args[1] == "doctor-repair" for c in self.transport.call.call_args_list))
        self.transport.call.reset_mock()
        with self.assertRaisesRegex(ValueError, "could not inspect every"):
            self.run_doctor([(NODES[1], "unreachable")])
        self.assertFalse(any(c.args[0] == NODES[1] for c in self.transport.call.call_args_list))

    def test_doctor_saves_missing_env_only_after_verification(self):
        self.doctor_transport()
        self.args.no_env = False
        self.run_doctor()
        calls = self.transport.call.call_args_list
        env_index = next(i for i, c in enumerate(calls) if c.args[1] == "env")
        self.assertEqual(sum(c.args[1] == "verify" for c in calls[:env_index]), 8)

    def test_env_failure_rolls_back_cluster(self):
        self.failed = (NODES[0], "env")
        with self.assertRaisesRegex(ValueError, "injected failure"):
            self.run_setup()
        self.assertFalse(self.path.exists())
        restored = [c.args[0] for c in self.transport.call.call_args_list if c.args[1] == "restore"]
        self.assertEqual(restored, NODES[:4][::-1])

    def test_save_env_recovers_legacy_setup_and_only_writes_configuration(self):
        order = [NODES[0], NODES[2], NODES[3], NODES[1]]
        plan = setup.make_plan(order, "ring", {}, "192.168.0.0/16")
        def call(node, action, **request):
            if action == "inspect":
                info = inventory(node, interfaces=plan[node]["interfaces"])
                for item in info["links"]:
                    if item["ifname"] in plan[node]["interfaces"]:
                        item["mtu"] = 9000
                return info
            if action == "inspect-setup":
                return {"interfaces": plan[node]["interfaces"]}
            return self.call(node, action, **request)
        self.transport.call.side_effect = call
        with contextlib.redirect_stdout(io.StringIO()):
            setup.save_existing_env(self.args, NODES[:4], {"transaction": "legacy"}, self.transport)
        calls = self.transport.call.call_args_list
        self.assertEqual(calls[-1].args, (NODES[0], "env"))
        self.assertEqual(calls[-1].kwargs["env_values"]["CLUSTER_NODES"], ",".join(order))
        self.assertEqual({c.args[1] for c in calls}, {"inspect", "inspect-setup", "preflight-env", "verify", "env"})
        self.assertEqual([c.args for c in calls[-5:-1]], [(n, "verify") for n in NODES[:4]])
        self.assertFalse(self.path.exists())  # Existing manifest is not replaced.
        self.transport.call.reset_mock()
        self.args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()):
            setup.save_existing_env(self.args, NODES[:4], {"transaction": "legacy"}, self.transport)
        self.assertEqual({c.args[1] for c in self.transport.call.call_args_list},
                         {"inspect", "inspect-setup", "preflight-env"})

    def test_save_env_rejects_partially_restored_configuration(self):
        plan = setup.make_plan(NODES[:4], "ring", {}, "10.20.0.0/16")
        saved = {n: {"interfaces": plan[n]["interfaces"]} for n in NODES[:4]}
        with self.assertRaisesRegex(ValueError, "not active"):
            setup.saved_configuration(NODES[:4], saved, {n: inventory(n) for n in NODES[:4]})

    def test_save_env_uses_saved_custom_mtu_and_rejects_runtime_drift(self):
        plan = setup.make_plan(NODES[:4], "ring", {}, "10.20.0.0/16", 1500)
        info = {n: inventory(n, interfaces=plan[n]["interfaces"]) for n in NODES[:4]}
        def call(node, action, **request):
            if action == "inspect":
                return info[node]
            if action == "inspect-setup":
                return {"interfaces": plan[node]["interfaces"], "mtu": plan[node]["mtu"]}
            return self.call(node, action, **request)
        self.transport.call.side_effect = call
        with contextlib.redirect_stdout(io.StringIO()):
            setup.save_existing_env(self.args, NODES[:4], {"transaction": "fixture"}, self.transport)
        checks = [c for c in self.transport.call.call_args_list if c.args[1] == "verify"]
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(c.kwargs["mtu"] == 1500 for c in checks))
        self.transport.call.reset_mock()
        info[NODES[1]]["links"][1]["mtu"] = 9000
        with self.assertRaisesRegex(ValueError, "not active"):
            setup.save_existing_env(self.args, NODES[:4], {"transaction": "fixture"}, self.transport)
        self.assertFalse(any(c.args[1] == "env" for c in self.transport.call.call_args_list))
        plan[NODES[1]]["mtu"] = 9000
        with self.assertRaisesRegex(ValueError, "MTUs differ"):
            setup.saved_configuration(NODES[:4], plan, info, require_active=False)

    def test_saved_switch_and_cross_port_configuration(self):
        for count in (2, 4):
            nodes = NODES[:count]
            ports = {n: i % 2 for i, n in enumerate(nodes)}
            plan = setup.make_plan(nodes, "switch", ports, "10.20.0.0/16")
            saved = {n: {"interfaces": plan[n]["interfaces"]} for n in nodes}
            info = {n: inventory(n, interfaces=plan[n]["interfaces"]) for n in nodes}
            for n in nodes:
                for link in info[n]["links"]:
                    link["mtu"] = 9000
            topology, order, recovered_ports, recovered = setup.saved_configuration(nodes, saved, info)
            self.assertEqual(topology, "direct" if count == 2 else "switch")
            self.assertEqual((order, recovered_ports), (nodes, ports))
            for n in nodes:
                self.assertEqual(sorted(recovered[n]["ping_targets"]), sorted(plan[n]["ping_targets"]))

    def test_custom_netplan_path_reaches_all_setup_phases_and_manifest(self):
        self.args.netplan_file = Path("/etc/netplan/40-cx7.yaml")
        self.run_setup()
        self.assertEqual(json.loads(self.path.read_text())["netplan_file"], str(self.args.netplan_file))
        for call in self.transport.call.call_args_list:
            if call.args[1] in ("preflight", "prepare", "network"):
                self.assertEqual(call.kwargs["netplan_file"], str(self.args.netplan_file))

    def test_custom_mtu_reaches_all_setup_phases_manifest_and_preview(self):
        self.args.mtu = 1500
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            setup.setup(self.args, NODES[:4], self.transport, "fixture")
        self.assertIn("MTU: 1500", output.getvalue())
        self.assertEqual(json.loads(self.path.read_text())["mtu"], 1500)
        for call in self.transport.call.call_args_list:
            if call.args[1] in ("preflight", "prepare", "network", "verify"):
                self.assertEqual(call.kwargs["mtu"], 1500)

    def test_preflight_failure_does_not_start_setup(self):
        self.failed = (NODES[2], "preflight")
        with self.assertRaises(ValueError):
            self.run_setup()
        self.assertFalse(self.path.exists())

    def test_success_then_reverse_order_restore(self):
        self.run_setup()
        manifest = json.loads(self.path.read_text())
        self.assertEqual(manifest["nodes"], NODES[:4])
        self.assertNotIn("public_key", self.path.read_text())
        setup.restore_cluster(self.transport, NODES[:4], manifest["transaction"], self.path)
        restored = [c.args[0] for c in self.transport.call.call_args_list if c.args[1] == "restore"]
        self.assertEqual(restored, NODES[:4][::-1])
        self.assertFalse(self.path.exists())

    def test_prepare_apply_and_verify_failure_roll_back_all_nodes(self):
        for action in ("prepare", "docker", "ssh", "network", "verify"):
            with self.subTest(action=action):
                self.transport.reset_mock()
                self.failed = (NODES[2], action)
                with self.assertRaisesRegex(ValueError, "injected failure"):
                    self.run_setup()
                restored = [c.args[0] for c in self.transport.call.call_args_list if c.args[1] == "restore"]
                self.assertEqual(restored, NODES[:4][::-1])
                self.assertFalse(self.path.exists())

    def test_restore_preflight_failure_and_partial_retry(self):
        self.run_setup()
        self.failed = (NODES[1], "check-restore")
        with self.assertRaisesRegex(ValueError, "no node was changed"):
            setup.restore_cluster(self.transport, NODES[:4], "id", self.path)
        self.assertFalse(any(c.args[1] == "restore" for c in self.transport.call.call_args_list))
        self.failed = (NODES[1], "restore")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            setup.restore_cluster(self.transport, NODES[:4], "id", self.path)
        self.assertTrue(self.path.exists())
        self.failed = None
        setup.restore_cluster(self.transport, NODES[:4], "id", self.path)
        self.assertFalse(self.path.exists())

    def test_transport_quotes_remote_code_and_keeps_sudo_secret_off_argv(self):
        with tempfile.TemporaryDirectory() as temp:
            transport = setup.Transport(NODES[:2], "fixture", temp)
            transport.passwords[NODES[1]] = "do-not-log-this"
            with patch.object(transport, "command", return_value=subprocess.CompletedProcess([], 0, '{"ok": {}}', "")) as call:
                transport.call(NODES[1], "restore", transaction="id")
            self.assertNotIn("do-not-log-this", str(call.call_args.args))
            self.assertTrue(call.call_args.kwargs["input"].startswith("do-not-log-this\n"))
            code = call.call_args.args[1][-1]
            compile(code, "remote-worker", "exec")
            self.assertIn("StrictHostKeyChecking=", " ".join(transport.ssh(NODES[1])))
            # Restore can use trust established by setup even when the user's
            # original known_hosts was empty during password bootstrap.
            self.assertIn(".ssh/spark-vllm-cluster/known_hosts", " ".join(transport.ssh(NODES[1])))

    def test_transport_payload_survives_sudo_consumed_or_cached_password(self):
        with tempfile.TemporaryDirectory() as temp:
            transport = setup.Transport(NODES[:2], "fixture", temp)
            transport.passwords[NODES[1]] = "test-password"
            transport.source = base64.b64encode(zlib.compress(b"def dispatch(request): return request['transaction']")).decode()
            for consumed in (False, True):
                def execute(node, argv, **kwargs):
                    if consumed:
                        kwargs["input"] = kwargs["input"].split("\n", 1)[1]
                    return subprocess.run([sys.executable, "-c", argv[-1]], **kwargs)
                with patch.object(transport, "command", side_effect=execute):
                    self.assertEqual(transport.call(NODES[1], "restore", transaction="round-trip"), "round-trip")


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="auth-")
        self.addCleanup(self.temp.cleanup)
        self.transport = setup.Transport(NODES[:3], "fixture", self.temp.name)
        self.tty = patch.object(sys.stdin, "isatty", return_value=True)
        self.tty.start()
        self.addCleanup(self.tty.stop)

    @staticmethod
    def sudo_required(node, command, **kwargs):
        return subprocess.CompletedProcess(command, 1 if "-n" in command else 0, "", "")

    def test_shared_password_is_prompted_once_for_all_ssh_and_sudo(self):
        def ssh(node):
            self.assertEqual(self.transport.ssh_prompt(node, "fixture password:", ""), "shared-fixture")
            return 0
        with patch.object(self.transport, "command", side_effect=self.sudo_required) as command, \
                patch.object(self.transport, "connect_ssh", side_effect=ssh), \
                patch.object(setup.getpass, "getpass", return_value="shared-fixture") as prompt:
            for node in NODES[:3]:
                self.transport.connect(node)
        self.assertEqual(prompt.call_count, 1)
        self.assertEqual(self.transport.passwords, dict.fromkeys(NODES[:3], "shared-fixture"))
        self.assertTrue(all("shared-fixture" not in str(call.args) for call in command.call_args_list))

    def test_node_password_and_separate_sudo_password_do_not_replace_common(self):
        self.transport.common_password, self.transport.common_verified = "shared-fixture", True
        def ssh(node):
            self.assertEqual(self.transport.ssh_prompt(node, "fixture password:", ""), "shared-fixture")
            self.assertEqual(self.transport.ssh_prompt(node, "fixture password:", "", retry=True), "node-fixture")
            return 0
        def sudo(node, command, **kwargs):
            code = 0 if kwargs.get("input") == "sudo-fixture\n" else 1
            return subprocess.CompletedProcess(command, code, "", "")
        with patch.object(self.transport, "connect_ssh", side_effect=ssh), \
                patch.object(self.transport, "command", side_effect=sudo), \
                patch.object(setup.getpass, "getpass", side_effect=["node-fixture", "sudo-fixture"]) as prompt:
            self.transport.connect(NODES[1])
        self.assertEqual(prompt.call_count, 2)
        self.assertEqual(self.transport.common_password, "shared-fixture")
        self.assertEqual(self.transport.ssh_passwords[NODES[1]], "node-fixture")
        self.assertEqual(self.transport.passwords[NODES[1]], "sudo-fixture")
        self.assertEqual(self.transport.account_password(NODES[2], "SSH"), "shared-fixture")

    def test_bad_password_retry_is_bounded(self):
        with patch.object(self.transport, "command", return_value=subprocess.CompletedProcess([], 1, "", "")) as command, \
                patch.object(setup.getpass, "getpass", side_effect=["first-fixture", "second-fixture"]) as prompt:
            with self.assertRaisesRegex(ValueError, "sudo authentication failed"):
                self.transport.connect(NODES[0])
        self.assertEqual(prompt.call_count, 2)
        self.assertEqual(command.call_count, 3)  # one noninteractive check, two password attempts

    def test_no_password_prompt_when_keys_and_passwordless_sudo_work(self):
        with patch.object(self.transport, "connect_ssh", return_value=0), \
                patch.object(self.transport, "command", return_value=subprocess.CompletedProcess([], 0, "", "")), \
                patch.object(setup.getpass, "getpass") as prompt:
            for node in NODES[:3]:
                self.transport.connect(node)
        prompt.assert_not_called()
        self.assertIsNone(self.transport.common_password)

    def test_host_confirmation_and_key_passphrases_never_receive_cached_password(self):
        self.transport.common_password = "shared-fixture"
        with patch("builtins.input", return_value="yes") as confirmation, \
                patch.object(setup.getpass, "getpass", return_value="key-fixture") as prompt:
            self.assertEqual(self.transport.ssh_prompt(NODES[1], "Are you sure you want to continue connecting?", ""), "yes")
            self.assertEqual(self.transport.ssh_prompt(NODES[1], "Confirm?", "confirm"), "yes")
            self.assertEqual(self.transport.ssh_prompt(NODES[1], "Enter passphrase for key:", ""), "key-fixture")
            self.assertEqual(self.transport.ssh_prompt(NODES[1], "Security key notification", "none"), "")
        self.assertEqual(confirmation.call_count, 2)
        self.assertEqual(prompt.call_count, 1)
        self.assertEqual(self.transport.common_password, "shared-fixture")
        self.assertFalse(self.transport.ssh_passwords)

    def test_noninteractive_password_requirement_fails_without_prompt(self):
        with patch.object(sys.stdin, "isatty", return_value=False), \
                patch.object(self.transport, "command", side_effect=self.sudo_required), \
                patch.object(setup.getpass, "getpass") as prompt:
            with self.assertRaisesRegex(ValueError, "run interactively"):
                self.transport.connect(NODES[0])
        prompt.assert_not_called()

    def test_expired_sudo_timestamp_reuses_common_password_before_worker(self):
        self.transport.common_password, self.transport.common_verified = "shared-fixture", True
        self.transport.passwords[NODES[0]] = ""
        def command(node, argv, **kwargs):
            if "-n" in argv:
                return subprocess.CompletedProcess(argv, 1, "", "")
            self.assertTrue(kwargs["input"].startswith("shared-fixture\n"))
            return subprocess.CompletedProcess(argv, 0, '{"ok": {}}' if "python3" in argv else "", "")
        with patch.object(self.transport, "command", side_effect=command) as calls, \
                patch.object(setup.getpass, "getpass") as prompt:
            self.assertEqual(self.transport.call(NODES[0], "inspect"), {})
        self.assertEqual(calls.call_count, 3)
        prompt.assert_not_called()

    def test_passwordless_worker_uses_sudo_n_never_an_empty_password_attempt(self):
        self.transport.passwords[NODES[0]] = ""
        with patch.object(self.transport, "command", return_value=subprocess.CompletedProcess([], 0, '{"ok": {}}', "")) as calls:
            self.transport.call(NODES[0], "inspect")
        worker_call = calls.call_args_list[-1]
        self.assertIn("-n", worker_call.args[1])
        self.assertNotIn("-S", worker_call.args[1])

    def test_actual_askpass_process_roundtrip_reuses_memory_password(self):
        # A fake SSH process invokes the real callback. No ssh, sudo, remote
        # server, or IP socket is used; replies travel over a private Unix socket.
        fake = Path(self.temp.name) / "fake_ssh.py"
        fake.write_text('''import os, subprocess, sys
questions = [("Are you sure you want to continue connecting?", "yes"),
             ("fixture password:", "shared-fixture")]
for question, expected in questions:
    result = subprocess.run([os.environ["SSH_ASKPASS"], question],
                            env={**os.environ, "SSH_ASKPASS_PROMPT": ""},
                            capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != expected:
        sys.exit(3)
''')
        original_popen = subprocess.Popen
        def process(command, **kwargs):
            self.assertNotIn("shared-fixture", str(command))
            self.assertNotIn("shared-fixture", str(kwargs["env"]))
            helper = Path(kwargs["env"]["SSH_ASKPASS"])
            self.assertNotIn("shared-fixture", helper.read_text())
            self.assertEqual(helper.stat().st_mode & 0o777, 0o700)
            return original_popen([sys.executable, str(fake), *command[1:]], **kwargs)
        with patch.object(setup.subprocess, "Popen", side_effect=process), \
                patch.object(setup.getpass, "getpass", return_value="shared-fixture") as prompt, \
                patch("builtins.input", return_value="yes"):
            self.assertEqual(self.transport.connect_ssh(NODES[1]), 0)
            self.assertEqual(self.transport.connect_ssh(NODES[2]), 0)
        self.assertEqual(prompt.call_count, 1)
        self.assertFalse((Path(self.temp.name) / "askpass").exists())
        self.assertFalse((Path(self.temp.name) / "askpass.sock").exists())
        self.transport.close()
        self.assertFalse(self.transport.ssh_passwords)
        self.assertIsNone(self.transport.common_password)


if __name__ == "__main__":
    unittest.main()
