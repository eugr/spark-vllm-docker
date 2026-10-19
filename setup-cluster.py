#!/usr/bin/env python3
"""Configure CX7 and mutual SSH from the head node, with durable undo journals."""

import argparse
import base64
import getpass
import ipaddress
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import zlib

from setup_cluster_node import DEFAULT_MTU, NETPLAN, PORTS, mtu_value, netplan_path, requested_mtu

DEFAULT_STATE = Path.home() / ".local/state/spark-vllm-docker/cluster-setup.json"
LEGACY_STATE = Path.home() / ".local/state/spark-vllm/cluster-setup.json"
DEFAULT_ENV = Path(__file__).resolve().with_name(".env")


def state_file_path(path):
    if path is not None:
        return path
    if LEGACY_STATE.exists():
        if DEFAULT_STATE.exists():
            raise ValueError(f"Setup manifests exist at both {DEFAULT_STATE} and {LEGACY_STATE}; "
                             "select the intended manifest with --state-file")
        return LEGACY_STATE
    return DEFAULT_STATE


def parse_nodes(values):
    nodes = [str(ipaddress.IPv4Address(value)) for group in values for value in group.split(",")]
    if not 2 <= len(nodes) <= 244 or len(nodes) != len(set(nodes)):
        raise ValueError("Supply 2–244 distinct management IPv4 addresses, including the head first")
    if any(ipaddress.ip_address(node).is_multicast or ipaddress.ip_address(node).is_unspecified
           or ipaddress.ip_address(node).is_loopback for node in nodes):
        raise ValueError("Use unicast management IPv4 addresses")
    return nodes


def interface_networks(info, name):
    return {str(ipaddress.ip_interface(f"{a['local']}/{a['prefixlen']}").network)
            for item in info["links"] if item["ifname"] == name
            for a in item.get("addr_info", []) if a["family"] == "inet"}


def detect_ring(nodes, inventory):
    """Infer cross-port ring order only if both existing rails agree uniquely."""
    next_node = {}
    for node in nodes:
        matches = []
        for rail in (0, 1):
            local = interface_networks(inventory[node], PORTS[0][rail])
            matches.append([peer for peer in nodes if peer != node and
                            local & interface_networks(inventory[peer], PORTS[1][rail])])
        if len(matches[0]) != 1 or matches[0] != matches[1]:
            return None
        next_node[node] = matches[0][0]
    order = [nodes[0]]
    while next_node[order[-1]] not in order:
        order.append(next_node[order[-1]])
    return order if len(order) == len(nodes) and next_node[order[-1]] == order[0] else None


def detect_lldp_ring(nodes, inventory):
    """Require reciprocal, unambiguous LLDP observations on both CX7 twins.

    Spark twins share a physical port, so each can hear its local twin and
    both remote twins. Match port MACs against the authenticated inventories;
    local-twin advertisements do not represent another cable.
    """
    identities = {}
    for node in nodes:
        for link in inventory[node]["links"]:
            for port, names in PORTS.items():
                if link["ifname"] in names:
                    mac = link.get("address", "").lower()
                    if not mac or mac in identities:
                        return None
                    identities[mac] = (node, port)
    neighbors = {}
    for node in nodes:
        for port, names in PORTS.items():
            twins = []
            for name in names:
                peers = set()
                for mac in inventory[node].get("lldp", {}).get(name, []):
                    peer = identities.get(mac.lower())
                    if peer is None:
                        return None  # Switch, unknown node, or unknown port ID.
                    if peer[0] != node:
                        peers.add(peer)
                if len(peers) != 1:
                    return None
                twins.append(next(iter(peers)))
            if twins[0] != twins[1] or twins[0][1] != 1 - port:
                return None
            neighbors[node, port] = twins[0]
    if any(neighbors.get(peer) != local for local, peer in neighbors.items()):
        return None
    order = [nodes[0]]
    while neighbors[order[-1], 0][0] not in order:
        order.append(neighbors[order[-1], 0][0])
    return order if len(order) == len(nodes) and neighbors[order[-1], 0][0] == order[0] else None


def choose_topology(nodes, inventory, topology, port_values, interactive):
    order = nodes
    physical_ring = detect_lldp_ring(nodes, inventory) if len(nodes) >= 3 else None
    if topology == "auto":
        if port_values or all(len(inventory[node]["ports"]) == 1 for node in nodes):
            topology = "direct" if len(nodes) == 2 else "switch"
            print(f"One selected/active QSFP port per node: using {topology} topology.")
        elif physical_ring:
            topology, order = "ring", physical_ring
            print("LLDP neighbors indicate ring order: " + ", ".join(order))
        elif len(nodes) >= 3 and (detected := detect_ring(nodes, inventory)):
            topology, order = "ring", detected
            print("Existing CX7 subnets indicate ring order: " + ", ".join(order))
        elif interactive:
            print("Carrier alone cannot distinguish a ring from a switch with two connected ports.")
            topology = input("Topology (direct/switch/mesh/ring): ").strip().lower()
        else:
            raise ValueError("Topology is ambiguous; specify --topology and, for a ring, list nodes in cable order")
    if topology not in ("direct", "switch", "mesh", "ring"):
        raise ValueError("Unknown topology")
    if topology == "direct" and len(nodes) != 2:
        raise ValueError("Direct topology requires exactly two nodes")
    if topology == "mesh" and len(nodes) != 3:
        raise ValueError("A full mesh with two QSFP ports supports three nodes; use ring or switch for more nodes")
    if topology in ("mesh", "ring") and len(nodes) < 3:
        raise ValueError("Ring/mesh topology requires at least three nodes")
    if topology in ("mesh", "ring"):
        if port_values:
            raise ValueError("--ports is for direct/switch topology; rings use both ports")
        if physical_ring and order != physical_ring:
            raise ValueError("Listed ring order disagrees with LLDP cabling. Use --topology auto "
                             "or list nodes in port-0 to port-1 order: " + " ".join(physical_ring))
        for node in nodes:
            for port in (0, 1):
                check_port(inventory[node], port, node)
        return "ring", order, {}
    if port_values:
        values = [int(p) for p in port_values.split(",")]
        if len(values) == 1:
            values *= len(nodes)
        if len(values) != len(nodes) or any(p not in (0, 1) for p in values):
            raise ValueError("--ports takes 0/1, or one comma-separated port per node")
        ports = dict(zip(nodes, values))
    else:
        ports = {}
        for node in nodes:
            active = inventory[node]["ports"]
            if len(active) != 1:
                raise ValueError(f"Select one QSFP port on {node} with --ports")
            ports[node] = active[0]
    for node in nodes:
        check_port(inventory[node], ports[node], node)
    # Discovery counts every addressed active twin. Do not silently leave an
    # already-addressed second port in a purported single-port cluster.
    for node in nodes:
        for name in PORTS[1 - ports[node]]:
            if interface_networks(inventory[node], name):
                raise ValueError(f"Unused CX7 port is already addressed on {node}; deconfigure it before direct/switch setup")
    return topology, order, ports


def check_port(info, port, node):
    links = {item["ifname"]: item for item in info["links"]}
    for name in PORTS[port]:
        if name not in links:
            raise ValueError(f"Missing CX7 interface on {node}: {name}")
        flags = links[name]["flags"]
        if "UP" in flags and "LOWER_UP" not in flags:
            raise ValueError(f"CX7 interface has no carrier on {node}: {name}")
        # An administratively down interface cannot report carrier reliably.
        # Explicit topology/port choices let Netplan bring it up after backup;
        # end-to-end verification then checks the actual cable and both rails.


def make_plan(nodes, topology, ports, pool, mtu=DEFAULT_MTU):
    mtu = mtu_value(mtu)
    network = ipaddress.IPv4Network(pool)
    if not any(network.subnet_of(ipaddress.ip_network(private))
               for private in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")):
        raise ValueError("--subnet-pool must be within an RFC1918 private IPv4 range")
    if network.prefixlen > 24:
        raise ValueError("--subnet-pool must contain /24 networks")
    needed = len(nodes) * 2 if topology == "ring" else 2
    if network.num_addresses // 256 < needed:
        raise ValueError(f"--subnet-pool needs at least {needed} distinct /24 subnets")
    subnets = [str(ipaddress.IPv4Network((int(network.network_address) + i * 256, 24)))
               for i in range(needed)]
    if any(ipaddress.ip_address(node) in ipaddress.ip_network(subnet) for node in nodes for subnet in subnets):
        raise ValueError("CX7 subnets overlap management addresses")
    plan = {node: {"interfaces": {}, "mtu": mtu, "nodes": nodes, "subnets": subnets, "management_ip": node,
                   "ping_targets": [], "ssh_targets": [peer for peer in nodes if peer != node]}
            for node in nodes}
    groups = []
    if topology == "ring":
        for i, node in enumerate(nodes):
            peer = nodes[(i + 1) % len(nodes)]
            groups.append([(node, 0), (peer, 1)])
    else:
        groups.append([(node, ports[node]) for node in nodes])
    for cable, group in enumerate(groups):
        for rail in (0, 1):
            subnet = ipaddress.ip_network(subnets[2 * cable + rail])
            endpoints = [(node, PORTS[port][rail], str(subnet.network_address + 11 + nodes.index(node)))
                         for node, port in group]
            for node, interface, address in endpoints:
                plan[node]["interfaces"][interface] = address + "/24"
                for peer, _, target in endpoints:
                    if peer != node:
                        plan[node]["ping_targets"].append([interface, target])
                        plan[node]["ssh_targets"].append(target)
    return plan


def cluster_env(nodes, topology, ports, plan, inventory):
    """Build the same fields consumed by autodiscover.sh and the launcher."""
    def address(node, port, rail=0):
        return plan[node]["interfaces"][PORTS[port][rail]].split("/")[0]

    # A single ETH_IF/IB_IF applies on all ranks. Uniform direct/switch ports
    # can coordinate over CX7, as autodiscovery does. Cross-port connections
    # need the common management interface and the union of selected HCAs.
    common_port = next(iter(ports.values())) if len(set(ports.values())) == 1 else None
    if topology != "ring" and common_port is not None:
        ranks = [address(node, common_port) for node in nodes]
        return {"CLUSTER_NODES": ",".join(ranks), "COPY_HOSTS": ",".join(ranks[1:]),
                "LOCAL_IP": ranks[0], "ETH_IF": PORTS[common_port][0],
                "IB_IF": f"rocep1s0f{common_port},roceP2p1s0f{common_port}"}

    management = set()
    for node in nodes:
        names = [link["ifname"] for link in inventory[node]["links"]
                 if any(a.get("local") == node for a in link.get("addr_info", []))]
        if len(names) != 1:
            raise ValueError(f"Cannot identify the management interface on {node} for .env")
        management.add(names[0])
    if len(management) != 1:
        raise ValueError("Generated .env needs the same management interface name on every node; "
                         "use --no-env and configure launch interfaces separately")
    links = []
    if topology == "ring":
        pairs = [(node, 0, nodes[(i + 1) % len(nodes)], 1) for i, node in enumerate(nodes)]
    else:
        pairs = [(a, ports[a], b, ports[b]) for i, a in enumerate(nodes) for b in nodes[i + 1:]]
    for a, pa, b, pb in pairs:
        for rail in (0, 1):
            links.append({a: address(a, pa, rail), b: address(b, pb, rail)})
    values = {"CLUSTER_NODES": ",".join(nodes), "LOCAL_IP": nodes[0],
              "ETH_IF": management.pop(),
              "IB_IF": "rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1",
              "CLUSTER_LINKS": json.dumps(links, separators=(",", ":"))}
    if topology == "ring":
        values.update(CONTAINER_NCCL_NET_PLUGIN="none", CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING="1",
                      CONTAINER_NCCL_IB_MERGE_NICS="0")
        if len(nodes) >= 4:
            values["CONTAINER_NCCL_ALGO"] = "Ring"
    return values


def saved_configuration(nodes, saved, inventory, *, require_active=True):
    """Recover the plan from node journals, including older head manifests."""
    mtus = {requested_mtu(saved[node]) for node in nodes}
    if len(mtus) != 1:
        raise ValueError("Saved CX7 MTUs differ between nodes; reconcile journals before retrying")
    mtu = mtus.pop()
    topology_info, ports = {}, {}
    for node in nodes:
        interfaces = saved[node]["interfaces"]
        matching = [port for port, names in PORTS.items() if set(names) == set(interfaces)]
        if matching:
            ports[node] = matching[0]
        elif set(interfaces) != set(PORTS[0] + PORTS[1]):
            raise ValueError(f"Unrecognized saved CX7 interfaces on {node}")
        live = {item["ifname"]: item for item in inventory[node]["links"]}
        links = []
        for name, cidr in interfaces.items():
            ip = ipaddress.IPv4Interface(cidr)
            link = live.get(name, {})
            addresses = {f"{a['local']}/{a['prefixlen']}" for a in link.get("addr_info", [])}
            if require_active and (cidr not in addresses or link.get("mtu") != mtu):
                raise ValueError(f"Saved CX7 configuration is not active on {node}: {name}; restore before retrying")
            links.append({"ifname": name, "addr_info": [{"family": "inet", "local": str(ip.ip),
                                                        "prefixlen": ip.network.prefixlen}]})
        topology_info[node] = {"links": links}
    if not ports and len(nodes) >= 3:
        order = detect_ring(nodes, topology_info)
        if not order:
            raise ValueError("Saved CX7 assignments do not form a closed ring")
        topology = "ring"
    elif len(ports) == len(nodes):
        order, topology = nodes, "direct" if len(nodes) == 2 else "switch"
        for rail in (0, 1):
            subnets = {str(ipaddress.ip_interface(saved[n]["interfaces"][PORTS[ports[n]][rail]]).network)
                       for n in nodes}
            if len(subnets) != 1:
                raise ValueError("Saved direct/switch assignments do not share both rail subnets")
    else:
        raise ValueError("Saved setup mixes single-port and ring configurations")
    subnets = sorted({str(ipaddress.ip_interface(cidr).network)
                      for node in nodes for cidr in saved[node]["interfaces"].values()})
    plan = {node: {"interfaces": saved[node]["interfaces"], "mtu": mtu, "ping_targets": [],
                   "nodes": nodes, "subnets": subnets, "management_ip": node,
                   "ssh_targets": [peer for peer in nodes if peer != node]} for node in nodes}
    for node in nodes:
        for name, cidr in plan[node]["interfaces"].items():
            network = ipaddress.ip_interface(cidr).network
            for peer in nodes:
                if peer == node:
                    continue
                for target in plan[peer]["interfaces"].values():
                    ip = ipaddress.ip_interface(target)
                    if ip.network == network:
                        plan[node]["ping_targets"].append([name, str(ip.ip)])
                        plan[node]["ssh_targets"].append(str(ip.ip))
    return topology, order, ports, plan


def prepare_env(args, nodes, topology, ports, plan, inventory, transport):
    if args.no_env:
        return None
    env_path = Path(os.path.abspath((args.env_file or DEFAULT_ENV).expanduser()))
    if env_path == Path(os.path.abspath(args.state_file.expanduser())):
        raise ValueError("--env-file and --state-file must be different files")
    request = {"env_file": str(env_path), "env_values": cluster_env(nodes, topology, ports, plan, inventory)}
    # Only a digest leaves the head; existing .env content can contain secrets.
    request.update(transport.call(nodes[0], "preflight-env", **request))
    return request


def save_existing_env(args, nodes, manifest, transport):
    transaction = manifest["transaction"]
    inventory = {node: transport.call(node, "inspect") for node in nodes}
    saved = {node: transport.call(node, "inspect-setup", transaction=transaction) for node in nodes}
    topology, order, ports, plan = saved_configuration(nodes, saved, inventory)
    if args.env_file is None and manifest.get("env_file"):
        args.env_file = Path(manifest["env_file"])
    request = prepare_env(args, order, topology, ports, plan, inventory, transport)
    print(f"Head configuration: {request['env_file']} ({topology}; {len(nodes)} nodes)")
    if args.dry_run:
        print("Dry run complete; no persistent configuration was changed.")
        return
    for node in nodes:
        print(f"Verifying mutual SSH and both CX7 rails from {node}…", flush=True)
        transport.call(node, "verify", **plan[node])
    transport.call(nodes[0], "env", transaction=transaction, **request)
    print(f"Saved launch configuration to {request['env_file']}; --restore includes this file.")


def doctor(args, nodes, manifest, transport, connection_errors=None):
    transaction = manifest["transaction"]
    if "env_file" in manifest and manifest["env_file"] is None and args.env_file is None:
        args.no_env = True  # Preserve an explicit --no-env choice from setup.
    reports, errors = {}, list(connection_errors or [])
    failed_nodes = {node for node, _ in errors}
    for node in nodes:
        if node in failed_nodes:
            continue
        print(f"Checking saved configuration and Docker access on {node}…", flush=True)
        try:
            reports[node] = transport.call(node, "doctor-inspect", transaction=transaction)
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            errors.append((node, str(error)))
    for node, message in errors:
        print(f"{node}: {message}")
    for node, report in reports.items():
        for issue in report["issues"]:
            print(f"{node}: {'FIX' if issue['repairable'] else 'MANUAL'}: {issue['message']}")
    head_group = reports.get(nodes[0], {}).get("docker", {})
    if head_group.get("member") and head_group["gid"] not in [os.getgid(), *os.getgroups()]:
        print("The current head login has stale groups; log out and back in before running Docker.")
    if errors:
        raise ValueError("Doctor could not inspect every saved node; restore management access/journals and retry")
    inventory = {n: reports[n]["inventory"] for n in nodes}
    topology, order, ports, plan = saved_configuration(nodes, reports, inventory, require_active=False)
    physical = detect_lldp_ring(nodes, inventory) if topology == "ring" else None
    if physical and physical != order:
        raise ValueError("Cabling no longer matches the saved ring; restore/reconfigure in physical cable order")
    # A missing .env from an older setup can be generated without needing an
    # after-image. Already-journaled files are repaired on their owning node.
    env_request = None
    env_error = None
    if not args.no_env:
        if args.env_file is None and manifest.get("env_file"):
            args.env_file = Path(manifest["env_file"])
        try:
            request = prepare_env(args, order, topology, ports, plan, inventory, transport)
            status = transport.call(nodes[0], "doctor-env", transaction=transaction, **request)
            if status["needed"]:
                env_request = request
                print(f"{nodes[0]}: FIX: Save current launch configuration to {request['env_file']}")
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            env_error = str(error)
            print(f"{nodes[0]}: MANUAL: {env_error}")
    def verify_all():
        failures = []
        for node in nodes:
            print(f"Checking mutual SSH and both CX7 rails from {node}…", flush=True)
            try:
                transport.call(node, "verify", **plan[node])
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                failures.append(f"{node}: {error}")
        for message in failures:
            print(message)
        return failures
    failures = verify_all()
    repairs = [n for n in nodes if any(i["repairable"] for i in reports[n]["issues"])]
    manual = [f"{n}: {i['message']}" for n in nodes for i in reports[n]["issues"] if not i["repairable"]]
    if env_error:
        manual.append(env_error)
    if args.dry_run:
        if repairs or env_request or manual or failures:
            raise ValueError("Doctor found issues; dry run made no persistent changes")
        print("Doctor: all saved configuration, CX7 links, SSH, and Docker checks passed.")
        return
    if repairs or env_request:
        if not args.yes:
            if not sys.stdin.isatty():
                raise ValueError("Doctor found repairable issues; use --yes to apply the displayed repairs")
            if input("Apply the listed doctor repairs? [y/N] ").strip().lower() not in ("y", "yes"):
                raise ValueError("Doctor repairs cancelled; no configuration was changed")
        added = False
        repair_errors = []
        for node in reversed(nodes):
            if node not in repairs:
                continue
            print(f"Repairing {node}…", flush=True)
            try:
                result = transport.call(node, "doctor-repair", transaction=transaction,
                                        fingerprint=reports[node]["fingerprint"], **plan[node])
                added = added or result["docker_added"]
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                repair_errors.append(f"{node}: {error}")
        if added:
            print("Docker membership updated. Log out and back in before running Docker in existing sessions.")
        failures = verify_all()
        if env_request and not failures:
            try:
                transport.call(nodes[0], "env", transaction=transaction, **env_request)
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                repair_errors.append(str(error))
        manual = repair_errors
        for node in nodes:
            try:
                report = transport.call(node, "doctor-inspect", transaction=transaction)
                manual.extend(f"{node}: {i['message']}" for i in report["issues"])
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                manual.append(f"{node}: {error}")
        if env_error:
            try:
                request = prepare_env(args, order, topology, ports, plan, inventory, transport)
                status = transport.call(nodes[0], "doctor-env", transaction=transaction, **request)
                if status["needed"]:
                    manual.append("Launch configuration still needs saving; run --save-env")
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                manual.append(str(error))
    if manual or failures:
        raise ValueError("Doctor has unresolved issues (journals retained):\n" + "\n".join(manual + failures))
    print("Doctor: all saved configuration, CX7 links, SSH, and Docker checks passed.")


class Transport:
    """Temporary SSH masters allow interactive bootstrap without sshpass.

    Account passwords are kept in memory. SSH obtains them through an askpass
    helper over a private local socket; sudo reads them on stdin over SSH.
    Passwords never enter argv, environment variables, or files.
    """
    def __init__(self, nodes, user, directory):
        self.nodes, self.user = nodes, user
        self.directory = Path(directory)
        self.passwords = {}
        self.ssh_passwords = {}
        self.common_password = None
        self.common_verified = False
        self.connected = []
        self.source = base64.b64encode(zlib.compress(
            Path(__file__).with_name("setup_cluster_node.py").read_bytes())).decode()

    def ssh(self, node):
        known = (f'{self.directory}/known_hosts {Path.home()}/.ssh/known_hosts '
                 f'{Path.home()}/.ssh/known_hosts2 {Path.home()}/.ssh/spark-vllm-cluster/known_hosts')
        return ["ssh", "-o", "ControlMaster=auto", "-o", "ControlPersist=600",
                "-o", f"ControlPath={self.directory}/s{self.nodes.index(node)}",
                "-o", f"UserKnownHostsFile={known}",
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
                "-o", "ServerAliveCountMax=3", "-o", "NumberOfPasswordPrompts=1",
                "-l", self.user, node]

    def command(self, node, argv, **kwargs):
        command = argv if node == self.nodes[0] else self.ssh(node) + [shlex.join(argv)]
        return subprocess.run(command, **kwargs)

    def account_password(self, node, purpose, *, fresh=False):
        preferred, fallback = ((self.ssh_passwords, self.passwords) if purpose == "SSH"
                               else (self.passwords, self.ssh_passwords))
        candidate = preferred.get(node) or fallback.get(node) or self.common_password
        if candidate is None or fresh:
            if not sys.stdin.isatty():
                raise ValueError(f"{node} needs a {purpose} password; run interactively")
            prompt = (f"{purpose} password for {self.user}@{node} (previous password rejected): " if fresh else
                      f"Cluster SSH/sudo password for {self.user} (reused across these nodes): ")
            candidate = getpass.getpass(prompt)
            if not candidate or any(char in candidate for char in "\r\n\0"):
                raise ValueError("A nonempty, single-line password is required")
            if not self.common_verified:
                self.common_password = candidate
        preferred[node] = candidate
        return candidate

    def ssh_prompt(self, node, prompt, hint, *, retry=False):
        if hint == "none":
            print(prompt, file=sys.stderr)
            return ""
        lower = prompt.lower()
        if hint == "confirm" or "are you sure you want to continue connecting" in lower:
            # A host trust decision must never receive an account password.
            return input(prompt + " ")
        if lower.strip().endswith("password:"):
            return self.account_password(node, "SSH", fresh=retry)
        # Encrypted private-key passphrases and security-key prompts are distinct
        # from login passwords; do not cache or reuse them as account passwords.
        return getpass.getpass(prompt)

    def connect_ssh(self, node):
        if not sys.stdin.isatty():
            return self.command(node, ["true"], timeout=180).returncode
        helper = self.directory / "askpass"
        address = self.directory / "askpass.sock"
        helper.write_text("#!/bin/sh\nexec " + shlex.join([
            sys.executable, str(Path(__file__).resolve()), "--ssh-askpass"]) + ' "$@"\n')
        helper.chmod(0o700)
        environment = {**os.environ, "SSH_ASKPASS": str(helper), "SSH_ASKPASS_REQUIRE": "force",
                       "SPARK_SETUP_ASKPASS_SOCKET": str(address)}
        # These options precede the usual noninteractive command options.
        command = ["ssh", "-o", "BatchMode=no", "-o", "StrictHostKeyChecking=ask",
                   "-o", "NumberOfPasswordPrompts=2", "-o", "KbdInteractiveAuthentication=no",
                   *self.ssh(node)[1:], "true"]
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(address))
                address.chmod(0o600)
                server.listen(1)
                server.settimeout(0.2)
                process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL)
                try:
                    deadline = time.monotonic() + 180
                    password_prompts = 0
                    while process.poll() is None:
                        if time.monotonic() >= deadline:
                            raise ValueError(f"Management SSH timed out on {node}")
                        try:
                            connection, _ = server.accept()
                        except socket.timeout:
                            continue
                        with connection:
                            connection.settimeout(5)
                            with connection.makefile("rb") as stream:
                                question = json.loads(stream.readline(65536))
                            answer = self.ssh_prompt(node, question["prompt"], question["hint"],
                                                     retry=password_prompts > 0)
                            if (question["prompt"].lower().strip().endswith("password:")
                                    and question["hint"] not in ("confirm", "none")):
                                password_prompts += 1
                            connection.sendall(json.dumps({"answer": answer}).encode() + b"\n")
                        deadline = time.monotonic() + 180
                    if process.returncode == 0 and self.ssh_passwords.get(node) == self.common_password:
                        self.common_verified = self.common_password is not None
                    return process.returncode
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
        finally:
            address.unlink(missing_ok=True)
            helper.unlink(missing_ok=True)

    def connect(self, node):
        if node != self.nodes[0]:
            print(f"Connecting to {self.user}@{node}; verify its host fingerprint if prompted.", flush=True)
            if self.connect_ssh(node):
                raise ValueError(f"Cannot establish management SSH to {node}")
            self.connected.append(node)
        self.authenticate_sudo(node)

    def authenticate_sudo(self, node):
        result = self.command(node, ["sudo", "-n", "true"], capture_output=True, text=True, timeout=30)
        self.passwords[node] = ""
        if result.returncode:
            for attempt in range(2):
                password = self.account_password(node, "sudo", fresh=attempt > 0)
                result = self.command(node, ["sudo", "-S", "-p", "", "true"], input=password + "\n",
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    if password == self.common_password:
                        self.common_verified = True
                    break
            else:
                raise ValueError(f"sudo authentication failed on {node}")

    def call(self, node, action, **request):
        if not self.passwords.get(node):
            # An earlier sudo timestamp may have expired while other nodes were
            # being authenticated. Recheck without submitting an empty password.
            self.authenticate_sudo(node)
        request = {**request, "action": action, "user": self.user}
        payload = base64.b64encode(zlib.compress(json.dumps(request).encode())).decode()
        code = ("import base64,zlib,json,sys; "
                f"exec(zlib.decompress(base64.b64decode({self.source!r}))); "
                "request=json.loads(zlib.decompress(base64.b64decode(sys.stdin.buffer.read().splitlines()[-1])));\n"
                "try:\n print(json.dumps({'ok': dispatch(request)}))\n"
                "except Exception as error:\n print(json.dumps({'error': str(error)})); sys.exit(1)\n")
        timeout = 300 if action != "verify" else 300 + 160 * (len(request["ssh_targets"]) + len(request["ping_targets"]))
        result = self.command(node, ["sudo", "-S" if self.passwords[node] else "-n",
                                     "-p", "", "python3", "-c", code],
                              # sudo may consume the first line, or leave it for
                              # the worker when credentials are already cached.
                              # The final line is always the request; keeping it
                              # on stdin avoids argv limits on larger clusters.
                              input=self.passwords[node] + "\n" + payload + "\n", capture_output=True,
                              text=True, timeout=timeout)
        try:
            response = json.loads(result.stdout)
        except ValueError as exc:
            raise ValueError(f"{action} failed on {node} (no valid response; check sudo/Python prerequisites)") from exc
        if result.returncode or "error" in response:
            raise ValueError(f"{node}: {response.get('error', action + ' failed')}")
        return response["ok"]

    def close(self):
        for node in self.connected:
            try:
                subprocess.run(self.ssh(node)[:-1] + ["-O", "exit", node],
                               capture_output=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.passwords.clear()
        self.ssh_passwords.clear()
        self.common_password = None
        self.common_verified = False


def ssh_askpass():
    """Internal OpenSSH callback: exchange prompts/answers over a private socket."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(os.environ["SPARK_SETUP_ASKPASS_SOCKET"])
        question = {"prompt": sys.argv[2] if len(sys.argv) > 2 else "",
                    "hint": os.environ.get("SSH_ASKPASS_PROMPT", "")}
        connection.sendall(json.dumps(question).encode() + b"\n")
        with connection.makefile("rb") as stream:
            answer = json.loads(stream.readline(65536))["answer"]
        print(answer)


def check_head(nodes):
    result = subprocess.run(["ip", "-j", "-4", "address", "show"], capture_output=True, text=True, check=True)
    local = {a["local"] for link in json.loads(result.stdout) for a in link.get("addr_info", [])}
    if nodes[0] not in local or set(nodes[1:]) & local:
        raise ValueError("Run on the head; include exactly one of its management addresses first")


def save_manifest(path, manifest):
    from setup_cluster_node import no_symlinks
    no_symlinks(path)
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def restore_cluster(transport, nodes, transaction, state_path, *, best_effort=False):
    errors = []
    ready = []
    for node in reversed(nodes):
        try:
            transport.call(node, "check-restore", transaction=transaction)
            ready.append(node)
        except Exception as exc:
            errors.append(str(exc))
    if errors and not best_effort:
        raise ValueError("Restore preflight failed; no node was changed:\n" + "\n".join(errors))
    for node in ready:
        try:
            print(f"Restoring {node}…", flush=True)
            transport.call(node, "restore", transaction=transaction)
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("Restore is incomplete; journals were retained. Retry --restore.\n" + "\n".join(errors))
    state_path.unlink()


def setup(args, nodes, transport, user):
    inventory = {node: transport.call(node, "inspect") for node in nodes}
    topology, order, ports = choose_topology(nodes, inventory, args.topology, args.ports,
                                            sys.stdin.isatty() and not args.yes)
    mtu = args.mtu if args.mtu is not None else DEFAULT_MTU
    plan = make_plan(order, topology, ports, args.subnet_pool, mtu)
    target = str(netplan_path(args.netplan_file or NETPLAN))
    for request in plan.values():
        request["netplan_file"] = target
    env_request = prepare_env(args, order, topology, ports, plan, inventory, transport)
    print(f"Topology: {topology}; MTU: {mtu}; current user: {user}")
    if topology == "ring":
        print("Cabling: port 0 of each listed node -> port 1 of the next, including the closing cable.")
        print("Ring order: " + " -> ".join(order + order[:1]))
    for node in order:
        changes = transport.call(node, "preflight", **plan[node])
        print(f"\n{node}:")
        for interface, address in plan[node]["interfaces"].items():
            print(f"  {interface}: {address}")
        print("  Netplan files: " + ", ".join(changes["netplan_files"]))
    print("\nSSH: create one local key per node, authorize all cluster keys, and trust verified node host keys.")
    print(f"Docker: ensure {user} belongs to the docker group on every node, including the head.")
    if env_request:
        print(f"Head configuration: {env_request['env_file']} (save after verification; preserve unrelated settings)")
    print(f"Restore manifest: {args.state_file}")
    if args.dry_run:
        print("Dry run complete; no persistent configuration was changed.")
        return
    if not args.yes:
        if not sys.stdin.isatty():
            raise ValueError("Use --yes to apply this plan without an interactive terminal")
        if input("Apply this cluster setup plan? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return
    transaction = str(uuid.uuid4())
    save_manifest(args.state_file, {"version": 1, "nodes": nodes, "user": user,
                                   "transaction": transaction, "netplan_file": target, "mtu": mtu,
                                   "env_file": env_request["env_file"] if env_request else None})
    try:
        keys = []
        for node in nodes:
            print(f"Preparing journal and SSH key on {node}…", flush=True)
            keys.append(transport.call(node, "prepare", transaction=transaction, **plan[node])["public_key"])
        docker_added = False
        for node in nodes:
            print(f"Checking Docker group membership on {node}…", flush=True)
            result = transport.call(node, "docker", transaction=transaction)
            docker_added = docker_added or result["added"]
        aliases = [{"addresses": [node] + [a.split("/")[0] for a in plan[node]["interfaces"].values()],
                    "host_keys": inventory[node]["host_keys"]} for node in nodes]
        for node in nodes:
            transport.call(node, "ssh", transaction=transaction, public_keys=keys, aliases=aliases)
        for node in reversed(nodes):
            print(f"Applying CX7 networking on {node}…", flush=True)
            transport.call(node, "network", transaction=transaction, **plan[node])
        for node in nodes:
            print(f"Verifying mutual SSH and both CX7 rails from {node}…", flush=True)
            transport.call(node, "verify", **plan[node])
        if env_request:
            transport.call(nodes[0], "env", transaction=transaction, **env_request)
            print(f"Saved launch configuration to {env_request['env_file']}")
    except (Exception, KeyboardInterrupt):
        print("Setup failed/interrupted; attempting rollback on every node.", file=sys.stderr, flush=True)
        try:
            restore_cluster(transport, nodes, transaction, args.state_file, best_effort=True)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
        raise
    print("Cluster networking and mutual passwordless SSH are configured and verified.")
    if docker_added:
        print("Docker membership updated. Log out and back in before running Docker in existing sessions.")
    print("To revert: ./setup-cluster.sh --restore --state-file " + shlex.quote(str(args.state_file)))


def parser():
    result = argparse.ArgumentParser(description=__doc__, epilog=(
        "Include the head management IP first. For mesh/ring, list nodes in cable order: "
        "port 0 -> next node's port 1. See docs/NETWORKING.md."))
    result.add_argument("nodes", nargs="*", help="management IPv4 addresses, space- or comma-separated")
    result.add_argument("--topology", choices=("auto", "direct", "switch", "mesh", "ring"), default="auto")
    result.add_argument("--ports", help="direct/switch: 0 or 1 for all nodes, or one port per node in input order")
    result.add_argument("--subnet-pool", default="10.20.0.0/16", help="pool of /24 CX7 subnets (default: %(default)s)")
    result.add_argument("--mtu", type=mtu_value, metavar="BYTES",
                        help=f"MTU on all selected CX7 interfaces, 68–65535 (default: {DEFAULT_MTU})")
    result.add_argument("--netplan-file", type=netplan_path,
                        help=f"destination .yaml file in /etc/netplan on every node (default: {NETPLAN})")
    env_options = result.add_mutually_exclusive_group()
    env_options.add_argument("--env-file", type=Path, help="head .env destination (default: repository .env)")
    env_options.add_argument("--no-env", action="store_true", help="skip saving head launch configuration")
    result.add_argument("--dry-run", action="store_true", help="inspect and validate, without persistent changes")
    result.add_argument("--yes", action="store_true", help="apply the displayed plan without confirmation")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--restore", action="store_true", help="restore the setup recorded in --state-file")
    mode.add_argument("--save-env", action="store_true", help="verify an existing saved setup and save its head .env")
    mode.add_argument("--doctor", action="store_true", help="check a saved setup and offer repairs; --dry-run checks only")
    result.add_argument("--state-file", type=Path,
                        help=f"head-node restore manifest (default: {DEFAULT_STATE}; detects existing legacy state)")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.version_info < (3, 10):
        raise ValueError("Python 3.10+ is required")
    if os.getuid() == 0:
        raise ValueError("Run as the current login user, without sudo; the helper requests sudo when needed")
    user = pwd.getpwuid(os.getuid()).pw_name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", user):
        raise ValueError("Unsupported login username")
    if args.restore or args.save_env or args.doctor:
        if (args.nodes or args.ports or args.topology != "auto" or args.netplan_file or args.mtu is not None
                or (args.no_env and not args.doctor)
                or (args.restore and (args.dry_run or args.env_file))):
            raise ValueError("--restore/--save-env/--doctor use the saved node list; omit nodes and networking options")
        args.state_file = state_file_path(args.state_file)
        manifest = json.loads(args.state_file.read_text())
        if manifest["version"] != 1 or manifest["user"] != user:
            raise ValueError("Restore manifest version/user mismatch")
        nodes = parse_nodes(manifest["nodes"])
    else:
        nodes = parse_nodes(args.nodes)
        args.state_file = state_file_path(args.state_file)
        if args.state_file.exists():
            raise ValueError(f"Setup already recorded at {args.state_file}; use --restore before another setup")
    check_head(nodes)
    with tempfile.TemporaryDirectory(prefix="spark-setup-") as directory:
        transport = Transport(nodes, user, directory)
        try:
            connection_errors = []
            for node in nodes:
                try:
                    transport.connect(node)
                except (ValueError, OSError, subprocess.SubprocessError) as error:
                    if not args.doctor:
                        raise
                    connection_errors.append((node, str(error)))
            if args.restore:
                restore_cluster(transport, nodes, manifest["transaction"], args.state_file)
                print("Restored the saved network and SSH configuration on every node.")
            elif args.save_env:
                save_existing_env(args, nodes, manifest, transport)
            elif args.doctor:
                doctor(args, nodes, manifest, transport, connection_errors)
            else:
                setup(args, nodes, transport, user)
        finally:
            transport.close()


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--ssh-askpass"]:
            ssh_askpass()
        else:
            main()
    except KeyboardInterrupt:
        print("Interrupted. If rollback was incomplete, rerun with --restore.", file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
