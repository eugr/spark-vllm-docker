"""Saved RoCE links, rank ordering, and SSH copy paths (Python 3.10+).

CLUSTER_LINKS is a JSON list of two-endpoint objects. Each endpoint maps a
management IP to its RoCE IPv4 address. Parallel rails remain separate links,
but count as one neighbor when classifying the graph. No shell code is loaded.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


RING_ENV = {
    "NCCL_ALGO": "Ring",
    "NCCL_NET_PLUGIN": "none",
    "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
    "NCCL_IB_MERGE_NICS": "0",
}


def split_nodes(value):
    return [node.strip() for node in value.split(",") if node.strip()]


def ip_key(value):
    return int(ipaddress.IPv4Address(value))


class Topology:
    def __init__(self, nodes, links):
        self.nodes = list(nodes)
        if not self.nodes or len(set(self.nodes)) != len(self.nodes):
            raise ValueError("Topology requires a nonempty, unique CLUSTER_NODES list")
        for node in self.nodes:
            ip_key(node)
        if not isinstance(links, list):
            raise ValueError("CLUSTER_LINKS must be a JSON list of two-endpoint objects")
        self.links = links
        self.adj = {node: {} for node in self.nodes}
        self.owners = {node: node for node in self.nodes}
        seen = set()
        for link in links:
            if not isinstance(link, dict) or len(link) != 2:
                raise ValueError("Each CLUSTER_LINKS entry must map two nodes to their link IPs")
            a, b = link
            if a not in self.adj or b not in self.adj:
                raise ValueError("CLUSTER_LINKS contains a node absent from CLUSTER_NODES")
            for node, address in link.items():
                ip_key(address)
                if address in self.owners and self.owners[address] != node:
                    raise ValueError("A link IP belongs to more than one node")
                self.owners[address] = node
            identity = tuple(sorted(link.items()))
            if identity in seen:
                raise ValueError("CLUSTER_LINKS contains a duplicate link")
            seen.add(identity)
            self.adj[a].setdefault(b, []).append((link[a], link[b]))
            self.adj[b].setdefault(a, []).append((link[b], link[a]))
        self.paths(self.nodes[0])  # Reject disconnected graphs, including isolated nodes.

    @classmethod
    def from_env(cls, env):
        if not env.get("CLUSTER_LINKS"):
            return None
        try:
            links = json.loads(env["CLUSTER_LINKS"])
        except json.JSONDecodeError as exc:
            raise ValueError("CLUSTER_LINKS is not valid JSON") from exc
        topology = cls(split_nodes(env.get("CLUSTER_NODES", "")), links)
        if env.get("LOCAL_IP") and env["LOCAL_IP"] != topology.nodes[0]:
            raise ValueError("With CLUSTER_LINKS, LOCAL_IP must be first in CLUSTER_NODES")
        return topology

    @property
    def ring(self):
        return len(self.nodes) >= 4 and all(len(peers) == 2 for peers in self.adj.values())

    @property
    def complete(self):
        return all(len(peers) == len(self.nodes) - 1 for peers in self.adj.values())

    def paths(self, head):
        if head not in self.adj:
            raise ValueError("The local/head node is absent from CLUSTER_NODES")
        paths = {head: []}
        queue = deque([head])
        while queue:
            node = queue.popleft()
            for peer in sorted(self.adj[node], key=ip_key):
                if peer not in paths:
                    # Prefer one deterministic rail; do not treat twin rails as neighbors.
                    _, address = min(self.adj[node][peer], key=lambda pair: ip_key(pair[1]))
                    paths[peer] = paths[node] + [(peer, address)]
                    queue.append(peer)
        if len(paths) != len(self.nodes):
            raise ValueError("CLUSTER_LINKS is disconnected; not every node has a copy path")
        return paths

    def rank_order(self, head):
        self.paths(head)
        if self.complete:
            return [head] + sorted(set(self.nodes) - {head}, key=ip_key)
        if not self.ring:
            raise ValueError("Discovered links form neither a complete network nor a closed ring")
        order = [head]
        while len(order) < len(self.nodes):
            peers = set(self.adj[order[-1]]) - set(order)
            order.append(min(peers, key=ip_key))
        return order

    def subset(self, nodes):
        if not set(nodes) <= set(self.nodes):
            raise ValueError("Selected nodes are absent from the saved topology")
        return Topology(nodes, [link for link in self.links if set(link) <= set(nodes)])

    def validate_order(self, nodes):
        selected = self.subset(nodes)
        if selected.complete:
            return
        if not selected.ring:
            raise ValueError("Selected nodes break the ring; use the full ring or a directly connected pair")
        if any(b not in selected.adj[a] for a, b in zip(nodes, nodes[1:] + nodes[:1])):
            raise ValueError("CLUSTER_NODES must follow physical ring order; run --discover again")

    @contextmanager
    def ssh_config(self, head, user=None):
        """Override only routing; retain existing SSH identities and authentication.

        Management IPs are also the SSH aliases and host-key identities. This
        preserves per-host settings from the configurator at every jump.
        """
        with tempfile.TemporaryDirectory(prefix="spark-copy-ssh-") as directory:
            config = Path(directory) / "config"
            lines = []
            for node, path in self.paths(head).items():
                if not path:
                    continue
                lines += [f"Host {node}", f"    HostName {path[-1][1]}",
                          f"    HostKeyAlias {node}"]
                if user:
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
                        raise ValueError("Invalid SSH user")
                    lines.append(f"    User {user}")
                if len(path) > 1:
                    lines.append("    ProxyJump " + ",".join(hop[0] for hop in path[:-1]))
            # -F replaces both default config files; include them explicitly.
            lines += ["Host *", "    BatchMode yes", "    ConnectTimeout 10",
                      'Include "~/.ssh/config"', 'Include "/etc/ssh/ssh_config"']
            config.write_text("\n".join(lines) + "\n")
            config.chmod(0o600)
            yield config


def validate_launch(topology, nodes, command="", ray=False, env=None):
    """Check the actual selected ranks before any deployment or container cleanup."""
    if not nodes or nodes[0] != topology.nodes[0]:
        raise ValueError("The saved local/head node must remain first in the selected ranks")
    effective_env = dict(env or {})
    for line in command.splitlines():
        if line.strip().startswith("export "):
            for assignment in shlex.split(line.strip()[7:], comments=True):
                if "=" in assignment:
                    key, value = assignment.split("=", 1)
                    effective_env[key] = value
    sizes = []
    for name, short in (("tensor", "tp"), ("pipeline", "pp"), ("data", "dp")):
        matches = re.findall(r"(?:^|\s)--?" + rf"(?:{name}-parallel-size|{short})(?:=|\s+)(\d+)", command)
        sizes.append(int(matches[-1]) if matches else 1)
    tp, pp, dp = sizes
    if any(size < 1 for size in sizes):
        raise ValueError("Parallelism sizes must be positive")
    has_sizes = bool(re.search(r"--?(?:tensor-parallel-size|pipeline-parallel-size|data-parallel-size|tp|pp|dp)(?:=|\s)", command))
    if has_sizes:
        required = tp * pp * dp
        if required > len(nodes):
            raise ValueError("Requested parallelism exceeds the number of configured nodes")
        nodes = nodes[:required]
    topology.validate_order(nodes)
    if topology.ring and len(nodes) > 2:
        if ray:
            raise ValueError("Ring topology requires the native backend with fixed rank placement; omit --ray")
        if has_sizes and (pp != 1 or dp != 1):
            raise ValueError("Ring topology currently supports tensor parallelism only (PP=1, DP=1)")
        for key, value in RING_ENV.items():
            if key in effective_env and str(effective_env[key]) != value:
                raise ValueError(f"Ring topology requires {key}={value}")
        return RING_ENV
    return {}


# Sent through management SSH, never requires a checkout on worker nodes.
PROBE = r'''
import json, re, subprocess
devices = {}
for line in subprocess.check_output(["ibdev2netdev"], text=True).splitlines():
    match = re.match(r"(\S+) port \d+ ==> (\S+) \(Up\)", line)
    if match:
        devices[match[2]] = match[1]
addresses = json.loads(subprocess.check_output(["ip", "-j", "-4", "addr", "show"], text=True))
print(json.dumps([{"interface": item["ifname"], "address": addr["local"], "prefix": addr["prefixlen"]}
    for item in addresses if item["ifname"] in devices
    for addr in item.get("addr_info", []) if addr["family"] == "inet"]))
'''


def on_node(node, head, command, **kwargs):
    argv = command if node == head else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node, shlex.join(command)]
    return subprocess.run(argv, capture_output=True, text=True, timeout=30, **kwargs)


def discover(nodes, head):
    records = {}
    for node in nodes:
        result = on_node(node, head, ["python3", "-"], input=PROBE)
        if result.returncode:
            raise ValueError(f"Could not inspect RoCE interfaces on {node}")
        records[node] = json.loads(result.stdout)
    links = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            for left in records[a]:
                for right in records[b]:
                    net_a = ipaddress.ip_interface(f'{left["address"]}/{left["prefix"]}').network
                    net_b = ipaddress.ip_interface(f'{right["address"]}/{right["prefix"]}').network
                    if net_a != net_b:
                        continue
                    # Verify direct reachability in both directions, bound to the
                    # source interface so management routing cannot fake a link.
                    for src, dst, address, peer in ((a, b, left, right), (b, a, right, left)):
                        result = on_node(src, head, ["ping", "-n", "-c", "1", "-W", "2",
                                                   "-I", address["interface"], peer["address"]])
                        if result.returncode:
                            raise ValueError(f"RoCE link check failed between {src} and {dst}")
                    links.append({a: left["address"], b: right["address"]})
    topology = Topology(nodes, links)
    return Topology(topology.rank_order(head), links)


def dotenv_environment():
    return {key.removeprefix("DOTENV_"): value for key, value in os.environ.items() if key.startswith("DOTENV_")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["discover", "select", "check", "validate", "ssh", "rsync"])
    parser.add_argument("--nodes")
    parser.add_argument("--head")
    parser.add_argument("--user")
    parser.add_argument("--target")
    parser.add_argument("--command", default="")
    parser.add_argument("--ray", action="store_true")
    args, rest = parser.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    env = dotenv_environment()
    topology = Topology.from_env(env) if args.action != "discover" else None
    nodes = split_nodes(args.nodes or env.get("CLUSTER_NODES", ""))
    head = args.head or env.get("LOCAL_IP") or (nodes[0] if nodes else None)
    if args.action in ("discover", "select"):
        topology = discover(nodes, head) if args.action == "discover" else topology.subset(nodes)
        order = topology.rank_order(head)
        print(",".join(order))
        print(json.dumps(topology.links, separators=(",", ":")))
        print("ring" if topology.ring else "complete")
    elif args.action == "check":
        if topology:
            topology.paths(head)
    elif args.action == "validate":
        if topology:
            # Docker -e options are ordered; the last value is effective.
            nccl_env = {key.removeprefix("CONTAINER_"): value for key, value in env.items() if key.startswith("CONTAINER_")}
            docker_args = shlex.split(os.environ.get("TOPOLOGY_DOCKER_ARGS", ""))
            for i, token in enumerate(docker_args[:-1]):
                if token == "-e" and "=" in docker_args[i + 1]:
                    key, value = docker_args[i + 1].split("=", 1)
                    nccl_env[key] = value
            settings = validate_launch(topology, nodes, args.command, args.ray, nccl_env)
            print(" ".join(f"-e {key}={value}" for key, value in settings.items()))
    else:
        if not topology:
            raise ValueError("Copy routing requires CLUSTER_LINKS")
        target = args.target.strip()
        target = topology.owners.get(target, target)
        with topology.ssh_config(head, args.user) as config:
            user_args = ["-l", args.user] if args.user else []
            if args.action == "ssh":
                # Options precede the destination; the command is the final argument.
                command = ["ssh", "-F", str(config), *user_args, *rest[:-1], target, rest[-1]]
            else:
                source, destination = rest
                command = ["rsync", "-av", "-s", "--mkpath", "--progress", "--copy-unsafe-links",
                           "-e", shlex.join(["ssh", "-F", str(config), *user_args]), source, f"{target}:{destination}"]
            return subprocess.run(command).returncode
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
