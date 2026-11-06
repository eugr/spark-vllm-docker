#!/usr/bin/env python3
"""
node-status.py - Cluster node status monitor

Shows HF cache, Docker images/containers, GPU and system resource usage
across all nodes defined in .env (or --config). Remote nodes are compared
against the local node so you can spot missing images/models at a glance.

Usage:
    ./node-status.py
    ./node-status.py --config /path/to/.env
    ./node-status.py --user nvidia
    ./node-status.py --json
    ./node-status.py --no-color
    ./node-status.py --perf-only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Remote probe (runs verbatim on each node via bash) ────────────────────────
# No set -e: individual failures must not abort the whole probe.
REMOTE_PROBE = r"""
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== SYSTEM ==="
echo "hostname=$(hostname)"

CPU=$(top -bn1 2>/dev/null | grep -E "^(%Cpu|Cpu)" | head -1 \
      | awk '{printf "%.1f", 100 - $8}' 2>/dev/null) || CPU="?"
echo "cpu_pct=$CPU"

MEM_LINE=$(free -m 2>/dev/null | awk 'NR==2{print $2,$3,$7}') || MEM_LINE=""
if [ -n "$MEM_LINE" ]; then
    read -r mem_total mem_used mem_avail <<< "$MEM_LINE"
else
    mem_total="?"; mem_used="?"; mem_avail="?"
fi
echo "mem_total_mb=$mem_total"
echo "mem_used_mb=$mem_used"
echo "mem_avail_mb=$mem_avail"

echo "hf_dir=$HF_DIR"
if [ -d "$HF_DIR" ]; then
    echo "hf_used=$(du -sh "$HF_DIR" 2>/dev/null | cut -f1)"
    echo "hf_disk=$(df -h "$HF_DIR" 2>/dev/null | awk 'NR==2{print $2,$3,$4,$5}')"
else
    echo "hf_used=(not found)"
    echo "hf_disk="
fi

echo "=== HF_MODELS ==="
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli scan-cache --quiet 2>/dev/null \
        | awk 'NR>2 && /^[a-zA-Z0-9_\-\/]+/ {printf "%s\t%s\n", $1, $2}' \
        || true
elif [ -d "$HF_DIR/hub" ]; then
    find "$HF_DIR/hub" -maxdepth 1 -mindepth 1 -type d 2>/dev/null \
        | while read -r d; do
            name="${d##*/}"
            name="${name#models--}"
            name="${name/--//}"
            echo "$name"
          done
fi

echo "=== DOCKER_IMAGES ==="
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker images --format "{{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}" 2>/dev/null || true
else
    echo "(docker unavailable)"
fi

echo "=== DOCKER_CONTAINERS ==="
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker ps -a --format "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true
else
    echo "(docker unavailable)"
fi

echo "=== GPU ==="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=index,name,utilization.gpu,clocks.gr,clocks.sm,power.draw,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null || echo "(nvidia-smi error)"
else
    echo "(no GPU)"
fi

echo "=== ACPI_TEMPS ==="
if command -v sensors >/dev/null 2>&1; then
    sensors acpitz-acpi-0 2>/dev/null | awk '/^temp[0-9]+:/{printf "%s\n", $2}' || true
else
    echo "(sensors unavailable)"
fi
"""


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class GpuInfo:
    index: str
    name: str
    util_pct: str
    clocks_gr_mhz: str
    clocks_sm_mhz: str
    power_draw_w: str
    temp_c: str


@dataclass
class DockerImage:
    repo: str
    tag: str
    image_id: str
    size: str
    created: str

    @property
    def full_name(self) -> str:
        return f"{self.repo}:{self.tag}"


@dataclass
class DockerContainer:
    name: str
    image: str
    status: str
    ports: str


@dataclass
class HfModel:
    name: str
    size: str = ""


@dataclass
class NodeResult:
    node: str
    error: Optional[str] = None
    hostname: str = ""
    cpu_pct: str = "?"
    mem_used_mb: str = "?"
    mem_total_mb: str = "?"
    mem_avail_mb: str = "?"
    hf_dir: str = ""
    hf_used: str = ""
    hf_disk: str = ""
    hf_models: list[HfModel] = field(default_factory=list)
    docker_images: list[DockerImage] = field(default_factory=list)
    docker_containers: list[DockerContainer] = field(default_factory=list)
    gpus: list[GpuInfo] = field(default_factory=list)
    acpi_temps: list[str] = field(default_factory=list)
    is_local: bool = False


# ── Probe parsing ─────────────────────────────────────────────────────────────
def parse_probe_output(node: str, raw: str, is_local: bool) -> NodeResult:
    result = NodeResult(node=node, is_local=is_local)
    section = None

    for line in raw.splitlines():
        line = line.rstrip()
        if line == "=== SYSTEM ===":
            section = "system"
        elif line == "=== HF_MODELS ===":
            section = "hf"
        elif line == "=== DOCKER_IMAGES ===":
            section = "dimages"
        elif line == "=== DOCKER_CONTAINERS ===":
            section = "dcontainers"
        elif line == "=== GPU ===":
            section = "gpu"
        elif line == "=== ACPI_TEMPS ===":
            section = "acpi_temps"
        elif not line:
            continue
        elif section == "system" and "=" in line:
            key, _, val = line.partition("=")
            setattr(result, key, val) if hasattr(result, key) else None
        elif section == "hf":
            parts = line.split("\t")
            result.hf_models.append(HfModel(name=parts[0], size=parts[1] if len(parts) > 1 else ""))
        elif section == "dimages":
            if line.startswith("("):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                result.docker_images.append(DockerImage(
                    repo=parts[0], tag=parts[1], image_id=parts[2],
                    size=parts[3], created=parts[4] if len(parts) > 4 else "",
                ))
        elif section == "dcontainers":
            if line.startswith("("):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                result.docker_containers.append(DockerContainer(
                    name=parts[0], image=parts[1], status=parts[2],
                    ports=parts[3] if len(parts) > 3 else "",
                ))
        elif section == "gpu":
            if line.startswith("("):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                result.gpus.append(GpuInfo(
                    index=parts[0], name=parts[1], util_pct=parts[2],
                    clocks_gr_mhz=parts[3], clocks_sm_mhz=parts[4],
                    power_draw_w=parts[5], temp_c=parts[6],
                ))
        elif section == "acpi_temps":
            if not line.startswith("("):
                result.acpi_temps.append(line.strip().lstrip("+"))

    return result


# ── SSH / local execution ─────────────────────────────────────────────────────
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=no"]


def run_probe(node: str, ssh_user: str, is_local: bool) -> NodeResult:
    try:
        if is_local:
            proc = subprocess.run(
                ["bash", "-s"],
                input=REMOTE_PROBE, capture_output=True, text=True, timeout=30,
            )
        else:
            proc = subprocess.run(
                ["ssh"] + SSH_OPTS + [f"{ssh_user}@{node}", "bash", "-s"],
                input=REMOTE_PROBE, capture_output=True, text=True, timeout=30,
            )
        raw = proc.stdout
        if not raw.strip():
            stderr = proc.stderr.strip()
            return NodeResult(node=node, is_local=is_local,
                              error=f"empty output (stderr: {stderr})")
        return parse_probe_output(node, raw, is_local)
    except subprocess.TimeoutExpired:
        return NodeResult(node=node, is_local=is_local, error="timeout (30s)")
    except Exception as exc:
        return NodeResult(node=node, is_local=is_local, error=str(exc))


# ── .env loader ───────────────────────────────────────────────────────────────
def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            env[key.strip()] = val
    return env


# ── Terminal colors ───────────────────────────────────────────────────────────
class C:
    BOLD = RESET = RED = GREEN = YELLOW = CYAN = DIM = ""

    @classmethod
    def enable(cls):
        cls.BOLD   = "\033[1m"
        cls.RESET  = "\033[0m"
        cls.RED    = "\033[31m"
        cls.GREEN  = "\033[32m"
        cls.YELLOW = "\033[33m"
        cls.CYAN   = "\033[36m"
        cls.DIM    = "\033[2m"


# ── Text display ──────────────────────────────────────────────────────────────
def mem_bar(used: str, total: str, width: int = 20) -> str:
    try:
        pct = int(used) / int(total)
        filled = round(pct * width)
        bar = "█" * filled + "░" * (width - filled)
        color = C.GREEN if pct < 0.7 else C.YELLOW if pct < 0.9 else C.RED
        return f"{color}[{bar}]{C.RESET} {pct*100:.0f}%"
    except (ValueError, ZeroDivisionError):
        return ""


def print_node(result: NodeResult, local_hf: set[str], local_imgs: dict[str, str], perf_only: bool = False):
    label = result.node
    if result.hostname and result.hostname != result.node:
        label += f" ({result.hostname})"
    if result.is_local:
        label += "  [LOCAL]"

    print()
    print(f"{C.CYAN}{C.BOLD}{'━'*66}{C.RESET}")
    print(f"{C.BOLD}  Node: {label}{C.RESET}")
    print(f"{C.CYAN}{'━'*66}{C.RESET}")

    if result.error:
        print(f"  {C.RED}ERROR: {result.error}{C.RESET}")
        return

    # System
    print(f"{C.BOLD}  System{C.RESET}")
    bar = mem_bar(result.mem_used_mb, result.mem_total_mb)
    try:
        cpu_f = float(result.cpu_pct)
        cpu_color = C.GREEN if cpu_f < 70 else C.YELLOW if cpu_f < 90 else C.RED
    except ValueError:
        cpu_color = C.RESET
    print(f"    CPU:    {cpu_color}{result.cpu_pct}%{C.RESET}")
    print(f"    Memory: {result.mem_used_mb} / {result.mem_total_mb} MB  {bar}")
    if result.hf_used:
        print(f"    HF Cache: {result.hf_used}  ({result.hf_dir})")
    if result.hf_disk:
        print(f"    Disk:     {result.hf_disk}  (total used avail use%)")

    # GPU
    print(f"\n{C.BOLD}  GPU{C.RESET}")
    if result.gpus:
        for g in result.gpus:
            try:
                util_f = float(g.util_pct)
                uc = C.GREEN if util_f < 70 else C.YELLOW if util_f < 90 else C.RED
            except ValueError:
                uc = C.RESET
            try:
                pw = float(g.power_draw_w)
                pc = C.GREEN if pw < 200 else C.YELLOW if pw < 350 else C.RED
                power_str = f"{pc}{pw:.0f}W{C.RESET}"
            except ValueError:
                power_str = f"{g.power_draw_w}W"
            print(f"    GPU{g.index:<2}  {g.name:<32}  "
                  f"util:{uc}{g.util_pct:>3}%{C.RESET}  "
                  f"gr:{g.clocks_gr_mhz}MHz  sm:{g.clocks_sm_mhz}MHz  "
                  f"pwr:{power_str}  temp:{g.temp_c}°C")
    else:
        print("    (none)")

    # ACPI temperatures
    if result.acpi_temps:
        temps = "  ".join(result.acpi_temps)
        max_temp = max((float(t.rstrip("°C")) for t in result.acpi_temps
                        if t.rstrip("°C").replace(".", "").isdigit()), default=0)
        tc = C.GREEN if max_temp < 70 else C.YELLOW if max_temp < 85 else C.RED
        print(f"    ACPI temps: {tc}{temps}{C.RESET}")

    if not perf_only:
        # HF models
        print(f"\n{C.BOLD}  HuggingFace Cache{C.RESET}")
        if result.is_local:
            for m in result.hf_models:
                size_str = f"  {C.DIM}{m.size}{C.RESET}" if m.size else ""
                print(f"    {m.name:<55}{size_str}")
            if not result.hf_models:
                print("    (none)")
        else:
            remote_hf = {m.name: m for m in result.hf_models}
            printed = False
            for name in sorted(local_hf):
                if name.startswith("."):
                    continue
                if name in remote_hf:
                    m = remote_hf[name]
                    size_str = f"  {C.DIM}{m.size}{C.RESET}" if m.size else ""
                    print(f"    {name:<55}{size_str}  {C.GREEN}✓{C.RESET}")
                else:
                    print(f"    {C.RED}{name:<55}  ✗ missing{C.RESET}")
                printed = True
            for name, m in sorted(remote_hf.items()):
                if name.startswith("."):
                    continue
                if name not in local_hf:
                    size_str = f"  {C.DIM}{m.size}{C.RESET}" if m.size else ""
                    print(f"    {name:<55}{size_str}  {C.YELLOW}? remote only{C.RESET}")
                    printed = True
            if not printed:
                print("    (none)")

        # Docker images
        print(f"\n{C.BOLD}  Docker Images{C.RESET}")
        if result.is_local:
            for img in result.docker_images:
                print(f"    {img.full_name:<42}  {img.image_id:<14}  "
                      f"{img.size:<10}  {C.DIM}{img.created}{C.RESET}")
            if not result.docker_images:
                print("    (none)")
        else:
            remote_imgs = {img.full_name: img for img in result.docker_images}
            printed = False
            for full_name in sorted(local_imgs):
                if full_name in remote_imgs:
                    img = remote_imgs[full_name]
                    local_id = local_imgs[full_name]
                    if local_id and img.image_id and local_id != img.image_id:
                        status = f"{C.YELLOW}↑ sync required{C.RESET}"
                    else:
                        status = f"{C.GREEN}✓{C.RESET}"
                    print(f"    {full_name:<42}  {img.image_id:<14}  "
                          f"{img.size:<10}  {C.DIM}{img.created}{C.RESET}  {status}")
                else:
                    print(f"    {C.RED}{full_name:<42}  {'':14}  {'':10}  ✗ missing{C.RESET}")
                printed = True
            for full_name, img in sorted(remote_imgs.items()):
                if full_name not in local_imgs:
                    print(f"    {full_name:<42}  {img.image_id:<14}  "
                          f"{img.size:<10}  {C.DIM}{img.created}{C.RESET}  {C.YELLOW}? remote only{C.RESET}")
                    printed = True
            if not printed:
                print("    (none)")

        # Docker containers
        print(f"\n{C.BOLD}  Docker Containers{C.RESET}")
        if result.docker_containers:
            for c in result.docker_containers:
                if c.status.startswith("Up"):
                    sc = C.GREEN
                elif c.status.startswith(("Exited", "Dead")):
                    sc = C.RED
                else:
                    sc = C.YELLOW
                ports = f"  {C.DIM}{c.ports}{C.RESET}" if c.ports else ""
                print(f"    {c.name:<26}  {c.image:<36}  {sc}{c.status}{C.RESET}{ports}")
        else:
            print("    (none)")


# ── JSON output ───────────────────────────────────────────────────────────────
def result_to_dict(result: NodeResult, local_hf: set[str], local_imgs: dict[str, str]) -> dict:
    return {
        "node": result.node,
        "is_local": result.is_local,
        "error": result.error,
        "hostname": result.hostname,
        "cpu_pct": result.cpu_pct,
        "mem_used_mb": result.mem_used_mb,
        "mem_total_mb": result.mem_total_mb,
        "hf_dir": result.hf_dir,
        "hf_used": result.hf_used,
        "hf_disk": result.hf_disk,
        "hf_models": [
            {"name": m.name, "size": m.size,
             "on_local": m.name in local_hf}
            for m in result.hf_models
        ],
        "docker_images": [
            {"repo": i.repo, "tag": i.tag, "id": i.image_id,
             "size": i.size, "created": i.created,
             "on_local": i.full_name in local_imgs}
            for i in result.docker_images
        ],
        "docker_containers": [
            {"name": c.name, "image": c.image,
             "status": c.status, "ports": c.ports}
            for c in result.docker_containers
        ],
        "gpus": [
            {"index": g.index, "name": g.name,
             "util_pct": g.util_pct, "clocks_gr_mhz": g.clocks_gr_mhz,
             "clocks_sm_mhz": g.clocks_sm_mhz, "power_draw_w": g.power_draw_w,
             "temp_c": g.temp_c}
            for g in result.gpus
        ],
        "acpi_temps": result.acpi_temps,
    }


# ── Local reference sets ──────────────────────────────────────────────────────
def get_local_hf_models(hf_cache_dir: str) -> set[str]:
    models: set[str] = set()
    hub = Path(hf_cache_dir) / "hub"
    if hub.is_dir():
        for d in hub.iterdir():
            if d.is_dir():
                name = d.name
                name = re.sub(r"^models--", "", name)
                name = name.replace("--", "/")
                models.add(name)
    return models


def get_local_docker_images() -> dict[str, str]:
    """Returns {full_name: image_id} for all local docker images."""
    try:
        proc = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}"],
            capture_output=True, text=True, timeout=10,
        )
        result = {}
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            result[parts[0]] = parts[1] if len(parts) == 2 else ""
        return result
    except Exception:
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Cluster node status monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=str(script_dir / ".env"),
                        help="Path to .env config file (default: .env next to this script)")
    parser.add_argument("--user", default=None,
                        help="SSH username (default: $USER or SSH_USER in .env)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")
    parser.add_argument("--perf-only", action="store_true",
                        help="Show only System and GPU sections (skip HF models and Docker)")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON (implies --no-color)")
    args = parser.parse_args()

    if args.json:
        args.no_color = True
    if not args.no_color and sys.stdout.isatty():
        C.enable()

    # Load config
    env = load_env(Path(args.config))

    cluster_nodes_str = env.get("CLUSTER_NODES", os.environ.get("CLUSTER_NODES", ""))
    if not cluster_nodes_str:
        print("Error: CLUSTER_NODES not set. Use --config or set it in .env.", file=sys.stderr)
        sys.exit(1)

    nodes = [n.strip() for n in cluster_nodes_str.split(",") if n.strip()]
    local_ip = env.get("LOCAL_IP", os.environ.get("LOCAL_IP", ""))
    ssh_user = args.user or env.get("SSH_USER", os.environ.get("USER", ""))
    hf_cache_dir = env.get("HF_CACHE_DIR",
                            os.environ.get("HF_HOME",
                            str(Path.home() / ".cache" / "huggingface")))

    # Build local reference sets (from this machine)
    local_hf = get_local_hf_models(hf_cache_dir)
    local_imgs = get_local_docker_images()

    if not args.json:
        print(f"{C.BOLD}Cluster Node Status Monitor{C.RESET}  "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{C.DIM}Config: {args.config}  |  Nodes: {cluster_nodes_str}{C.RESET}")
        if not args.no_color:
            print(f"{C.DIM}Legend: {C.GREEN}✓ local{C.RESET}{C.DIM} = also on local node   "
                  f"{C.YELLOW}+ remote only{C.RESET}{C.DIM} = missing locally{C.RESET}")

    # Probe all nodes in parallel
    results: dict[str, NodeResult] = {}
    lock = threading.Lock()

    def probe(node: str):
        is_local = bool(local_ip and node == local_ip)
        r = run_probe(node, ssh_user, is_local)
        with lock:
            results[node] = r

    with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        futures = {pool.submit(probe, n): n for n in nodes}
        for f in as_completed(futures):
            f.result()  # surface exceptions

    # Output in original node order
    if args.json:
        output = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "nodes": [result_to_dict(results[n], local_hf, local_imgs) for n in nodes],
        }
        print(json.dumps(output, indent=2))
    else:
        for node in nodes:
            print_node(results[node], local_hf, local_imgs, perf_only=args.perf_only)
        print()


if __name__ == "__main__":
    main()
