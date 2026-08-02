#!/usr/bin/env python3
"""
spark - fleet manager for multi-model vLLM serving.

Subcommands (growing):
  scan    List models in the HuggingFace cache with size, quant, capabilities.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# HF cache discovery
# ---------------------------------------------------------------------------

def hub_dir(override: str | None = None) -> Path:
    """Resolve the HuggingFace hub cache directory."""
    if override:
        return Path(override).expanduser()
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def model_dirs(hub: Path) -> list[Path]:
    return sorted(p for p in hub.glob("models--*") if p.is_dir())


def repo_id(model_dir: Path) -> str:
    return model_dir.name.removeprefix("models--").replace("--", "/")


def latest_snapshot(model_dir: Path) -> Path | None:
    """Snapshot referenced by refs/main if present, else newest by mtime."""
    ref = model_dir / "refs" / "main"
    snaps = model_dir / "snapshots"
    if ref.is_file():
        commit = ref.read_text().strip()
        cand = snaps / commit
        if cand.is_dir():
            return cand
    dirs = [d for d in snaps.glob("*") if d.is_dir()] if snaps.is_dir() else []
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def weights_bytes(snapshot: Path) -> int:
    """Sum of *.safetensors (symlinks resolve to blobs; stat follows)."""
    return sum(f.stat().st_size for f in snapshot.glob("*.safetensors"))


def load_config(snapshot: Path) -> dict:
    cfg = snapshot / "config.json"
    if not cfg.is_file():
        return {}
    try:
        return json.loads(cfg.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Detection (pure)
# ---------------------------------------------------------------------------

def _text_cfg(cfg: dict) -> dict:
    """Capability fields may sit at top level or under text_config (VL models)."""
    return {**(cfg.get("text_config") or {}), **cfg}


def detect_pooling(snapshot: Path, name: str) -> str | None:
    """Detect a pooling model (embedder/reranker) — no KV cache, not generative.

    Signal is the sentence-transformers artifacts in the snapshot, since the HF arch
    is the same `Qwen3ForCausalLM` as a chat model. Embed vs rerank from the name.
    """
    markers = ("modules.json", "config_sentence_transformers.json", "sentence_bert_config.json")
    has_st = any((snapshot / m).is_file() for m in markers) or (snapshot / "1_Pooling").is_dir()
    if not has_st:
        return None
    return "rerank" if "rerank" in name.lower() else "embed"


def detect_quant(cfg: dict, name: str) -> str:
    n = name.lower()
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config")
    fmt = (qc or {}).get("format") or ""
    blob = f"{n} {fmt}".lower()
    for tag in ("nvfp4", "fp8", "awq", "gptq"):
        if tag in blob:
            return tag
    if "int4" in blob or "autoround" in blob:
        return "int4"
    if qc:
        return str(qc.get("quant_method", "quantized"))
    return "bf16"  # unquantized


def full_attention_layers(cfg: dict) -> int | None:
    """Layers that actually hold per-token KV.

    Hybrid models (linear/SSM + full attention) only cache on full-attention layers.
    Prefer the explicit `layer_types` list; else derive from `full_attention_interval`;
    else assume every layer is full attention.
    """
    t = _text_cfg(cfg)
    layers = t.get("num_hidden_layers")
    types = t.get("layer_types")
    if isinstance(types, list) and types:
        return sum(1 for x in types if x == "full_attention")
    interval = t.get("full_attention_interval")
    if isinstance(layers, int) and isinstance(interval, int) and interval > 0:
        return layers // interval
    return layers if isinstance(layers, int) else None


def kv_bytes_per_token(cfg: dict, dtype_bytes: int = 1) -> int | None:
    t = _text_cfg(cfg)
    layers = full_attention_layers(cfg)
    kv_heads, head_dim = t.get("num_key_value_heads"), t.get("head_dim")
    if not all(isinstance(x, int) for x in (layers, kv_heads, head_dim)):
        return None
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def capabilities(cfg: dict) -> list[str]:
    t = _text_cfg(cfg)
    caps: list[str] = []
    experts = t.get("num_experts") or t.get("n_routed_experts")
    if experts:
        caps.append(f"moe({experts}e)")
    else:
        caps.append("dense")
    if cfg.get("vision_config") or (cfg.get("model_type") or "").endswith(("vl", "ocr")):
        caps.append("vision")
    if t.get("linear_key_head_dim"):
        caps.append("hybrid-attn")
    if t.get("mtp_num_hidden_layers") or t.get("num_nextn_predict_layers"):
        caps.append("mtp")
    return caps


@dataclass
class ModelInfo:
    repo_id: str
    size_gb: float
    quant: str
    arch: str
    ctx: int | None
    caps: list[str]
    kv_mb_per_1k_fp8: float | None
    pooling: str | None   # "embed" / "rerank" => no KV cache

    @staticmethod
    def of(model_dir: Path) -> "ModelInfo | None":
        snap = latest_snapshot(model_dir)
        if snap is None:
            return None
        cfg = load_config(snap)
        t = _text_cfg(cfg)
        pooling = detect_pooling(snap, model_dir.name)
        kvpt = None if pooling else kv_bytes_per_token(cfg, dtype_bytes=1)
        archs = cfg.get("architectures") or []
        return ModelInfo(
            repo_id=repo_id(model_dir),
            size_gb=round(weights_bytes(snap) / 1024**3, 1),
            quant=detect_quant(cfg, model_dir.name),
            arch=cfg.get("model_type") or (archs[0] if archs else "?"),
            ctx=t.get("max_position_embeddings"),
            caps=[pooling] if pooling else capabilities(cfg),
            kv_mb_per_1k_fp8=round(kvpt * 1000 / 1024**2, 1) if kvpt else None,
            pooling=pooling,
        )


def scan(hub: Path) -> list[ModelInfo]:
    out = [ModelInfo.of(d) for d in model_dirs(hub)]
    return sorted((m for m in out if m), key=lambda m: m.size_gb)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _ctx_str(ctx: int | None) -> str:
    if not ctx:
        return "?"
    if ctx >= 1024**2:
        return f"{ctx / 1024**2:.0f}M".replace(".0M", "M") if ctx % 1024**2 == 0 else f"{ctx / 1024**2:.1f}M"
    return f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)


def parse_tokens(s: str) -> int:
    """Parse a token count with optional k/m suffix: '262k', '1M', '1048576'."""
    s = s.strip().lower()
    mult = {"k": 1024, "m": 1024**2}.get(s[-1:], 1)
    num = s[:-1] if mult > 1 else s
    return int(float(num) * mult)


def kv_gb_for_tokens(m: ModelInfo, tokens: int) -> float | None:
    """KV cache GB (fp8) to hold `tokens` total cached tokens across all sequences."""
    if m.kv_mb_per_1k_fp8 is None:
        return None
    return round(m.kv_mb_per_1k_fp8 * (tokens / 1000) / 1024, 1)


def render_table(models: list[ModelInfo], tokens: int | None = None) -> str:
    header = ["MODEL", "SIZE", "QUANT", "ARCH", "CTX", "KV/1K"]
    if tokens:
        header.append(f"KV@{_ctx_str(tokens)}")
    header.append("CAPS")
    rows = [tuple(header)]
    for m in models:
        row = [
            m.repo_id,
            f"{m.size_gb}G",
            m.quant,
            m.arch,
            _ctx_str(m.ctx),
            "—" if m.pooling else (f"{m.kv_mb_per_1k_fp8}M" if m.kv_mb_per_1k_fp8 is not None else "?"),
        ]
        if tokens:
            gb = kv_gb_for_tokens(m, tokens)
            row.append("—" if m.pooling else (f"{gb}G" if gb is not None else "?"))
        row.append(",".join(m.caps))
        rows.append(tuple(row))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    return "\n".join(lines)


def cmd_scan(args: argparse.Namespace) -> int:
    hub = hub_dir(args.hub)
    if not hub.is_dir():
        print(f"HF cache not found: {hub}", file=sys.stderr)
        return 1
    models = scan(hub)
    if args.json:
        out = [asdict(m) for m in models]
        if args.tokens:
            for m, d in zip(models, out):
                d["kv_gb_at_tokens"] = {str(args.tokens): kv_gb_for_tokens(m, args.tokens)}
        print(json.dumps(out, indent=2))
    else:
        print(render_table(models, args.tokens))
        print(f"\n{len(models)} model(s) · {round(sum(m.size_gb for m in models), 1)}G total · {hub}")
    return 0


# ---------------------------------------------------------------------------
# Fleet apply (Phase 1) — reconcile spark.yaml -> running containers
# ---------------------------------------------------------------------------

DEFAULT_REMOTE_DIR = "~/projects/spark-vllm-docker"


def _sanitize(name: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9._-]+", "-", str(name).lower()).strip("-._")
    return s or "x"


def container_of(dep: dict) -> str:
    return "spark-" + _sanitize(dep["name"])


def nodes_of(dep: dict) -> list[str]:
    if dep.get("nodes"):
        return [str(n) for n in dep["nodes"]]
    return [str(dep["node"])] if dep.get("node") else []


def assign_ports(deps: list[dict], base: int = 8001) -> dict:
    """Explicit `port:` wins; otherwise allocate sequential global-unique ports."""
    used = {d["port"] for d in deps if d.get("port")}
    out, nxt = {}, base
    for d in deps:
        if d.get("port"):
            out[d["name"]] = d["port"]
        else:
            while nxt in used:
                nxt += 1
            out[d["name"]] = nxt
            used.add(nxt)
    return out


def build_launch(dep: dict, port: int) -> list[str]:
    """run-recipe.sh argv for one deployment (solo or cluster).

    Memory caps live in the RECIPE (--gpu-memory-utilization-gb), not here — vLLM forbids
    combining that with --kv-cache-memory-bytes, so the manager does not inject mem flags.
    """
    nodes = nodes_of(dep)
    argv = ["./run-recipe.sh", dep["recipe"]]
    # Single-node = --node <ip> (master-orchestrated placement, local or remote).
    # Multi-node = -n a,b (Ray/cluster sharding).
    argv += ["--node", nodes[0]] if len(nodes) <= 1 else ["-n", ",".join(nodes)]
    argv += ["--name", container_of(dep), "--port", str(port), "-d"]
    # Optional per-deployment overrides (passed through to run-recipe.py).
    if dep.get("tp") is not None:
        argv += ["--tp", str(dep["tp"])]
    if dep.get("max_model_len") is not None:
        argv += ["--max-model-len", str(dep["max_model_len"])]
    if dep.get("gpu_mem") is not None:
        argv += ["--gpu-mem", str(dep["gpu_mem"])]
    extra = [str(a) for a in dep.get("extra_args", [])]
    if extra:
        argv += ["--"] + extra
    return argv


def is_serving(node: str, port: int) -> bool:
    """A deployment counts as 'up' only if it answers /v1/models. Retry with a generous
    timeout — a *busy* model can miss a single short probe, and we must NOT kill a working
    model just because it was slow to answer one curl."""
    import time
    for attempt in range(3):
        if _ssh(node, f"curl -sf -m10 http://localhost:{port}/v1/models >/dev/null 2>&1").returncode == 0:
            return True
        if attempt < 2:
            time.sleep(3)
    return False


def wait_ready(node: str, port: int, timeout: int = 600) -> bool:
    """Block (on the node) until host:port serves /v1/models, or timeout. Sequential start
    depends on this — never launch the next model until this one finishes profiling."""
    loop = (f'for i in $(seq 1 {max(1, timeout // 5)}); do '
            f'curl -sf -m 4 http://localhost:{port}/v1/models >/dev/null 2>&1 && exit 0; '
            f'sleep 5; done; exit 1')
    return _ssh(node, loop).returncode == 0


_LOCAL_HOSTS = None


def _is_local(host: str) -> bool:
    """Is `host` this machine? (master often == a fleet node; self-ssh isn't set up)."""
    global _LOCAL_HOSTS
    if _LOCAL_HOSTS is None:
        import subprocess as _sp, socket
        ips = {"localhost", "127.0.0.1", ""}
        try:
            ips.update(_sp.run(["hostname", "-I"], capture_output=True, text=True).stdout.split())
        except Exception:
            pass
        try:
            ips.add(socket.gethostname())
        except Exception:
            pass
        _LOCAL_HOSTS = ips
    return host in _LOCAL_HOSTS


def _ssh(node: str, remote_cmd: str, capture: bool = True):
    """Run a command on `node` — locally if it's this host, else over SSH."""
    import subprocess
    if _is_local(node):
        return subprocess.run(["bash", "-c", remote_cmd], capture_output=capture, text=True)
    return subprocess.run(["ssh", "-o", "BatchMode=yes", node, remote_cmd],
                          capture_output=capture, text=True)


def running_containers(node: str) -> set:
    r = _ssh(node, "docker ps -a --filter name=spark- --format '{{.Names}}'")
    return {ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()} if r.returncode == 0 else set()


def load_fleet(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        raise SystemExit(2)
    with open(path) as f:
        return yaml.safe_load(f)


def write_state(fleet: dict, deps: list[dict], ports: dict) -> None:
    state = {
        "fleet": fleet.get("fleet", []),
        "deployments": [
            {"name": d["name"], "recipe": d["recipe"], "nodes": nodes_of(d),
             "port": ports[d["name"]], "container": container_of(d),
             "kv_cache_gb": d.get("kv_cache_gb"), "status": "started"}
            for d in deps
        ],
    }
    Path(".state").mkdir(exist_ok=True)
    (Path(".state") / "cluster.json").write_text(json.dumps(state, indent=2))


def cmd_apply(args: argparse.Namespace) -> int:
    fleet = load_fleet(args.file)
    deps = fleet.get("deployments") or []
    if not deps:
        print(f"No deployments in {args.file}", file=sys.stderr)
        return 1
    import threading
    ports = assign_ports(deps)
    execute = args.execute
    # The master runs run-recipe.sh; --node places each container on its target node.
    master = args.master or (fleet.get("fleet") or [{}])[0].get("ip") or "localhost"

    # Group by target node: PARALLEL across nodes (independent GPUs), SEQUENTIAL within a
    # node (shared GPU — concurrent profiling would race).
    by_node: dict = {}
    for d in deps:
        by_node.setdefault(nodes_of(d)[0], []).append(d)
    print(f"{'APPLY' if execute else 'PLAN (dry-run)'} · {args.file} · {len(deps)} deployment(s) "
          f"· master={master} · {len(by_node)} node(s): parallel across, sequential within\n")
    for node, ds in by_node.items():
        print(f"node {node}:")
        for d in ds:
            port, cname = ports[d["name"]], container_of(d)
            cmd = " ".join(shlex.quote(p) for p in build_launch(d, port))
            print(f"  {cname} :{port}")
            print(f"      ssh {master} {shlex.quote(f'cd {args.remote_dir} && {cmd}')}")
        print()

    if not execute:
        print("(dry-run — re-run with --execute. Nodes load in parallel; each node sequential.)")
        return 0

    results: dict = {}
    lock = threading.Lock()

    def run_node(node: str, ds: list) -> None:
        for d in ds:
            port, cname = ports[d["name"]], container_of(d)
            if is_serving(node, port):                 # genuinely up -> leave it
                with lock:
                    results[cname] = "already serving (skipped)"
                    print(f"[{node}] {cname} already serving — skipped")
                continue
            _ssh(node, f"docker rm -f {cname} >/dev/null 2>&1")   # clear stale/crashed
            cmd = " ".join(shlex.quote(p) for p in build_launch(d, port))
            with lock:
                print(f"[{node}] starting {cname} :{port} ...")
            if _ssh(master, f"cd {args.remote_dir} && {cmd}").returncode != 0:
                with lock:
                    results[cname] = "LAUNCH FAILED"
                return  # stop this node's chain; don't pile on a shared GPU
            ok = wait_ready(node, port)
            with lock:
                results[cname] = "ready" if ok else "NOT READY (timeout)"
                print(f"[{node}] {cname} -> {results[cname]}")
            if not ok:
                return

    threads = [threading.Thread(target=run_node, args=(n, ds)) for n, ds in by_node.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    write_state(fleet, deps, ports)
    print("\n=== apply summary ===")
    for cname, status in results.items():
        print(f"  {cname:22} {status}")
    print("state written: .state/cluster.json")
    return 0 if all(s in ("ready", "already serving (skipped)") for s in results.values()) else 1


def cmd_down(args: argparse.Namespace) -> int:
    fleet = load_fleet(args.file)
    deps = fleet.get("deployments") or []
    for d in deps:
        node, cname = nodes_of(d)[0], container_of(d)
        cmd = f"docker rm -f {cname}"
        if args.execute:
            print(f"→ removing {cname} on {node}")
            _ssh(node, cmd, capture=False)
        else:
            print(f"[down] ssh {node} {shlex.quote(cmd)}")
    if not args.execute:
        print("(dry-run — re-run with --execute to stop containers)")
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    fleet = load_fleet(args.file)
    deps = fleet.get("deployments") or []
    target = args.name
    ports = assign_ports(deps)
    master = args.master or (fleet.get("fleet") or [{}])[0].get("ip") or "localhost"
    remote_dir = args.remote_dir

    found = False
    for d in deps:
        if d["name"] != target:
            continue
        found = True
        node, cname = nodes_of(d)[0], container_of(d)
        port = ports[d["name"]]

        print(f"Restarting {cname} on {node}:{port}")
        _ssh(node, f"docker rm -f {cname} >/dev/null 2>&1")
        cmd = " ".join(shlex.quote(p) for p in build_launch(d, port))
        print(f"[{node}] starting {cname} :{port} ...")
        if _ssh(master, f"cd {remote_dir} && {cmd}").returncode != 0:
            print(f"[{node}] LAUNCH FAILED")
            return 1
        ok = wait_ready(node, port)
        print(f"[{node}] {cname} -> {'ready' if ok else 'NOT READY (timeout)'}")
        return 0 if ok else 1

    if not found:
        print(f"Deployment not found: {target}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="spark", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="list downloaded HF models: size, quant, capabilities")
    s.add_argument("--hub", help="HF hub cache dir (default: $HF_HUB_CACHE / ~/.cache/huggingface/hub)")
    s.add_argument("--tokens", type=parse_tokens, metavar="N",
                   help="add a column: KV cache GB (fp8) to hold N total tokens (accepts 262k, 1M)")
    s.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    s.set_defaults(func=cmd_scan)

    a = sub.add_parser("apply", help="start spark.yaml deployments (dry-run unless --execute)")
    a.add_argument("-f", "--file", default="spark.yaml")
    a.add_argument("--master", help="control host that runs run-recipe.sh (default: first fleet node)")
    a.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="project dir on the master")
    a.add_argument("--execute", action="store_true", help="actually run (default: print plan)")
    a.set_defaults(func=cmd_apply)

    d = sub.add_parser("down", help="stop spark.yaml deployments (dry-run unless --execute)")
    d.add_argument("-f", "--file", default="spark.yaml")
    d.add_argument("--execute", action="store_true")
    d.set_defaults(func=cmd_down)

    r = sub.add_parser("restart", help="restart a specific model by name (always executes)")
    r.add_argument("-f", "--file", default="spark.yaml")
    r.add_argument("--master", help="control host (default: first fleet node)")
    r.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    r.add_argument("name", help="deployment name to restart")
    r.set_defaults(func=cmd_restart)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
