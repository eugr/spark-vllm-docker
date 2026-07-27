#!/usr/bin/env python3
"""
run-stack.py - Serve multiple vLLM recipes at once, each with its own placement.

A "stack" is a declarative YAML manifest (see stacks/*.yaml) listing several
recipes to serve at once, each in its own Docker container on its own port —
distributed across the cluster or pinned to a single node, per recipe. This
driver is a thin layer on top of run-recipe.py: it launches each recipe in the
listed order (which MUST be descending memory usage) and waits for each
recipe's /health endpoint to report ready before starting the next one.

Why the order and the health gate matter: vLLM profiles free memory at startup,
so the largest model must claim its slice against a clean pool first (and fail
fast if it won't fit). Health-gating each launch also serializes first-time
kernel compilation so co-located instances don't race on the shared compile
caches.

Scope: each recipe in a stack picks its own placement via a per-recipe
`placement:` key: 'all' distributes it across the cluster in .env's
CLUSTER_NODES (or runs it solo on this host when CLUSTER_NODES has fewer than
two nodes), or a single node IP pins it alone to that host -- this machine or
a remote one. A recipe with no 'placement:' defaults to 'all'. Every
distributed entry gets its own auto-assigned --master-port (base .env
MASTER_PORT, else 29501, plus the entry index; an explicit master_port:
overrides) so co-located distributed recipes don't collide.

Usage:
    ./run-stack.sh <stack> [--setup] [--dry-run] [--config FILE]
                                                    # bring the stack up
    ./run-stack.sh <stack> --status                # show each container + health
    ./run-stack.sh <stack> --stop                  # tear the whole stack down

Exit codes: 0 success, 1 error, 3 a recipe's cluster_only/solo_only flag
contradicts its resolved placement (CI keys on 3 to retry without a cluster).
"""

import argparse
import collections
import importlib.util
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_RECIPE = SCRIPT_DIR / "run-recipe.py"
LAUNCH_CLUSTER = SCRIPT_DIR / "launch-cluster.sh"
STACKS_DIR = SCRIPT_DIR / "stacks"
RECIPES_DIR = SCRIPT_DIR / "recipes"

# run-recipe.py as a module (the hyphenated filename rules out a plain import):
# shares its .env parser, node-list parser, and recipe-path resolution so the
# two drivers cannot drift on what a recipe reference or .env line means.
_spec = importlib.util.spec_from_file_location("run_recipe", RUN_RECIPE)
assert _spec is not None and _spec.loader is not None
run_recipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_recipe)

# Exit code for a cluster_only/solo_only recipe whose resolved placement
# contradicts it; CI retries such stacks without a cluster (see validate_modes).
EXIT_MODE_CONFLICT = 3

DEFAULT_HEALTH_TIMEOUT = 1200  # seconds to wait for a recipe to become ready
HEALTH_POLL_INTERVAL = 5       # seconds between /health polls
STATE_CHECK_EVERY = 6          # health polls between container-state probes
DEFAULT_PORT = 8000
DEFAULT_MASTER_PORT = 29501    # auto-assign base when .env sets no MASTER_PORT
STACK_VERSIONS = ("1",)        # manifest versions this driver understands

# Docker's accepted container-name charset.
CONTAINER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
# Characters in a 'volumes:' mapping that would not survive launch-cluster.sh's
# unquoted $DOCKER_ARGS expansion and its bash -c / ssh re-parse of the docker
# command line: whitespace word-splits, the rest are shell syntax or globs.
VOLUME_UNSAFE_RE = re.compile(r"""[\s$`;&|<>()'"\\*?\[\]{}]""")

# A resolved per-entry placement.
#   kind:      "distributed" | "single"
#   nodes:     distributed -> full cluster node list; single -> [] (unused)
#   host:      distributed -> None (API binds on the head/localhost)
#              single, this machine -> None (localhost)
#              single, remote node  -> the node IP string
#              (so a non-None host <=> the entry runs on a remote node)
#   source:    short human-readable reason, for banner + error messages
Placement = collections.namedtuple("Placement", "kind nodes host source")


# --------------------------------------------------------------------------- #
# Manifest loading / resolution
# --------------------------------------------------------------------------- #
def resolve_stack_path(name_or_path):
    """Resolve a stack manifest by path or by name under stacks/."""
    candidates = [
        Path(name_or_path),
        STACKS_DIR / name_or_path,
        STACKS_DIR / f"{name_or_path}.yaml",
        STACKS_DIR / f"{name_or_path}.yml",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"Stack manifest '{name_or_path}' not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def read_recipe_meta(recipe_path):
    """Read the recipe fields run-stack cares about: port and mode flags.

    Deliberately does NOT swallow parse errors: the mode flags gate
    validate_modes, so a recipe that cannot be parsed must not be silently
    treated as having no mode restriction.
    """
    data = yaml.safe_load(recipe_path.read_text()) or {}
    if not isinstance(data, dict):
        # Raise a yaml error so the call site's "could not be read" wrapper
        # converts it: a recipe whose top level is a list/scalar is just as
        # unreadable for our purposes as one that doesn't parse at all.
        raise yaml.YAMLError(
            f"top-level YAML is a {type(data).__name__}, not a mapping"
        )
    return {
        "port": (data.get("defaults") or {}).get("port", DEFAULT_PORT),
        "cluster_only": bool(data.get("cluster_only", False)),
        "solo_only": bool(data.get("solo_only", False)),
    }


def sanitize_name(recipe):
    """Derive a Docker-safe default container name from a recipe name."""
    base = Path(recipe).name
    # Strip only a manifest extension: Path.stem would truncate dotted recipe
    # names like 'qwen3.6-35b-a3b-nvfp4' to 'qwen3'.
    for ext in (".yaml", ".yml"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    cleaned = "".join(c if (c.isalnum() or c in "_.-") else "_" for c in base)
    return cleaned or "vllm_node"


def resolve_placement(raw_placement, env, config_path):
    """Resolve one entry's placement. 'all' -> distributed across CLUSTER_NODES
    (or solo-local if none). '<IP>' -> single node: local (host=None) if it is
    this machine, else that remote node. An IP must be a configured node or
    LOCAL_IP."""
    if raw_placement is not None and not isinstance(raw_placement, str):
        raise ValueError(
            f"'placement' must be 'all' or a single node-IP string, got "
            f"{raw_placement!r}. Arbitrary node subsets are not supported; use "
            "'all' or one IP from CLUSTER_NODES."
        )
    raw = (raw_placement or "all").strip()
    local = env.get("LOCAL_IP")
    cluster = run_recipe.parse_nodes(env.get("CLUSTER_NODES"))
    if raw == "all":
        # Distinguish an explicit 'placement: all' from the default: quoting a
        # key the manifest never set sends the reader hunting for it.
        how = "placement: all" if raw_placement is not None else "default placement: all"
        if len(cluster) > 1:
            return Placement("distributed", cluster, None, f"{how}, cluster from {config_path}")
        return Placement(
            "single", [], None, f"{how}, fewer than two cluster nodes configured"
        )
    if local and raw == local:
        return Placement("single", [], None, f"placement: {raw} (this host)")
    if raw in cluster:
        return Placement("single", [], raw, f"placement: {raw}")
    raise ValueError(
        f"placement '{raw}' is neither this host (LOCAL_IP={local!r}) nor one of "
        f"the configured cluster nodes ({', '.join(cluster) or 'none'}). "
        "Use 'all' or an IP from CLUSTER_NODES."
    )


def entry_host(entry, default_host):
    """Host to poll for /health: the placement node, else the default (localhost)."""
    p = entry["placement"]
    return p.host if p.host else default_host


def entry_container_state(entry):
    """Docker state of an entry's container, wherever it runs. Local for
    distributed/local placements, over ssh for a remote placement. Returns a
    docker state string, 'unreachable', or None (absent)."""
    p = entry["placement"]
    if p.host:
        return remote_container_state(p.host, entry["container_name"])
    return container_state(entry["container_name"])


def split_head_workers(nodes, env):
    """Partition nodes into (head, workers).

    launch-cluster.sh sets HEAD_IP to the local IP and requires it to appear
    somewhere in the node list. The head is therefore whichever configured
    node is THIS machine -- NOT nodes[0]. Returns (None, nodes) when LOCAL_IP
    is unknown, so callers degrade honestly instead of mislabelling a worker
    as the head.
    """
    local = env.get("LOCAL_IP")
    if local and local in nodes:
        return local, [n for n in nodes if n != local]
    return None, list(nodes)


def print_placement_summary(stack):
    for entry in stack["entries"]:
        p = entry["placement"]
        if p.kind == "distributed":
            where = f"distributed across {len(p.nodes)} nodes"
        elif p.host is None:
            where = "this host"
        else:
            where = f"node {p.host}"
        print(f"  {entry['recipe']:40s} {entry['container_name']:18s} -> {where}")
    # No banner when 'all' resolves solo for want of a cluster: 'all' means
    # every node available, so on a single-node host -- the common case -- it
    # is the expected placement, identical to pinning LOCAL_IP. The per-entry
    # lines above already name the target, and a cluster_only recipe in that
    # position is refused outright by validate_modes on the launch paths.


def placement_api_node(entry, env):
    """The node token an entry's API server binds on (for per-node uniqueness).
    Distributed workers are --headless, so a distributed entry's API is on the
    head; a local single placement is also the head; a remote single placement
    is its own node."""
    return entry["placement"].host or env.get("LOCAL_IP") or "local"


def parse_int_field(value, field, path, entry_no):
    """Validate an integer port-valued manifest field, rejecting YAML booleans
    (which are ints to Python: 'port: true' would become --port 1)."""
    if isinstance(value, bool):
        raise ValueError(
            f"Stack '{path}': entry #{entry_no}: '{field}' must be an integer, "
            f"got the YAML boolean {value!r}."
        )
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Stack '{path}': entry #{entry_no}: '{field}' must be an integer, "
            f"got {value!r}."
        ) from None
    if not 1 <= value <= 65535:
        raise ValueError(
            f"Stack '{path}': entry #{entry_no}: '{field}' must be between "
            f"1 and 65535, got {value}."
        )
    return value


def parse_volume_field(value, path, entry_no):
    """Validate one 'volumes:' mapping and resolve a relative host path.

    Docker syntax is host:container[:options]. A bare host token with no '/'
    is a *named volume* to Docker (volume names cannot contain '/'), so it is
    left alone. A relative path is resolved against the repo root, as 'mods:'
    are: Docker would otherwise resolve it against the caller's cwd, making the
    mount depend on where run-stack happened to be invoked from.

    The container path must be absolute -- Docker rejects a relative one, and
    catching it here beats discovering it mid-launch on a remote node. The same
    reasoning covers shell metacharacters: launch-cluster.sh expands the mapping
    unquoted inside $DOCKER_ARGS and re-parses the docker line through bash -c
    (and ssh for a remote node), so whitespace would word-split and '$' or ';'
    would execute rather than mount.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"Stack '{path}': entry #{entry_no}: 'volumes' entries must be "
            f"strings in Docker's host:container[:options] form, got {value!r}."
        )
    if VOLUME_UNSAFE_RE.search(value):
        raise ValueError(
            f"Stack '{path}': entry #{entry_no}: volume {value!r} contains "
            "whitespace or a shell metacharacter, which would not survive the "
            "docker command line. Use a path without them."
        )
    parts = value.split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Stack '{path}': entry #{entry_no}: volume {value!r} must be in "
            "Docker's host:container[:options] form."
        )
    host, container = parts[0], parts[1]
    if not container.startswith("/"):
        raise ValueError(
            f"Stack '{path}': entry #{entry_no}: volume {value!r} has container "
            f"path {container!r}, which must be absolute (start with '/')."
        )
    if host.startswith("~"):
        try:
            host = str(Path(host).expanduser())
        except RuntimeError:
            # expanduser() raises when ~user names an account with no home.
            raise ValueError(
                f"Stack '{path}': entry #{entry_no}: volume {value!r} has host "
                f"path {host!r}, whose home directory could not be resolved."
            ) from None
    elif "/" in host and not Path(host).is_absolute():
        host = str((SCRIPT_DIR / host).resolve())
    return ":".join([host, container, *parts[2:]])


def is_distributed(entry):
    return entry["placement"].kind == "distributed"


def load_stack(name_or_path, master_base, env, config_path):
    """Load and normalize a stack manifest into a list of resolved entries."""
    path = resolve_stack_path(name_or_path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Stack '{path}': top-level YAML must be a mapping "
            f"(got a {type(data).__name__})."
        )

    version = data.get("stack_version")
    if version is not None and str(version) not in STACK_VERSIONS:
        raise ValueError(
            f"Stack '{path}': unsupported stack_version {version!r} "
            f"(supported: {', '.join(STACK_VERSIONS)})."
        )
    if "solo_only" in data:
        print(
            f"Warning: stack '{path}': top-level 'solo_only' is no longer "
            "supported and is ignored; use per-recipe 'placement:' instead.",
            file=sys.stderr,
        )

    raw_entries = data.get("recipes")
    if not raw_entries:
        raise ValueError(f"Stack '{path}' has no 'recipes:' list.")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Stack '{path}': 'recipes:' must be a list.")

    entries = []
    seen_names = {}
    for i, raw in enumerate(raw_entries):
        if isinstance(raw, str):
            raw = {"recipe": raw}
        if not isinstance(raw, dict) or not raw.get("recipe"):
            raise ValueError(
                f"Stack '{path}': entry #{i + 1} must have a 'recipe' field."
            )
        recipe = raw["recipe"]
        recipe_path = run_recipe.resolve_recipe_path(Path(recipe))
        if recipe_path is None:
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: recipe '{recipe}' not found "
                f"(looked for a file at that path and under {RECIPES_DIR}/)."
            )
        explicit_name = raw.get("container_name")
        cname = explicit_name or sanitize_name(recipe)
        if not CONTAINER_NAME_RE.match(cname):
            derived = (
                "" if explicit_name
                else f" (derived from recipe '{recipe}'; set 'container_name' explicitly)"
            )
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: container_name '{cname}' is not "
                "a valid Docker container name "
                f"(must match [a-zA-Z0-9][a-zA-Z0-9_.-]*){derived}."
            )
        try:
            meta = read_recipe_meta(recipe_path)
        except (OSError, yaml.YAMLError) as e:
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: recipe '{recipe}' could not be "
                f"read ({type(e).__name__}). A recipe that cannot be parsed "
                "cannot be checked for cluster_only/solo_only, so this stack is "
                "refused rather than launched with unknown mode requirements."
            ) from None
        port = raw.get("port")
        if port is None:
            port = meta["port"]
        port = parse_int_field(port, "port", path, i + 1)

        if cname in seen_names:
            raise ValueError(
                f"Stack '{path}': duplicate container_name '{cname}' "
                f"(entries #{seen_names[cname] + 1} and #{i + 1}). "
                "Set an explicit 'container_name' for one of them."
            )
        seen_names[cname] = i

        raw_mp = raw.get("master_port")
        if raw_mp is None:
            master_port = master_base + i
        else:
            # The range check matters: run-recipe.py drops a falsy
            # --master-port entirely, so 0 would silently fall back to the
            # 29501 default and collide with the entry using it.
            master_port = parse_int_field(raw_mp, "master_port", path, i + 1)
        try:
            placement = resolve_placement(raw.get("placement"), env, config_path)
        except ValueError as e:
            raise ValueError(f"Stack '{path}': entry #{i + 1} ('{recipe}'): {e}") from None

        # Resolve mod paths relative to the repo root so run-recipe.py's
        # cwd-relative resolution doesn't depend on where run-stack is invoked.
        mods = []
        for m in raw.get("mods", []) or []:
            mp = Path(m)
            mods.append(str(mp if mp.is_absolute() else (SCRIPT_DIR / mp)))

        raw_volumes = raw.get("volumes") or []
        if not isinstance(raw_volumes, list):
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: 'volumes' must be a list of "
                f"host:container[:options] strings, got {raw_volumes!r}."
            )
        volumes = [parse_volume_field(v, path, i + 1) for v in raw_volumes]

        entries.append(
            {
                "recipe": recipe,
                "container_name": cname,
                "port": port,
                "master_port": master_port,
                "gpu_mem": raw.get("gpu_mem"),
                "mods": mods,
                "volumes": volumes,
                "extra_args": [str(a) for a in (raw.get("extra_args") or [])],
                "cluster_only": meta["cluster_only"],
                "solo_only": meta["solo_only"],
                "placement": placement,
            }
        )

    node_ports = {}
    for i, entry in enumerate(entries):
        node = placement_api_node(entry, env)
        key = (node, entry["port"])
        if key in node_ports:
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: port {entry['port']} on node "
                f"{node} is already used by entry #{node_ports[key] + 1}. Ports "
                "must be unique per node.")
        node_ports[key] = i

    dist_mports = {}
    for i, entry in enumerate(entries):
        if not is_distributed(entry):
            continue
        mp = entry["master_port"]
        if mp in dist_mports:
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: master_port {mp} duplicates "
                f"entry #{dist_mports[mp] + 1}. Distributed recipes share the head, "
                "so each needs a distinct coordination port.")
        dist_mports[mp] = i

    head = env.get("LOCAL_IP") or "local"
    head_api_ports = {e["port"] for e in entries if placement_api_node(e, env) == head}
    for i, entry in enumerate(entries):
        if is_distributed(entry) and entry["master_port"] in head_api_ports:
            raise ValueError(
                f"Stack '{path}': entry #{i + 1}: master_port {entry['master_port']} "
                "collides with an API port on the head node.")

    raw_timeout = data.get("health_timeout", DEFAULT_HEALTH_TIMEOUT)
    try:
        health_timeout = int(raw_timeout)
    except (TypeError, ValueError):
        raise ValueError(
            f"Stack '{path}': 'health_timeout' must be an integer (seconds), "
            f"got {raw_timeout!r}."
        ) from None

    return {
        "path": path,
        "name": data.get("name", path.stem),
        "health_timeout": health_timeout,
        "entries": entries,
    }


class ModeConflictError(ValueError):
    """A recipe's cluster_only/solo_only flag contradicts its resolved
    placement. Reported with EXIT_MODE_CONFLICT so CI can tell 'this stack is
    legitimately single-host' apart from a genuinely broken manifest."""


def validate_modes(stack):
    """Reject recipes whose mode requirement contradicts their resolved placement."""
    for i, entry in enumerate(stack["entries"], 1):
        p = entry["placement"]
        if entry["cluster_only"] and p.kind == "single":
            raise ModeConflictError(
                f"Stack '{stack['path']}': entry #{i}: recipe '{entry['recipe']}' "
                f"is cluster_only, but its placement resolved to a single node "
                f"({p.source}). Use 'placement: all' with a configured cluster.")
        if entry["solo_only"] and p.kind == "distributed":
            raise ModeConflictError(
                f"Stack '{stack['path']}': entry #{i}: recipe '{entry['recipe']}' "
                f"is solo_only, but its placement resolved to distributed "
                f"({p.source}). Pin it with 'placement: <node-IP>'.")


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #
def placement_flags(entry):
    """Placement -> CLI flags, shared by launch (run-recipe.py) and stop
    (launch-cluster.sh) so the two cannot diverge. A distributed entry must
    pass -n and NEVER --solo: --solo skips launch-cluster.sh's ssh worker
    loop, which on stop would strand worker containers."""
    p = entry["placement"]
    if p.kind == "distributed":
        return ["-n", ",".join(p.nodes)]
    if p.host is None:
        return ["--solo"]
    return ["--placement-node", p.host]


def recipe_args(entry, setup, config_path=None, daemon=True):
    """Args to run-recipe.py for one entry (everything after the program name)."""
    args = [entry["recipe"], *placement_flags(entry)]
    if is_distributed(entry):
        args += ["--master-port", str(entry["master_port"])]
    args += ["--name", entry["container_name"]]
    if daemon:
        args.append("-d")
    args += ["--port", str(entry["port"])]
    if entry["gpu_mem"] is not None:
        args += ["--gpu-mem", str(entry["gpu_mem"])]
    for m in entry["mods"]:
        args += ["--apply-mod", m]
    for v in entry["volumes"]:
        args += ["-v", v]
    if setup:
        args.append("--setup")
    if config_path is not None:
        args += ["--config", str(config_path)]
    if entry["extra_args"]:
        args += ["--", *entry["extra_args"]]
    return args


def health_url(host, port):
    return f"http://{host}:{port}/health"


def is_healthy(host, port, timeout=2):
    try:
        with urllib.request.urlopen(health_url(host, port), timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def container_state(cname):
    """Return the container's Docker state ('running', 'exited', ...) or None if absent."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", cname],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def wait_healthy(entry, host, timeout):
    """Poll /health until ready. Returns None on success, else a failure reason.
    Watches the container on its placement node so a crash fails fast. The
    state probe runs only every STATE_CHECK_EVERY polls: for a remote entry
    each probe is a fresh ssh round-trip, too costly to pay every 5s."""
    cname = entry["container_name"]
    poll_host = entry_host(entry, host)
    deadline = time.monotonic() + timeout
    url = health_url(poll_host, entry["port"])
    print(f"    waiting for {url} (timeout {timeout}s)...", flush=True)
    polls = 0
    while time.monotonic() < deadline:
        if is_healthy(poll_host, entry["port"]):
            print("    ready.", flush=True)
            return None
        if polls % STATE_CHECK_EVERY == 0:
            state = entry_container_state(entry)
            if state not in ("running", "unreachable"):
                return (
                    f"container '{cname}' "
                    + (f"is in state '{state}'" if state else "no longer exists")
                    + " before becoming healthy"
                )
        polls += 1
        time.sleep(HEALTH_POLL_INTERVAL)
    hint = "" if not is_distributed(entry) else (
        " (a worker container may have died; run --status to check every node)"
    )
    return f"did not become healthy within {timeout}s{hint}"


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #
def cmd_dry_run(stack, host, setup, explicit_config):
    print(f"=== Stack: {stack['name']} ({len(stack['entries'])} recipes) ===")
    print(f"Health timeout per recipe: {stack['health_timeout']}s")
    print_placement_summary(stack)
    print("Load order (as listed == descending memory usage):")
    print()
    for i, entry in enumerate(stack["entries"], 1):
        cmd = [
            "./run-recipe.py",
            *recipe_args(entry, setup=setup, config_path=explicit_config),
        ]
        print(f"{i}. {shlex.join(cmd)}")
        print(f"   then wait for {health_url(entry_host(entry, host), entry['port'])}")
        print()
    return 0


def cmd_up(stack, host, setup, explicit_config):
    print(f"=== Bringing up stack: {stack['name']} ===", flush=True)
    print_placement_summary(stack)
    for i, entry in enumerate(stack["entries"], 1):
        cname = entry["container_name"]
        port = entry["port"]
        poll_host = entry_host(entry, host)
        print(
            f"\n[{i}/{len(stack['entries'])}] {entry['recipe']} "
            f"-> container '{cname}', port {port}",
            flush=True,
        )
        # If something is already answering /health on this port: our own
        # running container means this entry is already up (re-`up` is a
        # no-op for it); anything else owns the port, and vLLM would fail to
        # bind (host networking) while the health gate saw the impostor as
        # "ready" — refuse up front.
        if is_healthy(poll_host, port):
            if entry_container_state(entry) == "running":
                print(
                    f"    '{cname}' is already up and healthy on port {port}; "
                    "skipping.",
                    flush=True,
                )
                continue
            print(
                f"\nError: port {port} is already serving /health but container "
                f"'{cname}' is not running — another process owns that port. "
                f"Recipes started before this one are left running.",
                file=sys.stderr,
            )
            return 1
        cmd = [
            sys.executable,
            str(RUN_RECIPE),
            *recipe_args(entry, setup=setup, config_path=explicit_config),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"\nError: launching '{entry['recipe']}' failed "
                f"(run-recipe.py exit {result.returncode}). "
                f"Recipes started before this one are left running.",
                file=sys.stderr,
            )
            return result.returncode
        failure = wait_healthy(entry, host, stack["health_timeout"])
        if failure:
            print(
                f"\nError: '{entry['recipe']}' (container '{cname}', port {port}) "
                f"{failure}. Check logs: docker logs {cname}. "
                f"Recipes started before this one are left running.",
                file=sys.stderr,
            )
            return 1
    print(f"\n=== Stack '{stack['name']}' is up. ===")
    for entry in stack["entries"]:
        print(f"  {entry['recipe']:40s} http://{entry_host(entry, host)}:{entry['port']}")
    return 0


def stop_command(entry, config_path=None):
    """argv to tear down one entry per its placement. Forwards --config so
    launch-cluster resolves LOCAL_IP from the same env that decided the
    placement (needed to tell local from remote)."""
    cmd = [str(LAUNCH_CLUSTER), *placement_flags(entry)]
    cmd += ["--name", entry["container_name"], "stop"]
    if config_path is not None:
        cmd += ["--config", str(config_path)]
    return cmd


def remote_container_state(host, cname):
    """Container state on a remote host: a docker state, 'unreachable', or None."""
    result = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5", host,
            f"docker inspect -f '{{{{.State.Status}}}}' {shlex.quote(cname)}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 255:
        return "unreachable"
    return result.stdout.strip() if result.returncode == 0 else None


def cmd_stop(stack, explicit_config):
    print(f"=== Stopping stack: {stack['name']} ===")
    print_placement_summary(stack)
    rc = 0
    # Stop in reverse (last-started first) as a courtesy; order is not critical.
    for entry in reversed(stack["entries"]):
        print(f"  stopping '{entry['container_name']}'...")
        result = subprocess.run(stop_command(entry, explicit_config))
        if result.returncode != 0:
            rc = result.returncode
    return rc


def cmd_status(stack, host, env):
    print(f"=== Stack: {stack['name']} ===")
    print_placement_summary(stack)
    header = f"{'RECIPE':40s} {'CONTAINER':20s} {'PORT':>6s} {'STATE':>10s} {'HEALTH':>8s}"
    print(header)
    print("-" * len(header))
    for entry in stack["entries"]:
        cname = entry["container_name"]
        port = entry["port"]
        raw_state = entry_container_state(entry)
        state_str = raw_state or "absent"
        health = "ok" if is_healthy(entry_host(entry, host), port) else "-"
        print(
            f"{entry['recipe']:40s} {cname:20s} {port:>6d} "
            f"{state_str:>10s} {health:>8s}"
        )
        p = entry["placement"]
        if p.kind == "distributed":
            head, workers = split_head_workers(p.nodes, env)
            # Reuse the state already fetched above — the distributed head is
            # this machine, so entry_container_state already inspected it.
            print(f"    [HEAD] {head or '?'}: {raw_state or 'absent'}")
            for worker in workers:
                worker_state = remote_container_state(worker, cname)
                print(f"    [WORKER] {worker}: {worker_state or 'absent'}")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Serve multiple vLLM recipes at once, each with its own "
        "placement: distributed across the cluster or pinned to one node.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stack", nargs="?", help="Stack manifest: name under stacks/ or a path"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available stacks"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Pass --setup to each recipe (build image + download model if missing)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to poll for /health (default: localhost)",
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        metavar="FILE",
        help="Path to .env configuration file (default: .env in script directory)",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered run-recipe.py commands without executing",
    )
    action.add_argument("--stop", action="store_true", help="Stop the whole stack")
    action.add_argument(
        "--status", action="store_true", help="Show each container's state + health"
    )
    args = parser.parse_args()

    if args.list:
        if not STACKS_DIR.is_dir():
            print("No stacks/ directory found.")
            return 0
        stacks = sorted(
            p.stem for p in STACKS_DIR.glob("*.y*ml") if p.suffix in (".yaml", ".yml")
        )
        print("Available stacks:")
        for s in stacks:
            print(f"  {s}")
        return 0

    if not args.stack:
        parser.error("a stack name/path is required (or use --list)")

    explicit_config = (
        Path(args.config_file).resolve() if args.config_file else None
    )
    if explicit_config is not None and not explicit_config.is_file():
        # A typo'd --config silently loading as an empty env would flip a
        # cluster-sized stack onto one host; only the *default* .env may be
        # absent without complaint.
        print(
            f"Error: --config file not found: {args.config_file}",
            file=sys.stderr,
        )
        return 1
    config_path = explicit_config or (SCRIPT_DIR / ".env")
    env = run_recipe.load_env_file(config_path)
    try:
        master_base = int(env.get("MASTER_PORT") or DEFAULT_MASTER_PORT)
    except (TypeError, ValueError):
        print(
            f"Error: MASTER_PORT in {config_path} must be an integer, "
            f"got {env.get('MASTER_PORT')!r}.",
            file=sys.stderr,
        )
        return 1

    try:
        stack = load_stack(args.stack, master_base, env, config_path)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        if not (args.stop or args.status):
            # Launch paths only: --stop/--status must inspect/tear down even
            # a stack whose recipes' mode flags contradict its placements.
            validate_modes(stack)
    except ModeConflictError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_MODE_CONFLICT

    if args.stop:
        return cmd_stop(stack, explicit_config)
    if args.status:
        return cmd_status(stack, args.host, env)
    if args.dry_run:
        return cmd_dry_run(stack, args.host, args.setup, explicit_config)
    return cmd_up(stack, args.host, args.setup, explicit_config)


if __name__ == "__main__":
    sys.exit(main())
