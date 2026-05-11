#!/usr/bin/env python3
"""
run-recipe.py - One-click model deployment using YAML recipes

This script provides a high-level interface for deploying models with
pre-configured settings. It handles:
- Model download from HuggingFace (optional)
- Container building and distribution to worker nodes
- Mod application
- Launch script generation
- Both solo (single node) and cluster deployments

Usage:
    ./run-recipe.py recipes/glm-4.7-nvfp4.yaml
    ./run-recipe.py glm-4.7-nvfp4 --port 9000 --solo
    ./run-recipe.py minimax-m2-awq --setup  # Full setup: build + download + run
    ./run-recipe.py --list

================================================================================
ARCHITECTURE OVERVIEW (for developers extending this script)
================================================================================

DEPLOYMENT PIPELINE:
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │  CLI Args  →  Load Recipe  →  Resolve Nodes  →  Build  →  Download  →  Run  │
    └─────────────────────────────────────────────────────────────────────────┘

KEY ABSTRACTIONS:
    - Recipe (YAML): Declarative model configuration (see load_recipe docstring)
    - Phases: Build, Download, Run - each can run independently (--build-only, etc.)
    - Nodes: Head (first) + Workers (rest) - images/models copied to workers

EXTENSION POINTS:

    1. ADD NEW RECIPE FIELDS:
       - Update load_recipe() to validate/set defaults
       - Use the field in generate_launch_script() or main()
       - Document in recipe YAML schema below

    2. ADD NEW CLI OPTIONS:
       - Add to appropriate argument group in main()
       - Handle in the corresponding phase (build/download/run)
       - Pass to generate_launch_script() via overrides dict if needed

    3. ADD NEW DEPLOYMENT PHASES:
       - Follow the pattern: check if needed → dry-run print → execute
       - Insert between existing phases in main()
       - Add corresponding --phase-only flag

    4. SUPPORT NEW MODEL SOURCES:
       - Add detection logic in download_model() or check_model_exists()
       - Create new download script or handle inline

    5. SUPPORT NEW CONTAINER RUNTIMES:
       - Modify check_image_exists() and build_image()
       - May need to update launch-cluster.sh as well

RECIPE YAML SCHEMA:
    name: str              # Required: Human-readable name
    recipe_version: str    # Required: Recipe schema version (e.g., '1'). Used by run-recipe.py
                           #           to check compatibility and available features.
    container: str         # Required: Docker image tag
    command: str           # Required: vLLM serve command with {placeholders}
    description: str       # Optional: Brief description
    model: str             # Optional: HuggingFace model ID for --setup
    mods: list[str]        # Optional: Mod directories to apply
    defaults: dict         # Optional: Default values for command placeholders
    env: dict              # Optional: Environment variables
    build_args: list[str]  # Optional: Args for build-and-copy.sh
    cluster_only: bool     # Optional: Require cluster mode (default: false)
    solo_only: bool        # Optional: Require solo mode (default: false)

RECIPE VERSION HISTORY:
    Version 1 (default): Initial schema with all fields above supported.

RELATED FILES:
    - run-recipe.sh: Bash wrapper that ensures Python deps are installed
    - recipes/*.yaml: Recipe definitions
    - examples/: Example launch scripts for direct use with launch-cluster.sh
    - launch-cluster.sh: Low-level container orchestration
    - build-and-copy.sh: Docker build and distribution
    - hf-download.sh: HuggingFace model download and sync
    - autodiscover.sh: Network topology detection
"""

import argparse
import os
import subprocess
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent.resolve()
RECIPES_DIR = SCRIPT_DIR / "recipes"
LAUNCH_SCRIPT = SCRIPT_DIR / "launch-cluster.sh"
BUILD_SCRIPT = SCRIPT_DIR / "build-and-copy.sh"
DOWNLOAD_SCRIPT = SCRIPT_DIR / "hf-download.sh"
AUTODISCOVER_SCRIPT = SCRIPT_DIR / "autodiscover.sh"
ENV_FILE = None  # Will be set from CLI argument or default


def load_recipe(recipe_path: Path) -> dict[str, Any]:
    """
    Load and validate a recipe YAML file.

    This function handles recipe resolution from multiple locations and validates
    required fields. Recipes are the core configuration format for deployments.

    EXTENSIBILITY:
    - To add new required fields: Add to the 'required' list below
    - To add new optional fields with defaults: Add to the setdefault() calls at the end
    - Recipe search order: exact path -> recipes/ dir -> with .yaml -> with .yml

    RECIPE SCHEMA:
        name (str, required): Human-readable name for the recipe
        recipe_version (str, required): Schema version for compatibility checking.
            Used by run-recipe.py to determine which features are available.
            Current version: '1'. Bump when adding new recipe fields.
        container (str, required): Docker image tag to use (e.g., 'vllm-node-mxfp4')
        command (str, required): vLLM serve command template with {placeholders}
        description (str, optional): Brief description shown in --list
        model (str, optional): HuggingFace model ID for --setup downloads
        mods (list[str], optional): List of mod directories to apply (e.g., 'mods/fix-glm')
        defaults (dict, optional): Default values for command placeholders
        env (dict, optional): Environment variables to export before running
        build_args (list[str], optional): Extra args for build-and-copy.sh (e.g., ['-f', 'Dockerfile.mxfp4'])
        cluster_only (bool, optional): If True, recipe cannot run in solo mode
        solo_only (bool, optional): If True, recipe cannot run in cluster mode

    Args:
        recipe_path: Path object pointing to YAML file or just recipe name

    Returns:
        Validated recipe dictionary with all fields populated (defaults applied)

    Raises:
        SystemExit: If recipe not found or validation fails
    """
    if not recipe_path.exists():
        # Try candidates in order: add extension to original path first,
        # then fall back to flat recipes/ directory (for bare recipe names)
        candidates = [
            Path(str(recipe_path) + ".yaml"),
            Path(str(recipe_path) + ".yml"),
            RECIPES_DIR / recipe_path.name,
            RECIPES_DIR / f"{recipe_path.name}.yaml",
            RECIPES_DIR / f"{recipe_path.name}.yml",
            RECIPES_DIR / f"{recipe_path.stem}.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                recipe_path = candidate
                break
        else:
            print(f"Error: Recipe not found: {recipe_path}")
            print(f"Searched in: {recipe_path}, {RECIPES_DIR}")
            sys.exit(1)

    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)

    # Validate required fields
    required = ["name", "recipe_version", "container", "command"]
    for field in required:
        if field not in recipe:
            print(f"Error: Recipe missing required field: {field}")
            sys.exit(1)

    # Set defaults for optional fields
    recipe.setdefault("description", "")
    recipe.setdefault("model", None)
    recipe.setdefault("mods", [])
    recipe.setdefault("defaults", {})
    recipe.setdefault("env", {})
    recipe.setdefault("cluster_only", False)
    recipe.setdefault("solo_only", False)

    # Validate recipe version compatibility
    # EXTENSIBILITY: When adding new schema versions, update SUPPORTED_VERSIONS
    # and add migration/compatibility logic below
    SUPPORTED_VERSIONS = ["1"]
    recipe_ver = str(recipe["recipe_version"])
    if recipe_ver not in SUPPORTED_VERSIONS:
        print(
            f"Warning: Recipe uses schema version '{recipe_ver}', but this run-recipe.py supports: {SUPPORTED_VERSIONS}"
        )
        print("Some features may not work correctly. Consider updating run-recipe.py.")

    return recipe


def list_recipes() -> None:
    """
    List all available recipes with their metadata.

    Scans the recipes/ directory for YAML files and displays key information.
    Used by the --list CLI option.

    EXTENSIBILITY:
    - To show additional fields: Add them to the print statements in the loop
    - To support different output formats (e.g., JSON): Add a format parameter
    - Recipe directory is defined by RECIPES_DIR constant at module level
    """
    if not RECIPES_DIR.exists():
        print("No recipes directory found.")
        return

    recipes = sorted(RECIPES_DIR.glob("*.yaml"))
    if not recipes:
        print("No recipes found in recipes/ directory.")
        return

    print("Available recipes:\n")
    for recipe_path in recipes:
        try:
            recipe = load_recipe(recipe_path)
            name = recipe.get("name", recipe_path.stem)
            recipe_version = recipe.get("recipe_version", "1")
            desc = recipe.get("description", "")
            container = recipe.get("container", "vllm-node")
            build_args = recipe.get("build_args", [])
            model = recipe.get("model", "")
            mods = recipe.get("mods", [])
            cluster_only = recipe.get("cluster_only", False)
            solo_only = recipe.get("solo_only", False)

            print(f"  {recipe_path.name}")
            print(f"    Name: {name}")
            if desc:
                print(f"    Description: {desc}")
            if model:
                print(f"    Model: {model}")
            if cluster_only:
                print("    Cluster only: Yes")
            if solo_only:
                print("    Solo only: Yes")
            print(f"    Container: {container}")
            if build_args:
                print(f"    Build args: {' '.join(build_args)}")
            if mods:
                print(f"    Mods: {', '.join(mods)}")
            print()
        except Exception as e:
            print(f"  {recipe_path.name} (error loading: {e})")
            print()


def check_image_exists(image: str, host: str | None = None) -> bool:
    """
    Check if a Docker image exists locally or on a remote host.

    Used to avoid redundant builds and to verify cluster nodes have the image.

    EXTENSIBILITY:
    - To support other container runtimes (podman): Modify the docker command
    - To add image version/digest checking: Parse 'docker image inspect' JSON output
    - For custom SSH options: Modify the ssh command array

    Args:
        image: Docker image tag to check (e.g., 'vllm-node-mxfp4')
        host: Optional remote hostname/IP. If None, checks locally.

    Returns:
        True if image exists, False otherwise
    """
    if host:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                host,
                f"docker image inspect '{image}'",
            ],
            capture_output=True,
        )
    else:
        result = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True
        )
    return result.returncode == 0


def build_image(
    image: str, copy_to: list[str] | None = None, build_args: list[str] | None = None
) -> bool:
    """
    Build the container image using build-and-copy.sh.

    Delegates to the build-and-copy.sh script which handles multi-stage builds,
    cache optimization, and distribution to worker nodes.

    EXTENSIBILITY:
    - To add new build options: Add them to build_args in the recipe's build_args field
    - To support different Dockerfiles: Use build_args = ['-f', 'Dockerfile.custom']
    - To add build-time secrets: Modify cmd array to include --secret flags
    - To add progress callbacks: Capture subprocess output line-by-line

    BUILD_ARGS EXAMPLES:
        ['-f', 'Dockerfile.mxfp4']  - Use alternate Dockerfile
        ['--no-cache']               - Force full rebuild
        ['--build-arg', 'VAR=value'] - Pass build-time variables

    Args:
        image: Target image tag
        copy_to: List of worker hostnames to copy image to after build
        build_args: Extra arguments passed to build-and-copy.sh

    Returns:
        True if build (and copy) succeeded, False otherwise
    """
    if not BUILD_SCRIPT.exists():
        print(f"Error: Build script not found: {BUILD_SCRIPT}")
        return False

    cmd = [str(BUILD_SCRIPT), "-t", image]
    if build_args:
        cmd.extend(build_args)
    if copy_to:
        cmd.extend(["--copy-to", ",".join(copy_to), "--copy-parallel"])

    print(f"Building image '{image}'...")
    if build_args:
        print(f"Build args: {' '.join(build_args)}")
    if copy_to:
        print(f"Will copy to: {', '.join(copy_to)}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def download_model(model: str, copy_to: list[str] | None = None) -> bool:
    """
    Download model from HuggingFace using hf-download.sh.

    Delegates to hf-download.sh which handles HF authentication, caching,
    and rsync to worker nodes.

    EXTENSIBILITY:
    - To support other model sources: Create a new download script and switch based on model URL
    - To add download progress: Capture subprocess output
    - To support private models: hf-download.sh uses HF_TOKEN env var
    - To add model verification: Check sha256 of downloaded files

    Args:
        model: HuggingFace model ID (e.g., 'Salyut1/GLM-4.7-NVFP4')
        copy_to: List of worker hostnames to copy model cache to

    Returns:
        True if download (and copy) succeeded, False otherwise
    """
    if not DOWNLOAD_SCRIPT.exists():
        print(f"Error: Download script not found: {DOWNLOAD_SCRIPT}")
        return False

    cmd = [str(DOWNLOAD_SCRIPT), model]
    if copy_to:
        cmd.extend(["--copy-to", ",".join(copy_to), "--copy-parallel"])

    print(f"Downloading model '{model}'...")
    if copy_to:
        print(f"Will copy to: {', '.join(copy_to)}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def check_model_exists(model: str) -> bool:
    """
    Check if a model exists in the HuggingFace cache.

    Checks the standard HF cache location for completed downloads.

    EXTENSIBILITY:
    - To support custom cache locations: Add HF_HOME env var support
    - To verify model integrity: Check for complete snapshot with config.json
    - To support other model sources: Add URL/path prefix detection

    Args:
        model: HuggingFace model ID (e.g., 'org/model-name')

    Returns:
        True if model appears to be fully downloaded, False otherwise
    """
    # Convert model name to cache directory format
    # e.g., "Salyut1/GLM-4.7-NVFP4" -> "models--Salyut1--GLM-4.7-NVFP4"
    cache_name = f"models--{model.replace('/', '--')}"
    cache_path = Path.home() / ".cache" / "huggingface" / "hub" / cache_name

    if cache_path.exists():
        # Check for snapshots directory which indicates complete download
        snapshots = cache_path / "snapshots"
        if snapshots.exists() and any(snapshots.iterdir()):
            return True
    return False


def generate_launch_script(
    recipe: dict[str, Any],
    overrides: dict[str, Any],
    is_solo: bool = False,
    extra_args: list[str] | None = None,
    no_ray: bool = False,
) -> str:
    """
    Generate a bash launch script from the recipe.

    Creates a self-contained bash script that runs inside the container.
    Handles template substitution, environment variables, and solo mode adjustments.

    EXTENSIBILITY:
    - To add new template variables: Add them to recipe['defaults'] or CLI overrides
    - To add pre/post hooks: Add 'pre_command'/'post_command' fields to recipe schema
    - To add conditional logic: Use Jinja2 templating instead of str.format()
    - To support GPU selection: Add CUDA_VISIBLE_DEVICES to env handling

    TEMPLATE VARIABLES (use {variable_name} in recipe command):
        port: API server port (default from recipe)
        host: API server bind address
        tensor_parallel: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
        (custom variables can be added via recipe defaults)

    SOLO MODE BEHAVIOR:
        - Removes '--distributed-executor-backend ray' lines
        - Typically sets tensor_parallel=1 (handled by caller)

    EXTRA ARGS:
        - Appended verbatim to the end of the vLLM command
        - Allows passing any vLLM argument not covered by template variables
        - vLLM uses "last wins" semantics for duplicate arguments

    Args:
        recipe: Loaded recipe dictionary
        overrides: CLI-provided parameter overrides (take precedence over defaults)
        is_solo: If True, strip distributed executor configuration
        extra_args: Additional arguments to append to vLLM command (after --)

    Returns:
        Complete bash script content as string

    Raises:
        SystemExit: If required template variables are missing
    """
    # Merge defaults with overrides
    params = {**recipe.get("defaults", {}), **overrides}

    # Build the script
    lines = ["#!/bin/bash", f"# Generated from recipe: {recipe['name']}", ""]

    # Add environment variables
    env_vars = recipe.get("env", {})
    if env_vars:
        lines.append("# Environment variables")
        for key, value in env_vars.items():
            lines.append(f'export {key}="{value}"')
        lines.append("")

    # Format the command with parameters
    command = recipe["command"]
    try:
        command = command.format(**params)
    except KeyError as e:
        print(f"Error: Missing parameter in recipe command: {e}")
        print(f"Available parameters: {list(params.keys())}")
        sys.exit(1)

    # In solo or no-ray mode, remove --distributed-executor-backend
    # (not needed for solo; no-ray uses PyTorch distributed instead)
    if is_solo or no_ray:
        import re

        # Remove just the flag and its value, not the whole line
        command = re.sub(r"--distributed-executor-backend\s+\S+", "", command)
        # Remove lines that are now empty or just a backslash continuation
        lines_list = command.split("\n")
        filtered_lines = [line for line in lines_list if line.strip() not in ("", "\\")]
        command = "\n".join(filtered_lines)

    # Remove trailing backslash if present
    command = command.rstrip()
    if command.endswith("\\"):
        command = command.rstrip("\\\n").rstrip()

    # Append extra args if provided (after --)
    if extra_args:
        # Join extra args and append to command
        extra_args_str = " ".join(shlex.quote(a) for a in extra_args)
        command = command + " " + extra_args_str

    lines.append("# Run the model")
    lines.append(command.strip())
    lines.append("")

    return "\n".join(lines)


def parse_nodes(nodes_arg: str | None) -> list[str]:
    """
    Parse comma-separated node list.

    Simple utility to split node specifications. The first node is
    always treated as the head node for cluster deployments.

    Args:
        nodes_arg: Comma-separated string like '192.168.1.1,192.168.1.2'

    Returns:
        List of stripped node identifiers, empty list if input is None/empty
    """
    if not nodes_arg:
        return []
    return [n.strip() for n in nodes_arg.split(",") if n.strip()]


def get_worker_nodes(nodes: list[str]) -> list[str]:
    """
    Get worker nodes (all nodes except the first/head node).

    In a Ray cluster, the first node runs the head process.
    Workers are all subsequent nodes that join the cluster.

    Args:
        nodes: Full list of nodes (head first, then workers)

    Returns:
        List of worker nodes (excluding head), empty if single node
    """
    if len(nodes) <= 1:
        return []
    return nodes[1:]


def load_env_file() -> dict[str, str]:
    """
    Load environment variables from .env file.

    Reads the .env file created by --discover for persistent cluster configuration.

    EXTENSIBILITY:
    - To support multiple .env files: Add a --env-file CLI argument
    - To add validation: Check for required keys after loading

    SUPPORTED KEYS (set by --discover):
        CLUSTER_NODES: Comma-separated list of node IPs
        LOCAL_IP: This machine's IP address
        ETH_IF: Ethernet interface name
        IB_IF: InfiniBand interface name (if available)

    Returns:
        Dictionary of key=value pairs from .env file
    """
    env = {}
    if ENV_FILE.exists():
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env[key.strip()] = value
    return env


def run_autodiscover() -> dict[str, str] | None:
    """
    Run autodiscover.sh interactively and return discovered configuration.

    Executes the autodiscover.sh script to detect cluster topology,
    including interactive per-node confirmation and .env saving.
    After autodiscover.sh completes, reads configuration from .env file.

    Returns:
        Dictionary with discovered configuration from .env, or None if discovery failed
    """
    if not AUTODISCOVER_SCRIPT.exists():
        print(f"Error: Autodiscover script not found: {AUTODISCOVER_SCRIPT}")
        return None

    print("Running autodiscover...")
    print()

    # Pass CONFIG_FILE so autodiscover.sh knows where to save the config.
    # Do NOT set CONFIG_FILE_SET=true — that would cause an error if the file
    # doesn't exist yet (it's the file we're about to create).
    env_vars = os.environ.copy()
    env_vars["CONFIG_FILE"] = str(ENV_FILE)
    env_vars["FORCE_DISCOVER"] = "true"
    env_vars.pop("CONFIG_FILE_SET", None)

    # Run autodiscover interactively so its prompts are shown to the user
    script = f"""
        source '{AUTODISCOVER_SCRIPT}'
        run_autodiscover
    """

    result = subprocess.run(["bash", "-c", script], env=env_vars)

    if result.returncode != 0:
        print("Error: Autodiscover failed")
        return None

    # Read configuration from the .env file that autodiscover.sh wrote
    env = load_env_file()
    if not env.get("CLUSTER_NODES"):
        print("Autodiscover completed but no CLUSTER_NODES found in .env")
        return None

    return env


def main():
    """
    Main entry point for the recipe runner.

    Orchestrates the full deployment pipeline:
    1. Parse CLI arguments and load recipe
    2. Resolve cluster nodes (CLI -> .env -> autodiscover)
    3. Build phase: Build container if missing, copy to workers
    4. Download phase: Download model if missing, copy to workers
    5. Run phase: Generate launch script and execute via launch-cluster.sh

    EXTENSIBILITY:
    - To add new CLI options: Add to the appropriate argument group
    - To add new phases: Insert between existing phases with similar pattern
    - To add pre/post hooks: Add hook execution before/after subprocess calls
    - To add logging: Replace print() with logging module calls
    - To add config file support: Load defaults from ~/.config/vllm-recipes.yaml

    EXIT CODES:
        0: Success
        1: Error (recipe not found, build failed, validation error, etc.)

    Returns:
        Exit code for sys.exit()
    """
    parser = argparse.ArgumentParser(
        description="Run a model using a YAML recipe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s glm-4.7-nvfp4
  %(prog)s glm-4.7-nvfp4 --port 9000 --solo

  # Full setup (build container + download model + run)
  %(prog)s glm-4.7-nvfp4 --setup

  # Cluster deployment (manual)
  %(prog)s glm-4.7-nvfp4 -n 192.168.1.1,192.168.1.2 --setup

  # Cluster deployment (auto-discover)
  %(prog)s --discover              # Detect nodes and save to .env
  %(prog)s glm-4.7-nvfp4 --setup   # Uses nodes from .env

  # Just build/download without running
  %(prog)s glm-4.7-nvfp4 --build-only
  %(prog)s glm-4.7-nvfp4 --download-only

  # Pass extra arguments to vLLM (after --)
  %(prog)s glm-4.7-nvfp4 --solo -- --load-format safetensors
  %(prog)s glm-4.7-nvfp4 --solo -- --served-model-name my-api

  # List available recipes
  %(prog)s --list

  # Show current .env configuration
  %(prog)s --show-env
        """,
    )

    parser.add_argument(
        "recipe",
        nargs="?",
        help="Path to recipe YAML file (or just the name without .yaml)",
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available recipes"
    )

    # Setup options
    setup_group = parser.add_argument_group("Setup options")
    setup_group.add_argument(
        "--setup",
        action="store_true",
        help="Full setup: build container (if missing) + download model (if missing) + run",
    )
    setup_group.add_argument(
        "--build-only",
        action="store_true",
        help="Only build/copy the container image, don't run",
    )
    setup_group.add_argument(
        "--download-only",
        action="store_true",
        help="Only download/copy the model, don't run",
    )
    setup_group.add_argument(
        "--force-build", action="store_true", help="Force rebuild even if image exists"
    )
    setup_group.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if model exists",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )

    # Override options
    override_group = parser.add_argument_group("Recipe overrides")
    override_group.add_argument("--port", type=int, help="Override port")
    override_group.add_argument("--host", help="Override host")
    override_group.add_argument(
        "--tensor-parallel",
        "--tp",
        type=int,
        dest="tensor_parallel",
        help="Override tensor parallelism",
    )
    override_group.add_argument(
        "--gpu-memory-utilization",
        "--gpu-mem",
        type=float,
        dest="gpu_memory_utilization",
        help="Override GPU memory utilization",
    )
    override_group.add_argument(
        "--max-model-len",
        type=int,
        dest="max_model_len",
        help="Override max model length",
    )

    # Launch options (passed to launch-cluster.sh)
    launch_group = parser.add_argument_group(
        "Launch options (passed to launch-cluster.sh)"
    )
    launch_group.add_argument(
        "--solo", action="store_true", help="Run in solo mode (single node, no Ray)"
    )
    launch_group.add_argument(
        "-n", "--nodes", help="Comma-separated list of node IPs (first is head node)"
    )
    launch_group.add_argument(
        "-d", "--daemon", action="store_true", help="Run in daemon mode"
    )
    launch_group.add_argument(
        "-t",
        "--container",
        dest="container_override",
        help="Override container image from recipe",
    )
    launch_group.add_argument(
        "--nccl-debug",
        choices=["VERSION", "WARN", "INFO", "TRACE"],
        help="NCCL debug level",
    )
    launch_group.add_argument(
        "-e",
        "--env",
        action="append",
        dest="env_vars",
        default=[],
        metavar="VAR=VALUE",
        help="Environment variable to pass to container (e.g. -e HF_TOKEN=xxx). Can be used multiple times.",
    )
    launch_group.add_argument(
        "--no-ray",
        action="store_true",
        dest="no_ray",
        help="No-Ray mode: run multi-node vLLM without Ray (uses PyTorch distributed backend)",
    )
    launch_group.add_argument(
        "--master-port",
        "--head-port",
        type=int,
        dest="master_port",
        help="Port for cluster coordination (Ray head port or PyTorch distributed master port, default: 29501)",
    )
    launch_group.add_argument(
        "--name",
        dest="container_name",
        help="Override container name (default: vllm_node)",
    )
    launch_group.add_argument(
        "--eth-if",
        dest="eth_if",
        help="Ethernet interface (overrides .env and auto-detection)",
    )
    launch_group.add_argument(
        "--ib-if",
        dest="ib_if",
        help="InfiniBand interface (overrides .env and auto-detection)",
    )
    launch_group.add_argument(
        "-j",
        dest="build_jobs",
        type=int,
        metavar="N",
        help="Number of parallel build jobs inside container",
    )
    launch_group.add_argument(
        "--no-cache-dirs",
        action="store_true",
        dest="no_cache_dirs",
        help="Do not mount ~/.cache/vllm, ~/.cache/flashinfer, ~/.triton",
    )
    launch_group.add_argument(
        "--non-privileged",
        action="store_true",
        dest="non_privileged",
        help="Run in non-privileged mode (removes --privileged and --ipc=host)",
    )
    launch_group.add_argument(
        "--mem-limit-gb",
        type=int,
        dest="mem_limit_gb",
        help="Memory limit in GB (only with --non-privileged)",
    )
    launch_group.add_argument(
        "--mem-swap-limit-gb",
        type=int,
        dest="mem_swap_limit_gb",
        help="Memory+swap limit in GB (only with --non-privileged)",
    )
    launch_group.add_argument(
        "--pids-limit",
        type=int,
        dest="pids_limit",
        help="Process limit (only with --non-privileged, default: 4096)",
    )
    launch_group.add_argument(
        "--shm-size-gb",
        type=int,
        dest="shm_size_gb",
        help="Shared memory size in GB (only with --non-privileged, default: 64)",
    )

    # Config file option
    parser.add_argument(
        "--config",
        dest="config_file",
        metavar="FILE",
        help="Path to .env configuration file (default: .env in script directory)",
    )

    # Cluster discovery options
    discover_group = parser.add_argument_group("Cluster discovery")
    discover_group.add_argument(
        "--discover",
        action="store_true",
        help="Auto-detect cluster nodes and save to .env file",
    )
    discover_group.add_argument(
        "--show-env", action="store_true", help="Show current .env configuration"
    )

    # Use parse_known_args to allow extra vLLM arguments after --
    args, extra_args = parser.parse_known_args()

    # Set .env file path (use default if not specified)
    global ENV_FILE
    if args.config_file:
        ENV_FILE = Path(args.config_file).resolve()
    else:
        ENV_FILE = SCRIPT_DIR / ".env"

    # Filter out the -- separator if present
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    # Handle --discover (can be run with or without a recipe)
    if args.discover:
        env = run_autodiscover()
        if env is None:
            return 1

        print("Discovered configuration:")
        for key, value in sorted(env.items()):
            print(f"  {key}={value}")
        print()

        if not args.recipe:
            return 0

    # Handle --show-env
    if args.show_env:
        env = load_env_file()
        if env:
            print(f"Current .env configuration ({ENV_FILE}):")
            for key, value in sorted(env.items()):
                print(f"  {key}={value}")
        else:
            print(f"No .env file found at {ENV_FILE}")
            print("Run with --discover to auto-detect cluster nodes.")

        if not args.recipe:
            return 0
        print()

    if args.list:
        list_recipes()
        return 0

    if not args.recipe:
        parser.print_help()
        return 1

    # Load recipe
    recipe_path = Path(args.recipe)
    recipe = load_recipe(recipe_path)

    print(f"Recipe: {recipe['name']}")
    if recipe.get("description"):
        print(f"  {recipe['description']}")
    print()

    # Determine container image
    container = args.container_override or recipe["container"]
    model = recipe.get("model")
    build_args = recipe.get("build_args", [])

    # Parse nodes - check command line first, then .env file, then autodiscover
    nodes = parse_nodes(args.nodes) if not args.solo else []
    nodes_from_env = False
    eth_if = None
    ib_if = None

    if not args.solo:
        # Try to load from .env file
        env = load_env_file()
        if not nodes:
            if env.get("CLUSTER_NODES"):
                nodes = parse_nodes(env["CLUSTER_NODES"])
                nodes_from_env = True
                if nodes:
                    print(f"Using cluster nodes from .env: {', '.join(nodes)}")
                    print()
            else:
                # No nodes specified and no .env - run autodiscover
                print("No cluster nodes configured. Running autodiscover...")
                print()

                discovered_env = run_autodiscover()
                if discovered_env and discovered_env.get("CLUSTER_NODES"):
                    env = discovered_env  # use freshly loaded env from autodiscover
                    nodes = parse_nodes(discovered_env["CLUSTER_NODES"])
                    nodes_from_env = True

        # Resolve network interfaces: CLI > .env > auto-detect by launch-cluster.sh
        eth_if = args.eth_if or None
        ib_if = args.ib_if or None
        if not eth_if or not ib_if:
            if not eth_if and env.get("ETH_IF"):
                eth_if = env["ETH_IF"]
            if not ib_if and env.get("IB_IF"):
                ib_if = env["IB_IF"]

    worker_nodes = get_worker_nodes(nodes) if nodes else []
    is_cluster = len(nodes) > 1

    # Check if recipe requires cluster mode
    cluster_only = recipe.get("cluster_only", False)
    solo_only = recipe.get("solo_only", False)
    is_solo = args.solo or not is_cluster

    if getattr(args, "no_ray", False) and is_solo:
        print(
            "Error: --no-ray is incompatible with --solo. Solo mode already runs without Ray."
        )
        return 1

    if cluster_only and is_solo:
        print(f"Error: Recipe '{recipe['name']}' requires cluster mode.")
        print(f"This model is too large to run on a single node.")
        print()
        print("Options:")
        print(
            f"  1. Specify nodes directly:  {sys.argv[0]} {args.recipe} -n node1,node2"
        )
        print(f"  2. Auto-discover and save:  {sys.argv[0]} --discover")
        print(f"     Then run:                {sys.argv[0]} {args.recipe}")
        return 1
    if solo_only and not is_solo:
        print(f"Error: Recipe '{recipe['name']}' requires solo mode.")
        print("This recipe is intended to run on a single node only.")
        print()
        print("Options:")
        print(f"  1. Run solo:                {sys.argv[0]} {args.recipe} --solo")
        print(f"  2. Remove nodes from .env:  {sys.argv[0]} --show-env")
        return 1

    # Determine copy targets for build/model distribution.
    # Prefer COPY_HOSTS from .env (may differ from CLUSTER_NODES in mesh mode),
    # fall back to worker_nodes derived from CLUSTER_NODES.
    if is_cluster:
        copy_hosts_str = env.get("COPY_HOSTS")
        if copy_hosts_str:
            copy_targets = [h.strip() for h in copy_hosts_str.split(",") if h.strip()]
        else:
            copy_targets = worker_nodes
    else:
        copy_targets = None

    if args.dry_run:
        print("=== Dry Run ===")
        print(f"Container: {container}")
        if build_args:
            print(f"Build args: {' '.join(build_args)}")
        if model:
            print(f"Model: {model}")
        if cluster_only:
            print("Cluster only: Yes (model too large for single node)")
        if solo_only:
            print("Solo only: Yes (single node only)")
        if nodes:
            source = "(from .env)" if nodes_from_env else ""
            print(f"Nodes: {', '.join(nodes)} {source}".strip())
            print(f"  Head: {nodes[0]}")
            if worker_nodes:
                print(f"  Workers: {', '.join(worker_nodes)}")
        print(f"Solo mode: {is_solo}")
        if eth_if:
            print(
                f"Ethernet interface: {eth_if}{' (from .env)' if not args.eth_if else ''}"
            )
        if ib_if:
            print(
                f"InfiniBand interface: {ib_if}{' (from .env)' if not args.ib_if else ''}"
            )
        if args.container_name:
            print(f"Container name: {args.container_name}")
        if args.non_privileged:
            print("Non-privileged mode: Yes")
        print()

    # --- Build Phase ---
    if args.build_only or args.setup or args.force_build:
        if args.dry_run:
            image_exists = check_image_exists(container)
            if args.force_build or not image_exists:
                print(f"Would build container: {container}")
                if copy_targets:
                    print(f"  Would copy to: {', '.join(copy_targets)}")
            else:
                print(f"Container '{container}' already exists locally.")
                if copy_targets:
                    print(f"  Would check/copy to workers: {', '.join(copy_targets)}")
            print()
        else:
            image_exists = check_image_exists(container)

            if args.force_build or not image_exists:
                print("=== Building Container ===")
                if not build_image(container, copy_targets, build_args):
                    print("Error: Failed to build container")
                    return 1
                print()
            else:
                print(f"Container '{container}' already exists locally.")
                # Check worker nodes in cluster mode
                if copy_targets:
                    missing_on = []
                    for worker in copy_targets:
                        if not check_image_exists(container, worker):
                            missing_on.append(worker)
                    if missing_on:
                        print(f"Container missing on workers: {', '.join(missing_on)}")
                        print("Building and copying...")
                        if not build_image(container, missing_on, build_args):
                            print("Error: Failed to build/copy container")
                            return 1
                print()

        if args.build_only:
            print("Build complete." if not args.dry_run else "")
            return 0

    # --- Download Phase ---
    if model and (args.download_only or args.setup or args.force_download):
        if args.dry_run:
            model_exists = check_model_exists(model)
            if args.force_download or not model_exists:
                print(f"Would download model: {model}")
                if copy_targets:
                    print(f"  Would copy to: {', '.join(copy_targets)}")
            else:
                print(f"Model '{model}' already exists in cache.")
            print()
        else:
            model_exists = check_model_exists(model)

            if args.force_download or not model_exists:
                print("=== Downloading Model ===")
                if not download_model(model, copy_targets):
                    print("Error: Failed to download model")
                    return 1
                print()
            else:
                print(f"Model '{model}' already exists in cache.")
                print()

        if args.download_only:
            print("Download complete." if not args.dry_run else "")
            return 0

    # --- Run Phase ---
    if args.build_only or args.download_only:
        return 0

    # Check if image exists (if not using --setup)
    if not args.dry_run and not args.setup and not check_image_exists(container):
        print(f"Container image '{container}' not found locally.")
        print()
        print("Options:")
        print(f"  1. Use --setup to build and run")
        print(f"  2. Build manually: ./build-and-copy.sh -t {container}")
        print()
        response = input("Build now? [y/N] ").strip().lower()
        if response == "y":
            if not build_image(container, copy_targets, build_args):
                print("Error: Failed to build image")
                return 1
        else:
            print("Aborting.")
            return 1

    # Build overrides from CLI args
    overrides = {}
    for key in [
        "port",
        "host",
        "tensor_parallel",
        "gpu_memory_utilization",
        "max_model_len",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value

    # In solo mode, default tensor_parallel to 1 (unless user explicitly set --tp)
    if is_solo and "tensor_parallel" not in overrides:
        overrides["tensor_parallel"] = 1

    # Check for duplicate arguments (warn if extra_args duplicate CLI overrides)
    if extra_args:
        # Map vLLM flags to our override keys
        flag_to_override = {
            "--port": "port",
            "--host": "host",
            "--tensor-parallel-size": "tensor_parallel",
            "-tp": "tensor_parallel",
            "--gpu-memory-utilization": "gpu_memory_utilization",
            "--max-model-len": "max_model_len",
        }
        for i, arg in enumerate(extra_args):
            # Check both exact flag and =value syntax
            flag = arg.split("=")[0] if "=" in arg else arg
            if flag in flag_to_override:
                override_key = flag_to_override[flag]
                if override_key in overrides:
                    print(
                        f"Warning: '{arg}' in extra args duplicates --{override_key.replace('_', '-')} override"
                    )
                    print(
                        f"         vLLM uses last value; extra args appear after template substitution"
                    )

    # Generate launch script
    script_content = generate_launch_script(
        recipe,
        overrides,
        is_solo=is_solo,
        extra_args=extra_args,
        no_ray=getattr(args, "no_ray", False),
    )

    if args.dry_run:
        print("=== Generated Launch Script ===")
        print(script_content)
        print("=== What would be executed ===")
        print()
        print("1. The above script is saved to a temporary file")
        print()
        print("2. launch-cluster.sh is called with:")
        cmd_parts = ["   ./launch-cluster.sh", "-t", container]
        for mod in recipe.get("mods", []):
            cmd_parts.extend(["--apply-mod", mod])
        if args.solo:
            cmd_parts.append("--solo")
        elif not is_cluster:
            cmd_parts.append("--solo")
        if args.daemon:
            cmd_parts.append("-d")
        if getattr(args, "no_ray", False):
            cmd_parts.append("--no-ray")
        if nodes:
            cmd_parts.extend(["-n", ",".join(nodes)])
        if args.nccl_debug:
            cmd_parts.extend(["--nccl-debug", args.nccl_debug])
        for env_var in args.env_vars:
            cmd_parts.extend(["-e", env_var])
        if args.master_port:
            cmd_parts.extend(["--master-port", str(args.master_port)])
        if args.container_name:
            cmd_parts.extend(["--name", args.container_name])
        if eth_if:
            cmd_parts.extend(["--eth-if", eth_if])
        if ib_if:
            cmd_parts.extend(["--ib-if", ib_if])
        if args.build_jobs:
            cmd_parts.extend(["-j", str(args.build_jobs)])
        if args.no_cache_dirs:
            cmd_parts.append("--no-cache-dirs")
        if args.non_privileged:
            cmd_parts.append("--non-privileged")
        if args.mem_limit_gb:
            cmd_parts.extend(["--mem-limit-gb", str(args.mem_limit_gb)])
        if args.mem_swap_limit_gb:
            cmd_parts.extend(["--mem-swap-limit-gb", str(args.mem_swap_limit_gb)])
        if args.pids_limit:
            cmd_parts.extend(["--pids-limit", str(args.pids_limit)])
        if args.shm_size_gb:
            cmd_parts.extend(["--shm-size-gb", str(args.shm_size_gb)])
        if args.config_file:
            cmd_parts.extend(["--config", args.config_file])
        cmd_parts.extend(["\\", "\n      --launch-script", "/tmp/tmpXXXXXX.sh"])
        print(" ".join(cmd_parts))
        print()
        print("3. The launch script runs inside the container")
        return 0

    # Write temporary launch script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        os.chmod(temp_script, 0o755)

        # Build launch-cluster.sh command
        cmd = [str(LAUNCH_SCRIPT), "-t", container]

        # Add mods
        for mod in recipe.get("mods", []):
            mod_path = SCRIPT_DIR / mod
            if not mod_path.exists():
                print(f"Warning: Mod path not found: {mod_path}")
            cmd.extend(["--apply-mod", str(mod_path)])

        # Add launch options
        if args.solo:
            cmd.append("--solo")
        elif not is_cluster:
            # Auto-enable solo mode if no cluster nodes specified
            cmd.append("--solo")

        if args.daemon:
            cmd.append("-d")

        if getattr(args, "no_ray", False):
            cmd.append("--no-ray")

        # Pass nodes to launch-cluster.sh (from command line, .env, or autodiscover)
        if nodes:
            cmd.extend(["-n", ",".join(nodes)])

        if args.nccl_debug:
            cmd.extend(["--nccl-debug", args.nccl_debug])

        for env_var in args.env_vars:
            cmd.extend(["-e", env_var])

        if args.master_port:
            cmd.extend(["--master-port", str(args.master_port)])
        if args.container_name:
            cmd.extend(["--name", args.container_name])
        if eth_if:
            cmd.extend(["--eth-if", eth_if])
        if ib_if:
            cmd.extend(["--ib-if", ib_if])
        if args.build_jobs:
            cmd.extend(["-j", str(args.build_jobs)])
        if args.no_cache_dirs:
            cmd.append("--no-cache-dirs")
        if args.non_privileged:
            cmd.append("--non-privileged")
        if args.mem_limit_gb:
            cmd.extend(["--mem-limit-gb", str(args.mem_limit_gb)])
        if args.mem_swap_limit_gb:
            cmd.extend(["--mem-swap-limit-gb", str(args.mem_swap_limit_gb)])
        if args.pids_limit:
            cmd.extend(["--pids-limit", str(args.pids_limit)])
        if args.shm_size_gb:
            cmd.extend(["--shm-size-gb", str(args.shm_size_gb)])

        if args.config_file:
            cmd.extend(["--config", args.config_file])

        # Add launch script
        cmd.extend(["--launch-script", temp_script])

        print(f"=== Launching ===")
        print(f"Container: {container}")
        if recipe.get("mods"):
            print(f"Mods: {', '.join(recipe['mods'])}")
        if is_cluster:
            print(f"Cluster: {len(nodes)} nodes")
        else:
            print("Mode: Solo")
        print()

        # Execute
        result = subprocess.run(cmd)
        return result.returncode

    finally:
        # Cleanup temp script
        try:
            os.unlink(temp_script)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
