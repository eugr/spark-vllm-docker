#!/usr/bin/env python3
"""
run_benchmark.py - Recipe benchmark runner.

This script reads benchmark configuration from a recipe file and runs
the configured benchmark framework.
"""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent.resolve()
RECIPES_DIR = SCRIPT_DIR / "recipes"
SUPPORTED_BENCHMARK_FRAMEWORKS = {"llama-benchy"}


def validate_recipe_benchmark_config(recipe: dict) -> None:
    """Validate and normalize benchmark config in a recipe."""
    benchmark = recipe.setdefault("benchmark", {})
    benchmark.setdefault("framework", "llama-benchy")
    benchmark.setdefault("args", {})

    if benchmark["framework"] not in SUPPORTED_BENCHMARK_FRAMEWORKS:
        raise ValueError(
            f"Unsupported benchmark framework in recipe: {benchmark['framework']}\n"
            f"Supported frameworks: {', '.join(sorted(SUPPORTED_BENCHMARK_FRAMEWORKS))}"
        )

def resolve_recipe_path(recipe_arg: str) -> Path:
    recipe_path = Path(recipe_arg)
    if recipe_path.exists():
        return recipe_path

    candidates = [
        RECIPES_DIR / recipe_path.name,
        RECIPES_DIR / f"{recipe_path.name}.yaml",
        RECIPES_DIR / f"{recipe_path.name}.yml",
        RECIPES_DIR / f"{recipe_path.stem}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Recipe not found: {recipe_arg}")


def apply_llama_benchy_defaults(benchmark_args: dict) -> None:
    benchmark_args.setdefault("pp", [2048])
    benchmark_args.setdefault("depth", [0])
    benchmark_args.setdefault("save_result", "test.md")
    benchmark_args.setdefault("enable_prefix_caching", True)


def append_cli_args_from_dict(cmd: list[str], args: dict, exclude_keys: set[str] | None = None) -> None:
    """Append CLI args from dict using --kebab-case conversion.

    Rules:
    - bool true  -> add flag only (e.g., --enable-feature)
    - bool false -> omit
    - list/tuple -> add flag once, then one value per element
    - other      -> add flag + single value
    """
    excludes = exclude_keys or set()

    for key, value in args.items():
        if key in excludes:
            continue

        flag = f"--{key.replace('_', '-')}"

        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue

        if isinstance(value, (list, tuple)):
            if not value:
                continue
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
            continue

        cmd.extend([flag, str(value)])


def get_recipe_base_url(recipe: dict) -> str:
    """Return configured OpenAI-compatible base URL for the recipe."""
    defaults = recipe.get("defaults", {})
    port = int(defaults.get("port", 8000))
    return f"http://localhost:{port}/v1"


def build_llama_benchy_command(recipe: dict, model: str = "__from_v1_models__") -> list[str]:
    benchmark = recipe.get("benchmark", {})
    benchmark_args = benchmark.setdefault("args", {})
    apply_llama_benchy_defaults(benchmark_args)

    base_url = get_recipe_base_url(recipe)

    cmd = [
        "llama-benchy",
        "--base-url",
        base_url,
        "--model",
        model,
    ]

    append_cli_args_from_dict(cmd, benchmark_args, exclude_keys={"base_url", "model"})
    return cmd


def wait_for_models_endpoint_and_get_first_model(
    base_url: str, timeout_seconds: int = 60*10, interval_seconds: int = 2
) -> str | None:
    """Wait for GET /v1/models, print payload, and return first model id when available."""
    models_url = f"{base_url.rstrip('/')}/models"
    deadline = time.time() + timeout_seconds
    last_error = "unknown error"

    print(f"Waiting for endpoint readiness: {models_url} (timeout: {timeout_seconds}s)")

    while time.time() < deadline:
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")

            print(f"\n=== vLLM Models Endpoint ===\nGET {models_url}")
            try:
                payload = json.loads(body)
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            except Exception:
                print(body)
                last_error = "non-JSON /v1/models response"
                time.sleep(interval_seconds)
                continue

            data = payload.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    for key in ("id", "model", "name"):
                        val = first.get(key)
                        if isinstance(val, str) and val.strip():
                            print(f"Resolved first model from /v1/models: {val.strip()}")
                            return val.strip()

            last_error = "No model entry found in /v1/models response"
            print("Endpoint reachable but model list is empty/invalid, retrying...")
        except Exception as e:
            last_error = str(e)

        time.sleep(interval_seconds)

    print(f"Timed out waiting for endpoint/model: {models_url}")
    print(f"Last probe error: {last_error}")
    return None


def run_recipe_benchmark(recipe: dict, dry_run: bool = False, benchmark_requested: bool = False) -> int:
    """Run benchmark for a loaded recipe, including readiness checks."""
    validate_recipe_benchmark_config(recipe)
    benchmark = recipe["benchmark"]
    if not benchmark_requested:
        print("Benchmark execution not requested (pass --benchmark to run).")
        return 0

    if benchmark["framework"] == "llama-benchy" and shutil.which("llama-benchy") is None:
        print("Error: 'llama-benchy' not found in PATH.")
        print("Install with: uv pip install -U llama-benchy")
        return 1

    if dry_run:
        cmd = build_llama_benchy_command(recipe)
        print("\n=== Running Benchmark ===")
        print(shlex.join(cmd))
        return 0

    base_url = get_recipe_base_url(recipe)

    discovered_model = wait_for_models_endpoint_and_get_first_model(base_url)
    if discovered_model:
        cmd = build_llama_benchy_command(recipe, model=discovered_model)
        print("\n=== Running Benchmark ===")
        print(f"Using model from /v1/models: {discovered_model}")
        print("Benchmark command (resolved):")
        print(shlex.join(cmd))
    else:
        print("Error: Could not resolve model from /v1/models.")
        return 1

    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        print("Error: llama-benchy not found in PATH. You can install with 'uv pip install -U llama-benchy'")
        return 1
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run model benchmark frameworks")
    parser.add_argument("recipe", help="Recipe YAML path or recipe name")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark",
    )
    parser.add_argument("--save-result", help="Override benchmark output file path")
    parser.add_argument("--dry-run", action="store_true", help="Print benchmark command without executing")
    args = parser.parse_args()

    recipe_path = resolve_recipe_path(args.recipe)

    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)

    validate_recipe_benchmark_config(recipe)

    benchmark_requested = args.benchmark
    if not benchmark_requested:
        print("Benchmark execution not requested (pass --benchmark to run).")
        return 0

    if args.save_result:
        recipe.setdefault("benchmark", {}).setdefault("args", {})["save_result"] = args.save_result

    return run_recipe_benchmark(recipe, dry_run=args.dry_run, benchmark_requested=benchmark_requested)


if __name__ == "__main__":
    sys.exit(main())
