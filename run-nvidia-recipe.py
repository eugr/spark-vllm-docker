#!/usr/bin/env python3
"""Translate a NVIDIA vLLM recipe into the local recipe format."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

import yaml


ROOT = Path(__file__).resolve().parent
ORCHESTRATION_FLAGS = {
    "--nnodes", "--node-rank", "--master-addr", "--master-port",
    "--distributed-executor-backend", "--headless",
}
VALUE_FLAGS = ORCHESTRATION_FLAGS - {"--headless"}
OVERRIDE_FLAGS = {
    "--host": "host", "--port": "port",
    "--tensor-parallel-size": "tensor_parallel", "-tp": "tensor_parallel",
    "--gpu-memory-utilization": "gpu_memory_utilization",
    "--max-model-len": "max_model_len",
}


def read_source(source: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme:
        if parsed.scheme != "https":
            raise ValueError("remote recipes must use HTTPS")
        if parsed.netloc == "github.com":
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 5 or parts[2] != "blob":
                raise ValueError("expected a GitHub blob URL pointing to a YAML file")
            source = "https://raw.githubusercontent.com/" + "/".join(
                [parts[0], parts[1], parts[3], *parts[4:]]
            )
        with urlopen(source, timeout=20) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def image_for(image: str, remap: bool) -> str:
    if not remap:
        return image
    if re.fullmatch(r"eugr/spark-vllm-b12x:[^\s]+", image):
        return "vllm-node-b12x"
    if re.fullmatch(r"eugr/spark-vllm:[^\s]+", image):
        return "vllm-node"
    return "vllm-node"


def parse_serve(command: str) -> tuple[list[str], dict[str, str]]:
    match = re.search(r"^\s*vllm[ \t]+serve[ \t]+", command, re.MULTILINE)
    if not match:
        raise ValueError("serve.command must contain a vllm serve command")
    prefix = command[:match.start()].strip()
    env = {}
    for line in prefix.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = shlex.split(line)
        if len(words) != 2 or words[0] != "export" or "=" not in words[1]:
            raise ValueError(f"unsupported command before vllm serve: {line}")
        key, value = words[1].split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"invalid environment variable: {key}")
        env[key] = value
    tokens = shlex.split(command[match.start():].replace("\\\n", " "))
    if tokens[:2] != ["vllm", "serve"]:
        raise ValueError("serve.command must start with vllm serve after exports")
    return tokens, env


def convert(source: dict, container_override: str | None, remap: bool) -> dict:
    if not isinstance(source, dict) or source.get("backend") != "vllm":
        raise ValueError("expected a NVIDIA vLLM recipe")
    runtime = source.get("runtime") or {}
    serve = source.get("serve") or {}
    if runtime.get("kind") != "container" or not runtime.get("image"):
        raise ValueError("recipe needs runtime.kind=container and runtime.image")
    if not isinstance(serve.get("command"), str):
        raise ValueError("recipe needs serve.command")
    tokens, env = parse_serve(serve["command"])
    if len(tokens) < 3:
        raise ValueError("vllm serve command needs a model")
    defaults = {}
    result = tokens[:3]
    index = 3
    while index < len(tokens):
        token = tokens[index]
        flag, separator, inline_value = token.partition("=")
        if flag in ORCHESTRATION_FLAGS:
            if flag in VALUE_FLAGS and not separator:
                if index + 1 >= len(tokens):
                    raise ValueError(f"missing value for {flag}")
                index += 1
            index += 1
            continue
        if flag in OVERRIDE_FLAGS:
            if separator:
                value = inline_value
            else:
                if index + 1 >= len(tokens):
                    raise ValueError(f"missing value for {flag}")
                index += 1
                value = tokens[index]
            key = OVERRIDE_FLAGS[flag]
            defaults[key] = value
            result.extend([flag, "{" + key + "}"])
        else:
            result.append(token)
        index += 1
    # run-recipe.py formats its command with str.format; protect literal JSON braces.
    command = " ".join(
        token if token.startswith("{") and token.endswith("}") and token[1:-1] in defaults
        else shlex.quote(token).replace("{", "{{").replace("}", "}}")
        for token in result
    )
    checkpoint = source.get("checkpoint") or {}
    model = checkpoint.get("target_label")
    if not model and isinstance(checkpoint.get("target"), str):
        model = checkpoint["target"].removeprefix("https://huggingface.co/")
    if not model:
        model = tokens[2]
    topology = source.get("topology") or {}
    node_count = topology.get("node_count")
    if node_count is not None and (not isinstance(node_count, int) or node_count < 1):
        raise ValueError("topology.node_count must be a positive integer")
    image = container_override or image_for(runtime["image"], remap)
    recipe = {
        "recipe_version": "1",
        "name": str(source.get("title") or model),
        "container": image,
        "model": model,
        "command": command,
    }
    if image == runtime["image"]:
        recipe["pull_image"] = True
    if defaults:
        recipe["defaults"] = defaults
    if env:
        recipe["env"] = env
    if node_count and node_count > 1:
        recipe["cluster_only"] = True
    if node_count == 1:
        recipe["solo_only"] = True
    return recipe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("recipe", help="local YAML path or HTTPS URL (GitHub blob or raw URL)")
    parser.add_argument("-t", "--container", help="override the recipe container image")
    parser.add_argument("--remap-container", action="store_true", help="map NVIDIA image to a local vllm-node tag")
    parser.add_argument(
        "--export", nargs="?", const="-", metavar="FILE",
        help="write an internal recipe to FILE, or stdout if FILE is omitted; do not launch",
    )
    args, forwarded = parser.parse_known_args(argv)
    if args.export is not None and forwarded:
        parser.error("runner options cannot be used with --export")
    try:
        source = yaml.safe_load(read_source(args.recipe))
        recipe = convert(source, args.container, args.remap_container)
        if args.export is not None:
            if args.export == "-":
                yaml.safe_dump(recipe, sys.stdout, sort_keys=False, allow_unicode=True)
                return 0
            target = Path(args.export)
            if target.exists():
                raise ValueError(f"export destination already exists: {target}")
            with target.open("x", encoding="utf-8") as handle:
                yaml.safe_dump(recipe, handle, sort_keys=False, allow_unicode=True)
            print(f"Exported recipe: {target}")
            return 0
        topology = source.get("topology") or {}
        if topology.get("node_count") == 1 and not any(
            flag in forwarded for flag in ("--solo", "-n", "--nodes")
        ):
            forwarded = ["--solo", *forwarded]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(recipe, handle, sort_keys=False, allow_unicode=True)
            temp_path = handle.name
        try:
            return subprocess.call([str(ROOT / "run-recipe.sh"), temp_path, *forwarded])
        finally:
            os.unlink(temp_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
