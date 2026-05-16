#!/usr/bin/env python3
"""Resolve a HuggingFace model ID to a local path using model-mapping.json.

Usage: resolve-model.py <model_id>

If model_id is found in model-mapping.json, prints shell-eval-friendly output:
    LOCAL_PATH=/expanded/path/to/model
    CONTAINER_PATH=/workspace/local-models/model-basename
    SERVED_MODEL_NAME=friendly-name
    DOCKER_MOUNT=-v /expanded/path:/workspace/local-models/model-basename

Exit code 0 if mapping found, 1 if not found or no mapping file.
"""
import json
import os
import shlex
import sys
from pathlib import Path

MAPPING_FILE = Path(__file__).parent / "model-mapping.json"
CONTAINER_MODEL_DIR = "/workspace/local-models"


def resolve(model_id: str) -> dict | None:
    if not MAPPING_FILE.exists():
        return None

    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    entry = mapping.get(model_id)
    if not entry:
        return None

    local_path = os.path.expanduser(entry["local_path"]).rstrip("/")
    basename = os.path.basename(local_path)
    container_path = f"{CONTAINER_MODEL_DIR}/{basename}"
    served_model_name = entry.get("served_model_name") or basename

    return {
        "local_path": local_path,
        "container_path": container_path,
        "served_model_name": served_model_name,
        "docker_mount": f"-v {local_path}:{container_path}",
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_id>", file=sys.stderr)
        sys.exit(1)

    result = resolve(sys.argv[1])
    if not result:
        sys.exit(1)

    q = shlex.quote
    print(f"LOCAL_PATH={q(result['local_path'])}")
    print(f"CONTAINER_PATH={q(result['container_path'])}")
    print(f"SERVED_MODEL_NAME={q(result['served_model_name'])}")
    print(f"DOCKER_MOUNT={q(result['docker_mount'])}")
