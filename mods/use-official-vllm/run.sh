#!/bin/bash
set -euo pipefail

# This mod enables patches that require git with official vLLM docker containers.
# Official vLLM containers are expected to be Ubuntu/Debian based.

run_as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "git is missing, and installing it requires root or sudo privileges." >&2
    exit 1
  fi
}

if command -v git >/dev/null 2>&1; then
  echo "git is already installed: $(git --version)"
  exit 0
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "git is missing, and apt-get was not found. This mod expects an Ubuntu/Debian-based container." >&2
  exit 1
fi

echo "git not found; installing git with apt-get."
run_as_root apt-get update
run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
echo "Installed $(git --version)"
