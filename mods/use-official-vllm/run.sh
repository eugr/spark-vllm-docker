#!/bin/bash
set -euo pipefail

# Compatibility setup for official vLLM docker containers.
#
# - Installs git, which official vLLM containers may omit and other mods need.
# - Redirects pip-installed nvidia-nccl-cu13's libnccl.so.2 to the system
#   libnccl2 soname when both are present. On DGX Spark, official vLLM images
#   can otherwise load the Python-package NCCL first and hang multi-node runs.

PREFIX="[use-official-vllm]"

run_with_privilege() {
  if "$@"; then
    return
  fi

  local status=$?
  if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi

  echo "$PREFIX Command failed with status $status: $*" >&2
  echo "$PREFIX This operation may require root or sudo privileges." >&2
  exit "$status"
}

install_git_if_needed() {
  if command -v git >/dev/null 2>&1; then
    echo "$PREFIX git is already installed: $(git --version)"
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "$PREFIX git is missing, and apt-get was not found. This mod expects an Ubuntu/Debian-based container." >&2
    exit 1
  fi

  echo "$PREFIX git not found; installing git with apt-get."
  run_with_privilege apt-get update
  run_with_privilege env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
  echo "$PREFIX Installed $(git --version)"

  # Install pytest in case some mods/patches/PR require it.
  echo "$PREFIX Installing additional Python dependencies..."
  pip install pytest
}

find_system_nccl() {
  if [ -n "${SYSTEM_NCCL_PATH:-}" ]; then
    printf '%s\n' "$SYSTEM_NCCL_PATH"
    return
  fi

  local candidates=()
  local multiarch=""

  if command -v gcc >/dev/null 2>&1; then
    multiarch="$(gcc -print-multiarch 2>/dev/null || true)"
  fi

  if [ -n "$multiarch" ]; then
    candidates+=("/usr/lib/$multiarch/libnccl.so.2")
  fi

  candidates+=(
    "/usr/lib/aarch64-linux-gnu/libnccl.so.2"
    "/usr/lib/x86_64-linux-gnu/libnccl.so.2"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -e "$candidate" ] || [ -L "$candidate" ]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | awk '/libnccl\.so\.2/ && $NF ~ /^\/usr\/lib\// { print $NF; exit }'
  fi
}

find_python_nccl_libs() {
  if [ -n "${PYTHON_NCCL_LIB_PATH:-}" ]; then
    printf '%s\n' "$PYTHON_NCCL_LIB_PATH"
    return
  fi

  shopt -s nullglob
  local candidates=(
    /usr/local/lib/python*/dist-packages/nvidia/nccl/lib/libnccl.so.2
    /usr/local/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2
    /usr/lib/python*/dist-packages/nvidia/nccl/lib/libnccl.so.2
    /usr/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2
    /opt/venv/lib/python*/site-packages/nvidia/nccl/lib/libnccl.so.2
  )
  shopt -u nullglob

  local candidate
  for candidate in "${candidates[@]}"; do
    printf '%s\n' "$candidate"
  done
}

prefer_system_nccl_if_present() {
  local system_nccl
  system_nccl="$(find_system_nccl)"

  if [ -z "$system_nccl" ]; then
    echo "$PREFIX No system libnccl.so.2 found under /usr/lib; skipping NCCL symlink fix."
    echo "$PREFIX Set SYSTEM_NCCL_PATH=/path/to/libnccl.so.2 to override."
    return
  fi

  if [ ! -e "$system_nccl" ] && [ ! -L "$system_nccl" ]; then
    echo "$PREFIX System NCCL path does not exist: $system_nccl" >&2
    exit 1
  fi

  local system_nccl_real
  system_nccl_real="$(readlink -f "$system_nccl")"
  echo "$PREFIX Using system NCCL: $system_nccl -> $system_nccl_real"

  local python_nccl_libs=()
  local python_nccl_candidate
  while IFS= read -r python_nccl_candidate; do
    if [ -n "$python_nccl_candidate" ]; then
      python_nccl_libs+=("$python_nccl_candidate")
    fi
  done < <(find_python_nccl_libs)

  if [ "${#python_nccl_libs[@]}" -eq 0 ]; then
    echo "$PREFIX No pip-installed nvidia/nccl/lib/libnccl.so.2 found; skipping NCCL symlink fix."
    return
  fi

  local patched=0
  local python_nccl
  for python_nccl in "${python_nccl_libs[@]}"; do
    if [ ! -e "$python_nccl" ] && [ ! -L "$python_nccl" ]; then
      echo "$PREFIX Skipping missing Python NCCL path: $python_nccl"
      continue
    fi

    local current_real
    current_real="$(readlink -f "$python_nccl" || true)"
    if [ "$current_real" = "$system_nccl_real" ]; then
      echo "$PREFIX Already using system NCCL: $python_nccl"
      continue
    fi

    local backup="${python_nccl}.spark-vllm-backup"
    if [ ! -e "$backup" ] && [ ! -L "$backup" ]; then
      echo "$PREFIX Backing up $python_nccl to $backup"
      run_with_privilege mv "$python_nccl" "$backup"
    else
      echo "$PREFIX Backup already exists at $backup; replacing current link/file."
      run_with_privilege rm -f "$python_nccl"
    fi

    echo "$PREFIX Linking $python_nccl -> $system_nccl"
    run_with_privilege ln -s "$system_nccl" "$python_nccl"
    patched=$((patched + 1))
  done

  if [ "$patched" -eq 0 ]; then
    echo "$PREFIX NCCL symlink fix did not need any changes."
  else
    echo "$PREFIX Patched $patched Python NCCL path(s)."
  fi
}

install_git_if_needed
prefer_system_nccl_if_present
