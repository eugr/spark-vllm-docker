#!/usr/bin/env python3
"""Prepare a cached DeepGEMM checkout compatible with the selected vLLM source."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess


# Last known good before the SM121 DeepSeek-V4 MXFP4 grouped scale-factor
# regression at nv_dev f8e8fb5 (DeepGEMM PR #384). Only for the old binding.
LEGACY_REPO = "https://github.com/deepseek-ai/DeepGEMM.git"
LEGACY_REF = "a6b593d2826719dcf4892609af7b84ee23aaf32a"


def cmake_literal(source: str, name: str) -> str:
    values = re.findall(
        rf'\bset\s*\(\s*{re.escape(name)}\s+"([^"\n]+)"\s*\)', source
    )
    if len(values) != 1 or any(char in values[0] for char in "$;\\"):
        raise ValueError(f"Cannot resolve {name} from vLLM's DeepGEMM CMake file")
    return values[0]


def select_source(vllm: Path, repo: str = "", ref: str = "") -> tuple[str, str, bool]:
    source = (vllm / "cmake/external_projects/deepgemm.cmake").read_text()
    source = re.sub(r"(?m)#.*$", "", source)
    stable_abi = "deep_gemm/_C.py" in source
    if stable_abi:
        repo = repo or cmake_literal(source, "_DEEPGEMM_UPSTREAM_REPO")
        ref = ref or cmake_literal(source, "_DEEPGEMM_UPSTREAM_TAG")
    else:
        repo = repo or LEGACY_REPO
        ref = ref or LEGACY_REF
    return repo, ref, stable_abi


def validate_source(checkout: Path, stable_abi: bool) -> None:
    required = ["csrc/python_api.cpp", "deep_gemm/__init__.py"]
    if stable_abi:
        required += ["deep_gemm/_C.py", "setup.py"]
    missing = [name for name in required if not (checkout / name).is_file()]
    if missing:
        layout = "stable-ABI" if stable_abi else "legacy"
        raise ValueError(
            f"DeepGEMM checkout is incompatible with vLLM's {layout} build: "
            f"missing {', '.join(missing)}. Check DEEPGEMM_REPO/DEEPGEMM_REF overrides."
        )


def prepare(vllm: Path, cache: Path, destination: Path, repo: str, ref: str) -> None:
    repo, ref, stable_abi = select_source(vllm, repo, ref)
    layout = "stable-ABI" if stable_abi else "legacy SM121 workaround"
    print(f"DeepGEMM source ({layout}): {repo} @ {ref}", flush=True)
    # The regular lane and older forks can require different repositories.
    # Never reuse the former, unkeyed /repo-cache/deepgemm checkout.
    key = hashlib.sha256(repo.encode()).hexdigest()[:16]
    checkout = cache / f"deepgemm-{key}"
    cache.mkdir(parents=True, exist_ok=True)
    if not checkout.exists():
        subprocess.run(["git", "clone", "--no-checkout", repo, str(checkout)], check=True)
    else:
        subprocess.run(["git", "fetch", "origin", "--tags", "--force"], cwd=checkout, check=True)

    # Prefer the freshly fetched remote branch over a stale local branch.
    remote_ref = f"refs/remotes/origin/{ref}"
    branch = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{remote_ref}^{{commit}}"],
        cwd=checkout, stdout=subprocess.DEVNULL,
    )
    selected_ref = remote_ref if branch.returncode == 0 else ref
    for args in (
        ["checkout", "--detach", selected_ref],
        ["reset", "--hard", "HEAD"],
        ["clean", "-fdx"],
    ):
        subprocess.run(["git", *args], cwd=checkout, check=True)
    validate_source(checkout, stable_abi)
    subprocess.run(["git", "submodule", "sync", "--recursive"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=checkout, check=True
    )
    # Preserve .git for the existing .deepgemm-commit wheel provenance export.
    shutil.copytree(checkout, destination, symlinks=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vllm", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo", default="")
    parser.add_argument("--ref", default="")
    args = parser.parse_args()
    try:
        prepare(args.vllm, args.cache, args.destination, args.repo, args.ref)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"DeepGEMM preparation failed: {error}") from error


if __name__ == "__main__":
    main()
