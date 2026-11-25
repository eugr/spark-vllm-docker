#!/usr/bin/env python3
"""Collect one memory-profile run from local/SSH containers and write its card."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile

from profile_card import write_card


def extract(archive, destination):
    # Docker cp emits a tar stream. Only directories and regular profile files
    # are needed; never restore links, devices, owners or archive permissions.
    with tarfile.open(fileobj=archive, mode="r:*") as stream:
        for member in stream:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe path in collected profile archive")
            path = destination.joinpath(*relative.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
            else:
                raise ValueError("Unsupported entry in collected profile archive")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="append", required=True, help="SSH host/alias, or local; repeat for every node")
    parser.add_argument("--container", default="vllm_node")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--directory", default="/memory-profiles", help="Output directory inside the containers")
    parser.add_argument("--output", type=Path, required=True, help="New local directory for collected raw data and profile.yaml")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.run_id):
        parser.error("Invalid run ID")
    if not PurePosixPath(args.directory).is_absolute():
        parser.error("--directory must be absolute")
    if args.output.exists():
        parser.error("--output must be a new directory to preserve previous profiles")
    args.output.mkdir(parents=True)
    directories = []
    try:
        for index, host in enumerate(args.host):
            command = ["docker", "cp", f"{args.container}:{args.directory}/{args.run_id}/.", "-"]
            if host != "local":
                command = ["ssh", "-o", "BatchMode=yes", "--", host, shlex.join(command)]
            with tempfile.TemporaryFile() as archive:
                result = subprocess.run(command, stdout=archive, stderr=subprocess.PIPE)
                if result.returncode:
                    raise ValueError(f"Collection from {host} failed: {result.stderr.decode(errors='replace').strip()}")
                archive.seek(0)
                destination = args.output / f"node-{index}"
                destination.mkdir()
                extract(archive, destination)
                directories.append(destination)
        card = write_card(directories, args.output / "profile.yaml")
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f"memory-profile: {error}; any collected files remain in {args.output}\n")
    print(f"{card['status']}: {args.output / 'profile.yaml'}")


if __name__ == "__main__":
    main()
