#!/usr/bin/env python3
"""Cache inventory and local transfers for hf-download.sh (Python stdlib only)."""

import argparse
import csv
import errno
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def model_directory(cache, model):
    # Prevent model IDs from becoming paths or options during local/SSH operations.
    if not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", model, flags=re.ASCII) or any(
        part in (".", "..") or ".." in part or "--" in part or part.startswith("-")
        for part in model.split("/")
    ):
        raise ValueError(f"Invalid model ID: {model!r}")
    return cache / ("models--" + model.replace("/", "--"))


def cache_entries(cache, sort):
    if not cache.exists():
        return []
    result = subprocess.run(
        ["uvx", "hf", "cache", "ls", "--cache-dir", str(cache),
         "--revisions", "--format", "json", "--sort", sort],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    entries = json.loads(result.stdout)
    if not isinstance(entries, list):
        raise ValueError("Expected a JSON array from hf cache ls; update your hf CLI.")
    return entries


def snapshot_files(snapshot):
    files = {}
    for root, directories, names in os.walk(snapshot, onerror=raise_error):
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError(f"Symlinked snapshot directory: {Path(root) / name}")
        for name in names:
            path = Path(root) / name
            files[path.relative_to(snapshot).as_posix()] = path
    return files


def raise_error(error):
    raise error


def inventory(cache, sort, node, model=""):
    rows = []
    entries = cache_entries(cache, sort)
    owners = {}
    for entry in entries:
        owners.setdefault(entry["revision"], set()).add((entry["repo_type"], entry["repo_id"]))
    for entry in entries:
        if entry["repo_type"] != "model" or (model and entry["repo_id"] != model):
            continue
        repo = model_directory(cache, entry["repo_id"])
        revision = entry["revision"]
        if not re.fullmatch(r"[\w-]+", revision, flags=re.ASCII):
            raise ValueError(f"Invalid cached revision: {revision!r}")
        snapshot = repo / "snapshots" / revision
        if not snapshot.is_dir():
            raise ValueError(f"Missing snapshot: {snapshot}")
        files = snapshot_files(snapshot)
        stats = {path.resolve(strict=True): path.stat() for path in files.values()}
        rows.append({
            "model": entry["repo_id"], "revision": revision, "refs": entry["refs"],
            "other_repos": sorted(f"{kind}/{name}" for kind, name in owners[revision]
                                 if (kind, name) != ("model", entry["repo_id"])),
            "size_bytes": sum(stat.st_size for stat in stats.values()),
            "modified": max((stat.st_mtime for stat in stats.values()), default=snapshot.stat().st_mtime),
            "accessed": max((stat.st_atime for stat in stats.values()), default=repo.stat().st_atime),
            "files": {name: path.stat().st_size for name, path in files.items()},
        })
    if model:
        repo = model_directory(cache, model)
        if repo.exists() or repo.is_symlink():
            # hf can skip inconsistent repositories with only a warning. A
            # backup/delete preflight must not mistake skipped data for absence.
            validate_tree(repo, cache)
            revisions = {path.name for path in (repo / "snapshots").iterdir()}
            if revisions != {row["revision"] for row in rows}:
                raise ValueError(f"Incomplete inventory for {model}; refusing to treat unlisted revisions as absent.")
    # hf sorts access time at repository level, even when listing revisions.
    for row in rows:
        row["accessed"] = max(r["accessed"] for r in rows if r["model"] == row["model"])
    return {"node": node, "entries": rows}


def resolve_revision(model, selector, inventories):
    rows = [row for item in inventories for row in item["entries"] if row["model"] == model]
    matches = {row["revision"] for row in rows
               if row["revision"] == selector.lower() or selector in row["refs"]}
    if not matches and re.fullmatch(r"[0-9a-fA-F]{7,40}", selector):
        matches = {row["revision"] for row in rows if row["revision"].startswith(selector.lower())}
    if not matches:
        raise ValueError(f"Revision {selector!r} of {model} is not cached on the selected node(s).")
    if len(matches) != 1:
        raise ValueError(f"Revision {selector!r} is ambiguous across the selected nodes: "
                         f"{', '.join(sorted(matches))}. Specify a full commit hash.")
    return matches.pop()


def check_revision_targets(model, revision, inventories):
    # hf cache rm hashes are global targets, including other models/datasets.
    for item in inventories:
        for row in item["entries"]:
            if row["model"] == model and row["revision"] == revision and row["other_repos"]:
                raise ValueError(f"Cannot delete {model}@{revision} on {item['node']}: "
                                 f"the hash also belongs to {', '.join(row['other_repos'])}.")


def render(inventories, output_format, sort, show_nodes, model=""):
    merged = {}
    for item in inventories:
        for row in item["entries"]:
            if model and row["model"] != model:
                continue
            key = (row["model"], row["revision"])
            if key not in merged:
                merged[key] = {**row, "nodes": []}
            result = merged[key]
            if item["node"] not in result["nodes"]:
                result["nodes"].append(item["node"])
            for field in ("size_bytes", "modified", "accessed"):
                result[field] = max(result[field], row[field])
    key, _, order = sort.lower().partition(":")
    field = {"size": "size_bytes", "name": "model", "modified": "modified", "accessed": "accessed"}[key]
    rows = sorted(merged.values(), key=lambda r: (r["model"], r["revision"]))
    rows.sort(key=lambda r: r[field].lower() if key == "name" else r[field],
              reverse=(order or ("asc" if key == "name" else "desc")) == "desc")
    output = []
    for row in rows:
        result = {"model": row["model"], "revision": row["revision"],
                  "size_gb": row["size_bytes"] / 1_000_000_000}
        if show_nodes:
            result["nodes"] = row["nodes"]
        output.append(result)
    if output_format == "json":
        print(json.dumps(output, indent=2))
        return
    fields = ["model", "revision", "size_gb"] + (["nodes"] if show_nodes else [])
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**row, **({"nodes": ";".join(row["nodes"])} if show_nodes else {})}
                         for row in output)
        return
    headers = ["Model", "Revision", "Size (GB)"] + (["Nodes"] if show_nodes else [])
    cells = [[row["model"], row["revision"], f'{row["size_gb"]:.3f}'] +
             ([", ".join(row["nodes"])] if show_nodes else []) for row in output]
    if output_format == "md":
        for row in [headers, ["---"] * len(headers), *cells]:
            print("| " + " | ".join(value.replace("|", r"\|") for value in row) + " |")
    else:
        widths = [max(len(row[i]) for row in [headers, *cells]) for i in range(len(headers))]
        for row in [headers, *cells]:
            print("  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip())
        if not cells:
            print("No models found.")


def validate_tree(model_dir, boundary):
    """Reject broken/external links and unsafe directory links before copying."""
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ValueError(f"Model directory missing or symlinked: {model_dir}")
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        raise ValueError(f"No snapshots in {model_dir}")
    for root, directories, files in os.walk(model_dir, onerror=raise_error):
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError(f"Symlinked cache directory: {Path(root) / name}")
        for name in files:
            path = Path(root) / name
            target = path.resolve(strict=True)
            if not target.is_relative_to(boundary.resolve()) or not target.is_file():
                raise ValueError(f"Unsafe cache file: {path}")
    refs = model_dir / "refs"
    if refs.exists():
        for path in refs.rglob("*"):
            if path.is_file():
                revision = path.read_text().strip()
                if not re.fullmatch(r"[\w-]+", revision, flags=re.ASCII) or not (snapshots / revision).is_dir():
                    raise ValueError(f"Ref points to a missing snapshot: {path}")


def revision_files(source, revision):
    """Select one snapshot, its blob links, and metadata for that commit."""
    snapshot = source / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(f"Missing snapshot: {snapshot}")
    paths = {snapshot, *snapshot_files(snapshot).values()}
    no_exist = source / ".no_exist" / revision
    if no_exist.is_dir():
        paths.update((no_exist, *snapshot_files(no_exist).values()))
    tree = source / "trees" / f"{revision}.json"
    if tree.is_file():
        paths.add(tree)
    for ref in (source / "refs").rglob("*"):
        if ref.is_file() and ref.read_text().strip() == revision:
            paths.add(ref)
    for path in list(paths):
        while path.is_symlink():
            target = Path(os.path.abspath(path.parent / os.readlink(path)))
            if not target.is_relative_to(source):
                # rsync --copy-unsafe-links materializes shared Hub blobs here.
                break
            paths.add(target)
            path = target
    return sorted(path.relative_to(source).as_posix() for path in paths)


def supports_symlinks(directory):
    """Probe inside our private staging directory, without touching user files."""
    probe = directory / ".symlink-probe"
    try:
        probe.symlink_to(".", target_is_directory=True)
    except OSError as error:
        if error.errno in (errno.EPERM, errno.EACCES, errno.EOPNOTSUPP, errno.ENOSYS):
            return False
        raise
    else:
        probe.unlink()
        return True


def transfer(cache, backup, model, restore=False, revision=""):
    cache, backup = cache.resolve(), backup.resolve(strict=True)
    if not backup.is_dir():
        raise ValueError(f"Backup directory does not exist: {backup}")
    if cache.is_relative_to(backup) or backup.is_relative_to(cache):
        raise ValueError("Backup directory and Hub cache must not overlap.")
    source_root, destination_root = (backup, cache) if restore else (cache, backup)
    source = model_directory(source_root, model)
    destination = model_directory(destination_root, model)
    validate_tree(source, source if restore else cache)
    if destination.exists() or destination.is_symlink():
        if not restore:
            raise ValueError(f"Backup already exists: {destination}. Use a different backup directory.")
        validate_tree(destination, cache)
    destination_root.mkdir(parents=True, exist_ok=True)
    # Incomplete backups stay hidden and are removed on failure. Publish only a
    # complete, self-contained repo, materializing links to hub-level blobs.
    with tempfile.TemporaryDirectory(prefix=".hf-transfer-", dir=destination_root) as temporary:
        staged = Path(temporary) / source.name
        staged.mkdir()
        copy_options = ["-a", "--copy-unsafe-links"]
        if not restore and not supports_symlinks(Path(temporary)):
            print("Destination symlinks are unavailable; backing up regular snapshot files.", file=sys.stderr)
            # Flat snapshots are a supported Hub cache layout. Dereference the
            # snapshot links and omit the separate blobs to avoid two copies of
            # every weight. These drives may also lack Unix ownership/permissions.
            copy_options = ["-t", "--dirs", "--copy-links", "--exclude=/blobs/***"]
            if not revision:
                copy_options.append("--recursive")
        options = []
        if revision:
            # With --files-from, -a does not imply recursion. Only these files
            # and their parent directories are copied, leaving other weights out.
            file_list = Path(temporary) / "files"
            file_list.write_bytes(b"\0".join(os.fsencode(path) for path in revision_files(source, revision)) + b"\0")
            options = ["--from0", f"--files-from={file_list}"]
        subprocess.run(["rsync", *copy_options, "--exclude=*.incomplete", *options,
                        str(source) + "/", str(staged) + "/"], check=True)
        validate_tree(staged, staged)
        if restore and destination.exists():
            # Keep revisions already on the head; never follow a destination's
            # links out to the shared blob store while writing restored files.
            subprocess.run(["rsync", "-a", "--copy-unsafe-links", str(staged) + "/",
                            str(destination) + "/"], check=True)
            validate_tree(destination, cache)
        else:
            staged.rename(destination)
    label = f"{model}@{revision}" if revision else model
    print(f'{"Restored" if restore else "Backed up"} {label}: {destination}', file=sys.stderr)


def check_backed_up(backup, model, inventories, revision=""):
    source = model_directory(backup, model)
    validate_tree(source, source)
    for item in inventories:
        for row in item["entries"]:
            if row["model"] != model or (revision and row["revision"] != revision):
                continue
            snapshot = source / "snapshots" / row["revision"]
            for name, size in row["files"].items():
                path = snapshot / name
                if not path.is_file() or path.stat().st_size != size:
                    raise ValueError(f"Backup does not cover {model}@{row['revision']} on {item['node']}; "
                                     "no models were deleted.")
            if not snapshot.is_dir():
                raise ValueError(f"Backup is missing revision {row['revision']}; no models were deleted.")


def delete(cache, model, cleanup, revision=""):
    if not cache.exists():
        return
    if cleanup:
        entries = cache_entries(cache, "name")
        # Revision hashes are global hf rm targets. Do not delete a hash also
        # belonging to a referenced revision or to a dataset/Space.
        protected = {e["revision"] for e in entries if e["repo_type"] != "model" or e["refs"]}
        targets = sorted({e["revision"] for e in entries
                          if e["repo_type"] == "model" and not e["refs"]} - protected)
    elif revision:
        item = inventory(cache, "name", "this node", model)
        if not any(row["revision"] == revision for row in item["entries"]):
            print(f"Skipping {model}@{revision}: not cached on this node.", file=sys.stderr)
            return
        check_revision_targets(model, revision, [item])
        targets = [revision]
    else:
        model_directory(cache, model)
        targets = ["model/" + model]
    if targets:
        subprocess.run(["uvx", "hf", "cache", "rm", *targets,
                        "--cache-dir", str(cache), "--yes"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "render", "backup", "restore", "delete", "covered",
                                          "validate", "resolve-revision", "check-revision"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--node", default="local")
    parser.add_argument("--nodes", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--format", choices=("console", "csv", "json", "md"), default="console")
    parser.add_argument("--sort", default="size:desc")
    parser.add_argument("files", nargs="*")
    args = parser.parse_intermixed_args()
    if not re.fullmatch(r"(name|size|accessed|modified)(:(asc|desc))?", args.sort.lower()):
        raise ValueError("Invalid --sort; use name|size|accessed|modified[:asc|:desc].")
    if args.model:
        model_directory(Path(), args.model)
    if args.revision and (not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)*", args.revision, flags=re.ASCII)
                          or any(part in (".", "..") for part in args.revision.split("/"))):
        raise ValueError("Invalid --revision; use a cached commit hash, unique hash prefix, branch, or tag.")
    inventories = [json.loads(Path(path).read_text()) for path in args.files]
    if args.action == "inventory":
        print(json.dumps(inventory(args.cache_dir, args.sort, args.node, args.model)))
    elif args.action == "render":
        render(inventories, args.format, args.sort, args.nodes, args.model)
    elif args.action in ("backup", "restore"):
        transfer(args.cache_dir, args.backup_dir, args.model, args.action == "restore", args.revision)
    elif args.action == "covered":
        check_backed_up(args.backup_dir, args.model, inventories, args.revision)
    elif args.action == "delete":
        delete(args.cache_dir, args.model, args.cleanup, args.revision)
    elif args.action == "resolve-revision":
        if not inventories:
            inventories = [inventory(args.cache_dir, "name", args.node, args.model)]
        print(resolve_revision(args.model, args.revision, inventories))
    elif args.action == "check-revision":
        check_revision_targets(args.model, args.revision, inventories)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"Error: {error}")
