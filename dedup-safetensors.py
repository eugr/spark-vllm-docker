#!/usr/bin/env python3
"""
dedup-safetensors.py — Remove duplicate tensor keys from sharded safetensors checkpoints.

Some quantized checkpoints produce shards with overlapping tensor keys.
fastsafetensors requires globally unique keys and will error with
"key ...qweight must be unique among files".

IMPORTANT: This script uses "last shard wins" semantics, NOT "index wins".
Standard PyTorch/HF loaders process shards sequentially (00001 → 00041).
If the same key appears in both shard 39 and shard 40, shard 40's version
silently overwrites shard 39's. This is significant because quantization
boundary bugs (e.g., Intel AutoRound on the 397B) can leave aborted/garbage
tensors in an earlier shard while the finalized version lands in a later shard.
The index.json may point to the EARLIER shard (the garbage), so trusting the
index would delete the correct data. Instead, we keep the last-writer-wins
version and update the index to match.

Usage:
    python3 dedup-safetensors.py /models/Qwen3.5-397B-A17B-int4-AutoRound
    python3 dedup-safetensors.py /models/... --dry-run

Requires: pip install safetensors torch
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from safetensors.torch import load_file, save_file


def scan_shards(model_dir: Path):
    """Scan all shards and find which keys exist in which shards."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found.")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Find all shard files (from index + any extras on disk)
    shard_files = sorted(set(weight_map.values()))

    # Build map: key -> list of shards that contain it (in shard order)
    key_locations = defaultdict(list)

    for shard_name in shard_files:
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            print(f"  WARN: {shard_name} referenced in index but not found, skipping.")
            continue

        tensors = load_file(str(shard_path), device="cpu")
        for key in tensors.keys():
            key_locations[key].append(shard_name)

    return key_locations, index, shard_files


def deduplicate(model_dir: Path, dry_run: bool = False):
    """Remove duplicate keys using last-shard-wins semantics."""
    print(f"Model directory: {model_dir}")
    print(f"Scanning all shards for duplicate keys...\n")

    key_locations, index, shard_files = scan_shards(model_dir)

    # Find keys that appear in multiple shards
    duplicated_keys = {k: shards for k, shards in key_locations.items() if len(shards) > 1}

    if not duplicated_keys:
        print("No duplicate keys found. Checkpoint is clean.")
        return

    # For each duplicated key, the LAST shard wins (matches standard loader behavior)
    # Remove the key from all earlier shards
    keys_to_remove = defaultdict(set)  # shard -> keys to remove from it
    index_updates = {}  # key -> new canonical shard

    for key, shards in duplicated_keys.items():
        winner = shards[-1]  # last shard wins
        for loser in shards[:-1]:
            keys_to_remove[loser].add(key)
        # Update index to point to the winner
        if index["weight_map"].get(key) != winner:
            index_updates[key] = winner

    total_removals = sum(len(keys) for keys in keys_to_remove.values())
    print(f"Found {len(duplicated_keys)} duplicated key(s) across shards.\n")

    print(f"Resolution (last shard wins):")
    for shard_name in sorted(keys_to_remove.keys()):
        keys = keys_to_remove[shard_name]
        print(f"\n  {shard_name}: remove {len(keys)} key(s)")
        for key in sorted(keys):
            winner = duplicated_keys[key][-1]
            index_ptr = index["weight_map"].get(key, "???")
            flag = " *** INDEX DISAGREED ***" if index_ptr == shard_name else ""
            print(f"    - {key}")
            print(f"      keep in: {winner}  |  index said: {index_ptr}{flag}")

    if index_updates:
        print(f"\n  Index corrections: {len(index_updates)} key(s) remapped")
        for key, new_shard in sorted(index_updates.items()):
            old_shard = index["weight_map"].get(key, "???")
            print(f"    {key}: {old_shard} -> {new_shard}")

    if dry_run:
        print(f"\nDry run — no files modified.")
        return

    print(f"\nDeduplicating...")

    # Rewrite shards that have keys to remove
    for shard_name in sorted(keys_to_remove.keys()):
        remove_keys = keys_to_remove[shard_name]
        shard_path = model_dir / shard_name
        backup_path = shard_path.with_suffix(".safetensors.bak")

        print(f"  {shard_name}: removing {len(remove_keys)} key(s)...")

        # Load all tensors
        tensors = load_file(str(shard_path), device="cpu")

        # Back up original
        shutil.copy2(shard_path, backup_path)

        # Remove duplicate keys (keeping them in the later shard)
        clean_tensors = {k: v for k, v in tensors.items() if k not in remove_keys}

        # Rewrite shard
        save_file(clean_tensors, str(shard_path))

        # Verify rewritten file is readable and has correct keys
        verify = load_file(str(shard_path), device="cpu")
        if set(verify.keys()) != set(clean_tensors.keys()):
            print(f"    ERROR: verification failed, restoring backup!")
            shutil.copy2(backup_path, shard_path)
            sys.exit(1)

        # Remove backup after successful verify
        backup_path.unlink()
        print(f"    OK — verified.")

    # Update index.json to reflect last-shard-wins
    if index_updates:
        index_path = model_dir / "model.safetensors.index.json"
        backup_index = index_path.with_suffix(".json.bak")
        shutil.copy2(index_path, backup_index)

        for key, new_shard in index_updates.items():
            index["weight_map"][key] = new_shard

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, sort_keys=False)
            f.write("\n")

        backup_index.unlink()
        print(f"\n  Updated index.json ({len(index_updates)} key(s) remapped).")

    print(f"\nDone. Removed {total_removals} duplicate(s) from {len(keys_to_remove)} shard(s).")
    print(f"fastsafetensors should now work with this checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove duplicate tensor keys from sharded safetensors checkpoints."
    )
    parser.add_argument("model_dir", type=Path, help="Path to model directory")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report duplicates without modifying files"
    )
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        print(f"ERROR: {args.model_dir} is not a directory.")
        sys.exit(1)

    deduplicate(args.model_dir, dry_run=args.dry_run)
