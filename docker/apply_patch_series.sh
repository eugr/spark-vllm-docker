#!/bin/bash
# Apply a checked-in source series only to its upstream build lane.
set -euo pipefail

project="$1"
repository="${2%/}"
repository="${repository%.git}"
enabled="$3"
series_dir="$4"

case "$project:$repository" in
    vllm:https://github.com/vllm-project/vllm|flashinfer:https://github.com/flashinfer-ai/flashinfer) ;;
    *)
        echo "Skipping regular $project patches for $repository."
        exit 0
        ;;
esac
if [ "$enabled" != 1 ]; then
    echo "Skipping regular $project patches for this source selection."
    exit 0
fi

# The source cache is copied into the builder; refresh its index stat data.
git update-index --refresh
git diff --quiet
git diff --cached --quiet
while IFS= read -r patch_name || [ -n "$patch_name" ]; do
    case "$patch_name" in ''|\#*) continue ;; esac
    patch_file="$series_dir/$patch_name"
    echo "Applying regular $project patch: $patch_name"
    if git apply --reverse --check --binary "$patch_file" >/dev/null 2>&1; then
        echo "Already present: $patch_name"
        continue
    fi
    # Keep conflicts visible, including tests/docs. Never discard one side.
    git apply --3way --index --binary "$patch_file"
    if ! git diff --cached --quiet; then
        git -c user.name='Docker Builder' -c user.email=builder@example.com \
            -c commit.gpgsign=false commit -m "Apply regular $project patch: $patch_name"
    fi
done < "$series_dir/series"
