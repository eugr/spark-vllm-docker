#!/usr/bin/env bash
set -e

echo "Reverting PR #35156"
if curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/35156.diff | patch -p1 -R -d /usr/local/lib/python3.12/dist-packages; then
  echo "    OK"
else
  echo "    Patch can't be reversed, skipping"
fi
