#!/bin/bash
set -e

cd /usr/local/lib/python3.12/dist-packages
echo "Applying PR #38909"
if curl -fsL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/38909.diff | git apply --exclude="tests/*"; then
  echo "- PR #38909 applied successfully"
else
  echo "- PR #38909 can't be applied, skipping"
fi
