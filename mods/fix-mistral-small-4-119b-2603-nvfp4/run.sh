#!/bin/bash
set -e

cd /usr/local/lib/python3.12/dist-packages
echo "Applying vllm-project/vllm#41119"
if curl -fsL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/41119.diff | git apply --exclude="tests/*"; then
  echo "- PR vllm-project/vllm#41119 applied successfully"
else
  echo "- PR vllm-project/vllm#41119 can't be applied, skipping"
fi
