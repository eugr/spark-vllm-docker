#!/bin/bash
set -euo pipefail

# vllm-spark-blocked: true
# vllm-spark-safety-note: TP=2 on two GB10 128G nodes caused NVIDIA OOM and ARK-6 reboot.

echo "DeepSeek V4 Flash TP=2 is blocked on GB10: live launches caused NVRM out-of-memory and ARK-6 reboot."
echo "Use examples/deepseek-v4-flash-tp4.sh with at least four model-bearing nodes instead."
exit 78
