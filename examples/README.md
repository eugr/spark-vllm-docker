# Example Launch Scripts

This directory contains example bash scripts that demonstrate how to use the `--launch-script` option directly with `launch-cluster.sh`.

**Note:** For most use cases, the recipe system (`./run-recipe.sh`) is the recommended approach. These examples are provided for reference and for advanced users who need direct control over the launch process.

## Why Launch Scripts?

- **Simple** - Just write a bash script that runs your command
- **Flexible** - Use any bash features: environment variables, conditionals, loops
- **Focused** - Keep model and inference settings separate from cluster plumbing
- **Launcher-aware** - `launch-cluster.sh` supplies backend and per-node arguments

## Usage

```bash
# Use a launch script by name (looks in examples/ directory)
./launch-cluster.sh --launch-script example-vllm-minimax

# Use a launch script by filename
./launch-cluster.sh --launch-script example-vllm-minimax.sh

# Use a launch script with absolute path
./launch-cluster.sh --launch-script /path/to/my-script.sh

# Combine with mods if needed
./launch-cluster.sh --launch-script my-script.sh --apply-mod mods/my-patch

# Combine with other launcher options
./launch-cluster.sh --ray --launch-script my-model.sh -d
```

When using `--launch-script`, the `exec` action is automatically implied if no action is specified.
The launcher loads `.env` or uses autodiscovery, derives the node count from the
parallelism flags, and supplies the multiprocessing backend and per-node
arguments. Do not put `--distributed-executor-backend`, `--nnodes`,
`--node-rank`, `--master-addr`, `--master-port`, or `--headless` in a vLLM launch
script.

## Script Structure

Launch scripts are simple bash scripts. The script is copied into the container at `/workspace/exec-script.sh` and executed.

```bash
#!/bin/bash
# PROFILE: Human-readable name
# DESCRIPTION: What this script does

# Optional: Set environment variables
export MY_VAR="value"

# Run your command
vllm serve org/model-name \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.8
```

### Metadata Comments

The `# PROFILE:` and `# DESCRIPTION:` comments are optional but recommended for documentation:

```bash
#!/bin/bash
# PROFILE: MiniMax-M2-AWQ Example
# DESCRIPTION: vLLM serving MiniMax-M2-AWQ through launch-cluster.sh
```

## Examples

### Basic vLLM Serving

```bash
#!/bin/bash
# PROFILE: MiniMax-M2-AWQ
# DESCRIPTION: vLLM serving MiniMax-M2-AWQ through launch-cluster.sh

vllm serve QuantTrio/MiniMax-M2-AWQ \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.8 \
    -tp 2 \
    --max-model-len 128000 \
    --load-format fastsafetensors \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2
```

### With Environment Variables

```bash
#!/bin/bash
# PROFILE: OpenAI GPT-OSS 120B
# DESCRIPTION: vLLM serving openai/gpt-oss-120b with FlashInfer MOE optimization

# Enable FlashInfer MOE with MXFP4/MXFP8 quantization
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

vllm serve openai/gpt-oss-120b \
    --tool-call-parser openai \
    --enable-auto-tool-choice \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

### With Conditional Logic

```bash
#!/bin/bash
# PROFILE: Adaptive Model Server
# DESCRIPTION: Adjusts settings based on available GPUs

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPUs"

if [[ $GPU_COUNT -ge 4 ]]; then
    TP_SIZE=4
    MEM_UTIL=0.9
else
    TP_SIZE=2
    MEM_UTIL=0.8
fi

vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    -tp $TP_SIZE \
    --gpu-memory-utilization $MEM_UTIL
```

### SGLang

```bash
#!/bin/bash
# PROFILE: SGLang Llama 3.1
# DESCRIPTION: SGLang runtime with Llama 3.1

sglang launch meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --tp 2
```

### With Model Requiring Patches

If your model requires patches, use `--apply-mod` alongside `--launch-script`:

```bash
# Script: vllm-glm-4.7-nvfp4.sh
#!/bin/bash
# PROFILE: Salyut1/GLM-4.7-NVFP4
# DESCRIPTION: vLLM serving GLM-4.7-NVFP4
# NOTE: Requires --apply-mod mods/fix-Salyut1-GLM-4.7-NVFP4

vllm serve Salyut1/GLM-4.7-NVFP4 \
    --attention-config.backend flashinfer \
    --tool-call-parser glm47 \
    -tp 2 \
    --host 0.0.0.0 \
    --port 8000
```

Usage:
```bash
./launch-cluster.sh --launch-script vllm-glm-4.7-nvfp4.sh --apply-mod mods/fix-Salyut1-GLM-4.7-NVFP4
```

## Creating a New Launch Script

1. Create a new `.sh` file in this directory
2. Add the shebang `#!/bin/bash`
3. Add `# PROFILE:` and `# DESCRIPTION:` comments
4. Write your command (e.g., `vllm serve ...`)
5. Run with `./launch-cluster.sh --launch-script my-script.sh`

## Testing Scripts

Validate shell syntax directly, then use the launcher for runtime testing so it
can add the required cluster arguments:

```bash
bash -n examples/my-script.sh
./launch-cluster.sh --launch-script my-script.sh
```

This makes development and debugging much easier than complex configuration systems.
