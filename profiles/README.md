# Runtime Profiles

This directory contains bash profile scripts that define execution environments for the cluster. Profiles decouple runtime configuration from the `launch-cluster.sh` script, making it easy to switch between different runtimes (vLLM, SGLang, TensorRT-LLM, etc.) and model configurations.

## Why Bash Profiles?

- **No dependencies** - No need for `jq` or other JSON parsers
- **Custom logic** - Add environment setup, model downloads, conditional configuration
- **No escaping issues** - Native bash arrays handle quotes, spaces, and special characters
- **Extensible** - Use any bash feature: conditionals, loops, sourcing other files

## Usage

```bash
# Use a profile by name (looks in profiles/ directory)
./launch-cluster.sh --profile example-vllm-minimax exec

# Use a profile by filename
./launch-cluster.sh --profile example-vllm-minimax.sh exec

# Use a profile with absolute path
./launch-cluster.sh --profile /path/to/my-profile.sh exec

# Combine with other options
./launch-cluster.sh -n 192.168.1.1,192.168.1.2 --profile my-model -d exec
```

When using `--profile`, the `exec` action is automatically implied if no action is specified.

## Profile Structure

```bash
#!/bin/bash
# Profile: Human-readable name
# Description: What this profile does

# Metadata (optional)
PROFILE_NAME="Human-readable name"
PROFILE_DESCRIPTION="Description of this profile"

# Runtime configuration (required)
PROFILE_RUNTIME="vllm"           # The runtime executable
PROFILE_COMMAND="serve"          # The command (can be empty)
PROFILE_MODEL="org/model-name"   # The model path or identifier

# Arguments array
PROFILE_ARGS=(
    --port 8000
    --host 0.0.0.0
    --flag-without-value
    -tp 2
)

# Optional: initialization hook
profile_init() {
    # Custom setup logic runs before the main command
    export MY_VAR="value"
}
```

### Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PROFILE_NAME` | No | Human-readable name for the profile |
| `PROFILE_DESCRIPTION` | No | Description of what this profile does |
| `PROFILE_RUNTIME` | Yes | The runtime executable (e.g., `vllm`, `sglang`, `trtllm-serve`) |
| `PROFILE_COMMAND` | No | The command to run (e.g., `serve`, `launch`). Can be empty. |
| `PROFILE_MODEL` | No | The model path or identifier |
| `PROFILE_ARGS` | No | Bash array of arguments |
| `PROFILE_MODS` | No | Bash array of mod paths to apply (relative to repo root or absolute) |

### Arguments

Arguments are specified as a bash array. Each element can be:
- A flag: `--enable-auto-tool-choice`
- A key-value pair: `--port 8000` (as two elements) or `--port=8000` (as one element)

```bash
# Two styles work equally well:
PROFILE_ARGS=(
    --port 8000              # Separate elements
    --config=/path/to/file   # Combined with =
    --enable-feature         # Boolean flag
)
```

### Initialization Hook

The optional `profile_init()` function runs before the main command executes. Use it for:

- Setting environment variables
- Downloading models or data
- Conditional configuration based on environment
- Validation or pre-flight checks

```bash
profile_init() {
    # Ensure HF token is available
    if [[ -z "$HF_TOKEN" ]]; then
        echo "Warning: HF_TOKEN not set, gated models may fail"
    fi
    
    # Pre-download the model
    huggingface-cli download "$PROFILE_MODEL" --quiet
    
    # Set CUDA devices based on available GPUs
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
}
```

## Examples

### vLLM Serving

```bash
#!/bin/bash
PROFILE_NAME="vLLM MiniMax M2"
PROFILE_RUNTIME="vllm"
PROFILE_COMMAND="serve"
PROFILE_MODEL="QuantTrio/MiniMax-M2-AWQ"

PROFILE_ARGS=(
    --port 8000
    --host 0.0.0.0
    --gpu-memory-utilization 0.7
    -tp 2
    --distributed-executor-backend ray
    --max-model-len 128000
    --load-format fastsafetensors
    --enable-auto-tool-choice
    --tool-call-parser minimax_m2
)
```

### SGLang

```bash
#!/bin/bash
PROFILE_NAME="SGLang Llama 3.1"
PROFILE_RUNTIME="sglang"
PROFILE_COMMAND="launch"
PROFILE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

PROFILE_ARGS=(
    --port 8000
    --host 0.0.0.0
    --tp 2
)
```

### TensorRT-LLM

```bash
#!/bin/bash
PROFILE_NAME="TensorRT-LLM Llama"
PROFILE_RUNTIME="trtllm-serve"
PROFILE_COMMAND=""
PROFILE_MODEL="/models/llama-engine"

PROFILE_ARGS=(
    --host 0.0.0.0
    --port 8000
)
```

### With Custom Initialization

```bash
#!/bin/bash
PROFILE_NAME="Production Deployment"
PROFILE_RUNTIME="vllm"
PROFILE_COMMAND="serve"
PROFILE_MODEL="meta-llama/Llama-3.1-70B-Instruct"

PROFILE_ARGS=(
    --port 8000
    --host 0.0.0.0
    -tp 8
    --distributed-executor-backend ray
)

profile_init() {
    # Load secrets from environment or file
    if [[ -f ~/.hf_token ]]; then
        export HF_TOKEN=$(cat ~/.hf_token)
    fi
    
    # Pre-download model on all nodes would happen via mods system
    echo "Starting production deployment of $PROFILE_MODEL"
    
    # Validate GPU availability
    local gpu_count=$(nvidia-smi -L | wc -l)
    echo "Found $gpu_count GPUs"
}
```

### With Mods/Patches

Profiles can include mods that are automatically applied before launch:

```bash
#!/bin/bash
PROFILE_NAME="GLM-4.7-NVFP4"
PROFILE_RUNTIME="vllm"
PROFILE_COMMAND="serve"
PROFILE_MODEL="Salyut1/GLM-4.7-NVFP4"

PROFILE_ARGS=(
    --attention-config.backend flashinfer
    --tool-call-parser glm47
    --reasoning-parser glm45
    --enable-auto-tool-choice
    -tp 2
    --gpu-memory-utilization 0.88
    --max-model-len 32000
    --distributed-executor-backend ray
    --host 0.0.0.0
    --port 8000
)

# Required mod to fix k/v scales incompatibility
PROFILE_MODS=(
    "mods/fix-Salyut1-GLM-4.7-NVFP4"
)
```

## Creating a New Profile

1. Copy an existing example profile
2. Modify the variables for your use case
3. Add a `profile_init()` function if you need custom setup
4. Save with a descriptive name (e.g., `my-model-prod.sh`)
5. Run with `./launch-cluster.sh --profile my-model-prod exec`

## Migration from JSON Profiles

If you have existing JSON profiles, convert them as follows:

**JSON:**
```json
{
  "runtime": "vllm",
  "command": "serve",
  "model": "org/model",
  "args": {
    "--port": "8000",
    "--enable-flag": null
  }
}
```

**Bash:**
```bash
#!/bin/bash
PROFILE_RUNTIME="vllm"
PROFILE_COMMAND="serve"
PROFILE_MODEL="org/model"
PROFILE_ARGS=(
    --port 8000
    --enable-flag
)
```
