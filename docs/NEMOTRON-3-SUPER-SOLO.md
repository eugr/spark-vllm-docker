# Nemotron-3-Super-120B: Single Spark Deployment & Agentic Integration Guide

This guide covers running [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) on a **single** DGX Spark and integrating it with agentic coding IDEs (Cline, RooCode, Continue).

> **See also:** The existing `recipes/nemotron-3-super-nvfp4.yaml` targets **dual-node** clusters with `-tp 2` and Ray. This guide and the companion recipe `recipes/nemotron-3-super-nvfp4-solo.yaml` are for **single-node solo** deployments.

## Quick Start

```bash
./run-recipe.sh nemotron-3-super-nvfp4-solo --solo
# or with full setup (build + download + run):
./run-recipe.sh nemotron-3-super-nvfp4-solo --solo --setup
```

## Single-Node vs Dual-Node Recipe Differences

The existing `nemotron-3-super-nvfp4.yaml` recipe was designed for dual-Spark clusters. Several of its flags do not work or are suboptimal on a single node:

| Flag | Dual-Node Recipe | Solo Recipe | Reason |
|---|---|---|---|
| `--tensor-parallel-size` | 2 | *(omitted)* | Single GPU, no TP needed |
| `--distributed-executor-backend` | ray | *(omitted)* | No cluster orchestration |
| `--gpu-memory-utilization` | 0.7 | 0.85 | Must use more of the single node's memory |
| `--load-format` | fastsafetensors | *(default)* | fastsafetensors OOMs at 0.85 util |
| `--reasoning-parser` | nemotron_v3 | *(omitted)* | Causes TCP timeouts with IDE clients (see below) |
| `--attention-backend` | TRITON_ATTN (env) | FLASHINFER (env) | Better performance observed on single GB10 |
| `--max-model-len` | 262144 | 131072 | Reduced to fit in single-node memory |
| `--max-num-seqs` | 10 | 4 | Reduced for single-node KV cache budget |

## Agentic IDE Integration

### Cline / RooCode / Continue Settings

| Setting | Value |
|---|---|
| Provider | OpenAI Compatible |
| Base URL | `http://localhost:8000/v1` |
| Model ID | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |

### The Reasoning Parser Problem

**Do NOT use `--reasoning-parser nemotron_v3`** when connecting via Cline, RooCode, or similar streaming HTTP clients.

The reasoning parser suppresses the model's `<think>...</think>` output, sending it as a separate `reasoning_content` field instead of streaming it as text. This means:
1. The HTTP connection receives **zero bytes** during the entire thinking phase
2. Thinking can take 30–90 seconds for complex coding tasks
3. Most IDE clients have a 60-second TCP timeout
4. The connection drops before the model produces its answer

**Without the reasoning parser**, the model's chain-of-thought streams as regular text, keeping the TCP socket alive. The IDE client sees the thinking output (which can be noisy), but the connection stays stable.

### RooCode vs Cline

For local reasoning models, **RooCode** is recommended over Cline because:
- RooCode has "fuzzy parsing" that tolerates formatting noise from reasoning models
- Cline requires strict XML tool-call tags; Nemotron occasionally outputs JSON or markdown instead, causing parsing failures
- RooCode handles partial/malformed tool calls more gracefully

## Known Limitations

### MoE Backend
The `flashinfer_trtllm` MoE backend does not work on GB10 (Blackwell) as of 2026-04-04:
```
ValueError: NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support
the deployment configuration since kernel does not support current device cuda.
```
The CUTLASS backend is auto-selected and is currently the fastest available option for NVFP4 on Blackwell.

### Memory Constraints
At `--gpu-memory-utilization 0.85`, there is limited headroom for KV cache. With `--max-num-seqs 4` and `--max-model-len 131072`, expect:
- ~0.7% KV cache utilisation per active request
- OOM risk if you increase `max_num_seqs` significantly

### Cold Start
CUDA graph compilation takes 7–10 minutes on first startup. Subsequent requests after warmup are at full speed.

## Model Architecture Reference

| Property | Value |
|---|---|
| Architecture | NemotronH (hybrid Mamba-2 + LatentMoE + Attention) |
| Total parameters | 120 Billion |
| Active parameters | 12 Billion per token |
| Expert count | 512 (top-22 routing) |
| Quantisation | NVFP4 |
| Weight footprint | ~60 GB |
