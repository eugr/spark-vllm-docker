# syntax=docker/dockerfile:1.6
#
# FLUX.2 image server with on-the-fly torchao NVFP4 quantization for Blackwell.
# Self-contained: build with this folder as context.
#   docker build -f servers/flux2/Dockerfile -t flux2-torchao servers/flux2
#
# NVFP4 (W4A4, Triton kernels) gives a real speedup on Blackwell — measured
# ~3x on a DGX Spark GB10 (sm_121a): 28-step 1024^2 in ~45s vs ~2.3min BF16,
# and ~40% less VRAM (~66GB vs ~112GB). Based on the PyTorch Blackwell blog:
#   https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
#
# See README.md for details. Versions are pinned below (validated 2026-07-08).

FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# 12.1a target so native NVFP4 MMA is available on GB10 (sm_121a)
ENV TORCH_CUDA_ARCH_LIST="12.0a;12.1a"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
        git curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# CUDA 13 (cu130) is required for Blackwell (sm_120a/121a). torch is a cu130
# nightly (sm_121 support); torchao (PyPI) provides the NVFP4 workflow; mslk
# provides the NVFP4 Triton kernels (without it, torchao raises "mslk is
# required for NVFP4 triton quantization").
#
# Versions below are the exact set validated on a DGX Spark GB10, 2026-07-08.
# NOTE: torch / torchvision / mslk are cu130 *nightlies* — PyTorch prunes old
# nightlies from the index after a few weeks. If one of these pins 404s at build
# time, bump it to the nearest available cu130 nightly (see the index URL) and
# expect to re-validate; the stable PyPI pins (torchao, diffusers, …) do not age.
RUN pip install --pre \
        torch==2.14.0.dev20260707+cu130 \
        torchvision==0.29.0.dev20260708+cu130 \
        --index-url https://download.pytorch.org/whl/nightly/cu130
RUN pip install torchao==0.17.0 triton
RUN pip install --pre mslk==2026.7.7+cu130 \
        --index-url https://download.pytorch.org/whl/nightly/cu130

# FLUX.2 pipelines (diffusers) + server stack
RUN pip install \
        diffusers==0.39.0 \
        transformers==5.13.0 accelerate==1.14.0 safetensors==0.8.0 \
        sentencepiece protobuf \
        Pillow fastapi "uvicorn[standard]" pydantic python-multipart

WORKDIR /app
COPY server.py /app/server.py

ENV PYTHONUNBUFFERED=1
# (no forced offline — the recipe controls HF download vs cache)

EXPOSE 8000
ENTRYPOINT []
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
