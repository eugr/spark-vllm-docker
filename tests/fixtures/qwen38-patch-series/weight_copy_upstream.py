# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpt from vllm/model_executor/utils.py
# at vllm-project/vllm@c7a760751977e4e3396d586956d8767a9bc44a82.

def copy_weight_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """`dst.copy_(src)` for weight loaders.

    Model-specific weight loaders should call this instead of `.copy_()` when
    copying a checkpoint tensor to the device. Off integrated GPUs it is a
    plain `copy_`.

    On integrated GPUs, a host-to-device copy straight from a safetensors mmap
    takes its page faults inside the driver copy, which is several times slower
    than faulting them on the CPU, so touch one byte per page of `src` first.
    PyTorch already stages non-contiguous and dtype-converting copies through
    a CPU temporary. Sources with at most one element are skipped: they count
    as contiguous with any stride, which `view(torch.uint8)` rejects.
    """
    if (
        src.device.type == "cpu"
        and src.dtype == dst.dtype
        and src.numel() > 1
        and src.is_contiguous()
        and _is_integrated_gpu(dst.device)
    ):
        flat = src.reshape(-1).view(torch.uint8)
        flat[:: mmap.PAGESIZE].sum()
        flat[-1:].sum()
    return dst.copy_(src)


def _extent(t: torch.Tensor) -> int:
    pass
