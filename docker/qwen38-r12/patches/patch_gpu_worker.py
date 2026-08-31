from pathlib import Path


path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py")
source = path.read_text()
marker = "VLLM_CUDA_MEMORY_FRACTION safety ceiling"
needle = """            torch.accelerator.set_device_index(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)
"""
replacement = f"""            torch.accelerator.set_device_index(self.device)

            # {marker}. GB10 CUDA allocations are not fully charged to the
            # Docker memory cgroup, so bound PyTorch's allocator before NCCL,
            # model loading, compilation, KV allocation, and graph capture.
            cuda_memory_fraction = os.getenv(\"VLLM_CUDA_MEMORY_FRACTION\", \"\")
            if cuda_memory_fraction:
                fraction = float(cuda_memory_fraction)
                if not 0.0 < fraction <= 1.0:
                    raise ValueError(
                        \"VLLM_CUDA_MEMORY_FRACTION must be in (0, 1]\"
                    )
                torch.cuda.set_per_process_memory_fraction(fraction, self.device)
                logger.info(
                    \"Applied per-worker CUDA allocator ceiling: %.2f%%\",
                    fraction * 100.0,
                )

            current_platform.check_if_supports_dtype(self.model_config.dtype)
"""

if marker in source:
    raise SystemExit("gpu_worker.py is already safety patched")
if source.count(needle) != 1:
    raise SystemExit("expected gpu_worker.py insertion point exactly once")
path.write_text(source.replace(needle, replacement))
