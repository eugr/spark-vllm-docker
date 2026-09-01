# WSL CUDA memory fix

This mod adapts current vLLM releases for CUDA on WSL2:

- `VLLM_WSL2_ENABLE_PIN_MEMORY` defaults to `1`, enabling vLLM's native pinned
  memory and UVA path on a compatible WSL2 kernel and NVIDIA driver.
- An explicit `VLLM_WSL2_ENABLE_PIN_MEMORY=0` still disables the feature.
- vLLM skips its Linux integrated-GPU host-memory adjustment under WSL and
  retains the raw CUDA memory values, which include the WDDM shared-memory
  allocation budget on supported drivers.

The mod does not install a non-UVA buffer fallback and does not use NVML for
capacity reporting. NVML reports the dedicated GPU segment under WSL rather
than CUDA's total allocatable budget on the validated RTX Spark configuration.

Apply the mod before starting vLLM:

```bash
./launch-cluster.sh --solo --apply-mod mods/uma-fix exec vllm serve ...
```

Pinned memory is page-locked host memory. Keep the WSL VM memory limit and
vLLM's pinned/offloaded allocation sizes within the host's safe operating
range.
