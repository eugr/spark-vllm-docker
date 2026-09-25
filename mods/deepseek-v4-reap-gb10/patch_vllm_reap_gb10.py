#!/usr/bin/env python3
"""Native-image patches for serving REAP DeepSeek-V4 MXFP4 checkpoints on GB10.

Two text patches against the ``vllm-node-dsv4:latest`` image (vllm-project/vllm
#41834 lineage):

1. Router fallback for nonstandard REAP expert counts (e.g. K160 -> 160,
   200B -> 180, 148B -> 130). The fused ``sqrtsoftplus`` CUDA top-k kernel is
   only instantiated for {16,32,64,128,192,256,320,384,512}; route others to the
   existing pure-Torch path.

2. MoE weight-processing memory hygiene. On SM12x the only viable MXFP4 MoE
   backend is Marlin, whose per-layer ``process_weights_after_loading`` appears
   to let the CUDA caching allocator hoard freed blocks across the 43 expert
   layers, overflowing GB10's 128 GiB unified memory at model init. We append a
   ``gc.collect() + torch.cuda.empty_cache()`` after each layer's setup (returns
   cached-but-free unified memory to the system) and log allocated/reserved so
   per-layer growth is visible.
"""

from __future__ import annotations

from pathlib import Path

ROUTER_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "fused_moe/router/fused_topk_bias_router.py"
)
MXFP4_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "quantization/mxfp4.py"
)
IMPORT_UTILS_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/utils/import_utils.py"
)
FLASHINFER_CUDA_IPC_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/flashinfer/comm/cuda_ipc.py"
)

# GB10/SM121: the DeepSeek-V4 cute-dsl indexer / K-cache-gather kernels crash in
# cutlass cute-dsl 4.5.0 MLIR codegen ("Expected an MLIR object ... OpResultList")
# during warmup. has_cutedsl()-gated call sites have Triton fallbacks; force them.
# Controlled by the K160_DISABLE_CUTEDSL env so we can A/B it.
HASCUTEDSL_OLD = (
    '    """Whether the optional `cutelass` package is available."""\n'
    "    return _has_module(\"cutlass\")"
)
HASCUTEDSL_NEW = (
    '    """Whether the optional `cutlass` package is available."""\n'
    "    import os as _os\n"
    '    if _os.environ.get("K160_DISABLE_CUTEDSL", "0") == "1":\n'
    "        return False\n"
    "    return _has_module(\"cutlass\")"
)

SUPPORTED = "(16, 32, 64, 128, 192, 256, 320, 384, 512)"

ROUTER_OLD = "    if not rocm_aiter_ops.is_fused_moe_enabled():"
ROUTER_NEW = (
    "    # REAP DeepSeek-V4 checkpoints prune to nonstandard routed-expert\n"
    "    # counts (e.g. 160, 180, 130). The fused sqrtsoftplus CUDA top-k kernel\n"
    "    # is only instantiated for a fixed set of counts. Route others to the\n"
    "    # pure-Torch fallback below instead of crashing during warmup.\n"
    "    if not rocm_aiter_ops.is_fused_moe_enabled() and not (\n"
    '        scoring_func == "sqrtsoftplus"\n'
    f"        and gating_output.shape[-1] not in {SUPPORTED}\n"
    "    ):"
)

TOPK_OLD = (
    "    if current_platform.is_xpu():\n"
    "        return _topk_softplus_sqrt_torch(\n"
)
TOPK_NEW = (
    "    if current_platform.is_xpu() or gating_output.shape[-1] not in "
    f"{SUPPORTED}:\n"
    "        return _topk_softplus_sqrt_torch(\n"
)

# Anchor on the Mxfp4MoEMethod (DeepSeek-V4) variant: its def line has no type
# annotation, unlike GptOssMxfp4MoEMethod's ``(self, layer: RoutedExperts) -> None``.
MXFP4_OLD = (
    "    def process_weights_after_loading(self, layer):\n"
    "        w13 = layer.w13_weight\n"
    "        w2 = layer.w2_weight\n"
    "        w13_scale = layer.w13_weight_scale\n"
    "        w2_scale = layer.w2_weight_scale\n"
    '        w13_bias = getattr(layer, "w13_bias", None)\n'
    '        w2_bias = getattr(layer, "w2_bias", None)\n'
    "\n"
    "        if self.mxfp4_backend == Mxfp4MoeBackend.NONE:\n"
    "            return\n"
    "\n"
    "        self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)"
)
MXFP4_NEW = (
    MXFP4_OLD
    + "\n"
    "        # SM12x Marlin per-layer memory hygiene: release cached-but-free\n"
    "        # unified memory so the 43-layer MoE setup does not overflow 128 GiB.\n"
    "        import torch as _t, gc as _gc\n"
    "        _gc.collect()\n"
    "        _t.cuda.empty_cache()\n"
    "        try:\n"
    '            print("[k160-moe-patch] layer setup alloc=%.1fGB reserved=%.1fGB"\n'
    "                  % (_t.cuda.memory_allocated() / 1e9, _t.cuda.memory_reserved() / 1e9),\n"
    "                  flush=True)\n"
    "        except Exception:\n"
    "            pass"
)

CUDA_IPC_OLD = (
    "        if so_file is None:\n"
    '            so_file = find_loaded_library("libcudart")\n'
    '            assert so_file is not None, "libcudart is not loaded in the current process"'
)
CUDA_IPC_NEW = (
    "        if so_file is None:\n"
    '            loaded_cudart = find_loaded_library("libcudart")\n'
    "            # TileLang ships a libcudart_stub.so that can be loaded before\n"
    "            # the real CUDA runtime. FlashInfer CUDA IPC needs symbols such\n"
    "            # as cudaDeviceReset, so prefer the real CUDA 13 wheel runtime.\n"
    "            preferred_cudart = (\n"
    '                "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/"\n'
    '                "libcudart.so.13"\n'
    "            )\n"
    "            import os as _os\n"
    "            if loaded_cudart and \"libcudart_stub\" not in loaded_cudart:\n"
    "                so_file = loaded_cudart\n"
    "            elif _os.path.exists(preferred_cudart):\n"
    "                so_file = preferred_cudart\n"
    "            else:\n"
    "                so_file = loaded_cudart\n"
    '            assert so_file is not None, "libcudart is not loaded in the current process"'
)


def patch_once(target: Path, old: str, new: str) -> str:
    text = target.read_text()
    if new in text:
        return f"PATCH_ALREADY_APPLIED {target.name}"
    if old not in text:
        raise SystemExit(f"PATCH_TARGET_NOT_FOUND {target}")
    if text.count(old) != 1:
        raise SystemExit(f"PATCH_TARGET_NOT_UNIQUE {target} count={text.count(old)}")
    target.write_text(text.replace(old, new, 1))
    return f"PATCH_APPLIED {target.name}"


def maybe_swap_cutedsl() -> None:
    """Optionally pin a different nvidia-cutlass-dsl version before serving.

    The DSv4 sparse-MLA compressor crashes in cutlass cute-dsl 4.5.0 MLIR
    codegen on GB10 ("Expected an MLIR object ... OpResultList"), which looks
    like an API-version mismatch. Set CUTEDSL_VERSION to test an older release.
    """
    import os
    import subprocess
    import sys

    version = os.environ.get("CUTEDSL_VERSION", "").strip()
    if not version:
        return
    print(f"CUTEDSL_SWAP installing nvidia-cutlass-dsl=={version}", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--force-reinstall",
         "--no-deps", "--no-cache-dir", f"nvidia-cutlass-dsl=={version}"]
    )
    # cutlass.__version__ is a stale hardcoded string; trust package metadata.
    from importlib.metadata import version as _pkgver

    print(f"CUTEDSL_SWAP_OK installed={_pkgver('nvidia-cutlass-dsl')}", flush=True)


def main() -> int:
    import py_compile

    maybe_swap_cutedsl()
    print(patch_once(ROUTER_TARGET, ROUTER_OLD, ROUTER_NEW))
    print(patch_once(ROUTER_TARGET, TOPK_OLD, TOPK_NEW))
    py_compile.compile(str(ROUTER_TARGET), doraise=True)
    mxfp4_text = MXFP4_TARGET.read_text()
    if (
        "self._setup_kernel(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias)" in mxfp4_text
        and "torch.cuda.empty_cache()" in mxfp4_text
    ):
        # Newer preview images already release cached memory after MXFP4 setup.
        print(f"PATCH_ALREADY_EQUIVALENT {MXFP4_TARGET.name}")
    else:
        print(patch_once(MXFP4_TARGET, MXFP4_OLD, MXFP4_NEW))
    py_compile.compile(str(MXFP4_TARGET), doraise=True)
    import_utils_text = IMPORT_UTILS_TARGET.read_text()
    if HASCUTEDSL_NEW in import_utils_text:
        print(f"PATCH_ALREADY_APPLIED {IMPORT_UTILS_TARGET.name}")
    elif HASCUTEDSL_OLD in import_utils_text:
        print(patch_once(IMPORT_UTILS_TARGET, HASCUTEDSL_OLD, HASCUTEDSL_NEW))
    else:
        # Newer preview images moved/removed this helper and use a different
        # sparse-MLA path. The override is only a fallback for older images.
        print(f"PATCH_SKIPPED_NO_CUTEDSL_HELPER {IMPORT_UTILS_TARGET.name}")
    py_compile.compile(str(IMPORT_UTILS_TARGET), doraise=True)
    print(patch_once(FLASHINFER_CUDA_IPC_TARGET, CUDA_IPC_OLD, CUDA_IPC_NEW))
    py_compile.compile(str(FLASHINFER_CUDA_IPC_TARGET), doraise=True)
    print("PATCH_COMPILE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
