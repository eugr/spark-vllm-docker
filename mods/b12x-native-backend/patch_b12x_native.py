#!/usr/bin/env python3
"""Runtime patches: native B12X_MLA_SPARSE attention + b12x MoE + b12x linear
backends, ported from local-inference-lab/vllm commit 3003860 (the fork
built into ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer) onto
pristine vllm/vllm-openai:v0.26.0.

That fork's files call SparkInfer (`sparkinfer.attention.*`,
`sparkinfer.moe.*`, `sparkinfer.gemm.*`) — a same-org, same-API sibling
package to the public https://github.com/local-inference-lab/b12x. The
overlay files here have every `sparkinfer.` import rewritten to `b12x.`
(mechanical rename; verified the module layout and call signatures match:
b12x.moe.fused_moe/ep_moe expose the same Caps/plan/run/prepare_expert_map,
b12x.gemm.blockscaled/block_fp8_linear/tensor_fp8_linear and
b12x._lib.intrinsics exist at the same paths). Unlike the vllm-starter
DSpark overlay's b12x_mxfp4_moe.py (which imports the nonexistent private
`b12x.integration.tp_moe`), none of these files touch `b12x.integration` —
they call the real public plan/bind/run primitives directly.

1. Overlay 9 new + 2 replaced files from the fork (see OVERLAY_FILES).
2. registry.py: add B12X_MLA_SPARSE + B12X_ATTN to AttentionBackendEnum.
3. config/kernel.py: add "b12x" to MoEBackend and LinearBackend Literals
   (pristine already has "flashinfer_b12x" — a different, FlashInfer-native
   kernel; "b12x" is missing and required for --moe-backend/--linear-backend
   b12x to pass Pydantic's Literal validation at all).
"""

from __future__ import annotations

from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
OVERLAY = Path(__file__).resolve().parent / "overlay/vllm"


def _patch_text(target: Path, old: str, new: str, label: str) -> str:
    if not target.is_file():
        return f"SKIP  {label}: not found"
    text = target.read_text()
    if old not in text:
        if new in text:
            return f"SKIP  {label}: already applied"
        return f"FAIL  {label}: old text not found"
    n = text.count(old)
    if n > 1:
        return f"FAIL  {label}: {n} ambiguous matches"
    target.write_text(text.replace(old, new, 1))
    return f"OK    {label}"


def _compile(target: Path):
    import py_compile
    try:
        py_compile.compile(str(target), doraise=True)
    except py_compile.PyCompileError as e:
        raise SystemExit(f"COMPILE_ERROR {target.name}: {e}")


OVERLAY_FILES = [
    "model_executor/kernels/linear/mxfp4/b12x.py",
    "model_executor/kernels/linear/nvfp4/b12x.py",
    "model_executor/kernels/linear/mxfp8/b12x.py",
    "model_executor/kernels/linear/mxfp8/__init__.py",
    "model_executor/kernels/linear/scaled_mm/b12x.py",
    "model_executor/kernels/linear/scaled_mm/b12x_tensor.py",
    "model_executor/layers/mla_cache_format.py",
    "model_executor/layers/fused_moe/b12x_moe.py",
    "model_executor/layers/fused_moe/b12x_ep_moe.py",
    "v1/attention/backends/mla/b12x_mla_sparse.py",
    "v1/attention/backends/b12x_attn.py",
    "models/deepseek_v4/nvidia/b12x.py",
    # These two are wholesale-replaced (not hand-patched): the fork's real
    # B12X wiring touches oracle/mxfp4.py at 8+ sites and linear/__init__.py
    # at 6+ sites (enum, kernel_cls dispatch, backend map, priority gating,
    # FP8-GEMM special-casing...) — far beyond what a couple of _patch_text
    # calls can safely express. Diff against pristine is small (+115 /
    # +56 lines) and purely additive, so a full overlay is lower-risk than
    # hand-transcribing every site.
    "model_executor/layers/fused_moe/oracle/mxfp4.py",
    "model_executor/kernels/linear/__init__.py",
]


def _overlay_b12x_native() -> list[str]:
    r = []
    mounted = OVERLAY.is_dir()
    for rel in OVERLAY_FILES:
        src = OVERLAY / rel
        dst = VLLM / rel
        if not src.is_file():
            r.append(f"SKIP  overlay:{rel}: overlay src not found")
            continue
        if not mounted:
            r.append(f"SKIP  overlay:{rel}: overlay not mounted")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())
        r.append(f"OK    overlay:{rel}")
        _compile(dst)
    return r


def _registry_attention_backends() -> list[str]:
    t = VLLM / "v1/attention/backends/registry.py"
    r = []
    r.append(_patch_text(t,
        '    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"',
        '    # Opt-in b12x unified sparse-MLA backend (SM120). Not in the platform\n'
        '    # auto-selection priority list; select it explicitly via\n'
        '    # VLLM_ATTENTION_BACKEND=B12X_MLA_SPARSE.\n'
        '    B12X_MLA_SPARSE = (\n'
        '        "vllm.v1.attention.backends.mla.b12x_mla_sparse.B12xMLASparseBackend"\n'
        '    )\n'
        '    B12X_ATTN = "vllm.v1.attention.backends.b12x_attn.B12XPagedAttentionBackend"\n'
        '    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"',
        "b12x-registry:enum"))
    _compile(t)
    return r


def _kernel_config_literals() -> list[str]:
    t = VLLM / "config/kernel.py"
    r = []
    r.append(_patch_text(t,
        'MoEBackend = Literal[\n'
        '    "auto",\n'
        '    "triton",\n'
        '    "deep_gemm",\n'
        '    "deep_gemm_mega_moe",\n'
        '    "cutlass",\n',
        'MoEBackend = Literal[\n'
        '    "auto",\n'
        '    "triton",\n'
        '    "deep_gemm",\n'
        '    "deep_gemm_mega_moe",\n'
        '    "b12x",\n'
        '    "cutlass",\n',
        "b12x-kernel-config:moe-literal"))
    r.append(_patch_text(t,
        'LinearBackend = Literal[\n'
        '    "auto",\n'
        '    "cutlass",\n',
        'LinearBackend = Literal[\n'
        '    "auto",\n'
        '    "b12x",\n'
        '    "cutlass",\n',
        "b12x-kernel-config:linear-literal"))
    _compile(t)
    return r


def _envs_b12x_flags() -> list[str]:
    """Register the 3 B12X env flags the overlay files read via
    vllm.envs.VLLM_USE_B12X_FP8_GEMM / VLLM_USE_B12X_MOE /
    VLLM_B12X_MOE_FORCE_MODELOPT_PREP. envs.py's __getattr__ only resolves
    names present in its `environment_variables` dict, so referencing an
    unregistered name raises AttributeError at first access (not import
    time) — this is what crashed model loading, not import."""
    t = VLLM / "envs.py"
    r = []
    r.append(_patch_text(t,
        '    VLLM_USE_DEEP_GEMM: bool = True\n',
        '    VLLM_USE_DEEP_GEMM: bool = True\n'
        '    VLLM_USE_B12X_FP8_GEMM: bool = False\n'
        '    VLLM_USE_B12X_MOE: bool = False\n'
        '    VLLM_B12X_MOE_FORCE_MODELOPT_PREP: bool = False\n',
        "b12x-envs:type-checking-block"))
    r.append(_patch_text(t,
        '    "VLLM_USE_DEEP_GEMM": lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM", "1"))),\n',
        '    "VLLM_USE_DEEP_GEMM": lambda: bool(int(os.getenv("VLLM_USE_DEEP_GEMM", "1"))),\n'
        '    "VLLM_USE_B12X_FP8_GEMM": lambda: bool(\n'
        '        int(os.getenv("VLLM_USE_B12X_FP8_GEMM", "0"))\n'
        '    ),\n'
        '    "VLLM_USE_B12X_MOE": lambda: bool(int(os.getenv("VLLM_USE_B12X_MOE", "0"))),\n'
        '    "VLLM_B12X_MOE_FORCE_MODELOPT_PREP": lambda: bool(\n'
        '        int(os.getenv("VLLM_B12X_MOE_FORCE_MODELOPT_PREP", "0"))\n'
        '    ),\n',
        "b12x-envs:dict-entries"))
    _compile(t)
    return r


def main() -> int:
    results = []

    patches = [
        _overlay_b12x_native,
        _registry_attention_backends,
        _kernel_config_literals,
        _envs_b12x_flags,
    ]

    for fn in patches:
        for result in fn():
            print(result, flush=True)
            results.append(result)

    ok = sum(1 for r in results if r.startswith("OK"))
    fail = sum(1 for r in results if r.startswith("FAIL"))
    print(f"\nPATCHES ok={ok} fail={fail} total={len(results)}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
