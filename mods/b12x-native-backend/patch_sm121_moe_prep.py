#!/usr/bin/env python3
"""sm_121 (GB10) b12x MoE weight-prep wiring fix.

The b12x MoE overlay (fused_moe/b12x_moe.py: B12xExperts) implements
process_weights_after_loading() to build its FP4 'prepared experts', and
workspace_shapes() hard-requires that prep. But B12xExperts is a modular
fused-experts impl, and pristine v0.26.0's Mxfp4MoEMethod.process_weights_after_loading
never delegates prep down to self.moe_kernel.fused_experts -- the base
FusedMoEExperts.process_weights_after_loading is a no-op. So B12xExperts' prep
never runs, and the first forward (profile_run) dies in workspace_shapes with
'B12X MoE workspace planning requires prepared weights'. The upstream fork's
mxfp4 method delegated this; the overlay didn't port that wiring.

This appends a wrapper to the installed quantization/mxfp4.py so the mxfp4 MoE
methods, after their own process_weights_after_loading, also call
moe_kernel.fused_experts.process_weights_after_loading(layer). Gated to
capability major==12; idempotent; safe for non-b12x experts (base method is a
no-op).
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(
    '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/mxfp4.py'
)
SENTINEL = '_b12x_delegate_experts_prep'

APPEND = '''

# ---- sm_121 (GB10) b12x MoE prep delegation (appended by b12x-native-backend mod) ----
def _b12x_delegate_experts_prep(_cls):
    if getattr(_cls, '_b12x_pwal_wrapped', False):
        return
    _orig_pwal = _cls.process_weights_after_loading

    def process_weights_after_loading(self, layer):
        _orig_pwal(self, layer)
        try:
            from vllm.platforms import current_platform as _cp
            _cap = _cp.get_device_capability()
        except Exception:
            _cap = None
        if _cap is None or _cap.major != 12:
            return
        _k = getattr(self, 'moe_kernel', None)
        _fe = getattr(_k, 'fused_experts', None)
        if _fe is not None and hasattr(_fe, 'process_weights_after_loading'):
            _fe.process_weights_after_loading(layer)

    _cls.process_weights_after_loading = process_weights_after_loading
    _cls._b12x_pwal_wrapped = True


for _c in (Mxfp4MoEMethod, GptOssMxfp4MoEMethod):
    _b12x_delegate_experts_prep(_c)
'''


def main() -> int:
    if not TARGET.is_file():
        print(f'FAIL sm121-moe-prep: {TARGET} not found')
        return 1
    text = TARGET.read_text()
    if SENTINEL in text:
        print('SKIP sm121-moe-prep: already applied')
        return 0
    for name in ('class Mxfp4MoEMethod', 'class GptOssMxfp4MoEMethod', 'def process_weights_after_loading'):
        if name not in text:
            print(f'FAIL sm121-moe-prep: expected {name!r} not in mxfp4.py')
            return 1
    TARGET.write_text(text + APPEND)
    import py_compile
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as e:
        print(f'COMPILE_ERROR sm121-moe-prep: {e}')
        return 1
    pc = TARGET.parent / '__pycache__'
    if pc.is_dir():
        for f in pc.glob('mxfp4*.pyc'):
            f.unlink()
    print('OK sm121-moe-prep: experts-prep delegation appended')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
