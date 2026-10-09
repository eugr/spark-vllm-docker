#!/bin/bash
# fix-dspark-sm120-topk — let DSpark serve on SM120 (DGX Spark / GB10, sm_121 → sm120).
#
# Problem: DeepSeek-V4-Flash-DSpark's draft runs a sparse-MLA decode with topk=256, but
# FlashInfer's SM120 `decode_dsv4` kernel only has compiled buckets {128, 512, 1024}
# (see _DECODE_DSV4_DISPATCH in flashinfer/mla/_sparse_mla_sm120.py). topk=256 sits in
# the gap → the dispatcher falls through to the paged kernel, which asserts
# `num_tokens > 64` and crashes during sparse-MLA warmup:
#   tvm.error.InternalError: Check failed: num_tokens > 64 (5 vs. 64) :
#   Decode (num_tokens <= 64) must go through ... sparse_mla_sm120_decode_dsv4
# (The main model uses topk=128 and is unaffected; only the DSpark draft asks for 256.)
#
# Fix: round an unsupported draft topk DOWN to the nearest supported bucket (256 -> 128)
# right before dispatch — slice the indices to that width and clamp topk_length.
#
# Why this is correctness-safe: the DSpark draft is a SPECULATOR. Every token it proposes
# is verified by the full target model before emission, so reshaping the draft's own
# attention can only change accept-length (speed), never the emitted output. Rounding down
# drops the lowest-ranked sparse candidates from the draft's attention only.
#
# If draft accept-length suffers, replace the "round down" with "pad up" (256 -> 512 with
# -1 sentinels + real topk_length) to keep all candidates — lossless but needs matching
# scratch (num_splits) sizing.
set -e
F="$(python3 -c 'import os,flashinfer; print(os.path.join(os.path.dirname(flashinfer.__file__),"mla","_sparse_mla_sm120.py"))' 2>/dev/null)"
[ -f "$F" ] || { echo "[fix-dspark-sm120-topk] ERROR: flashinfer _sparse_mla_sm120.py not found ($F)" >&2; exit 1; }

python3 - "$F" <<'PY'
import sys
f = sys.argv[1]
s = open(f).read()
if "fix-dspark-sm120-topk" in s:
    print("[fix-dspark-sm120-topk] already applied"); raise SystemExit(0)
anchor = "        extra_topk = int(extra_indices.size(-1)) if extra_indices is not None else 0\n"
if anchor not in s:
    raise SystemExit("[fix-dspark-sm120-topk] ERROR: anchor not found — flashinfer version drift")
inject = (
    "        # fix-dspark-sm120-topk: route an unsupported topk (e.g. the DSpark draft's\n"
    "        # 256) to the nearest supported decode_dsv4 bucket. Speculator draft only —\n"
    "        # the target verifies all proposals, so this affects accept-length, not output.\n"
    "        _DSV4_TOPK_OK = (128, 512, 1024)\n"
    "        if model_type == _MODEL_TYPE_DSV4 and topk not in _DSV4_TOPK_OK:\n"
    "            _lo = [b for b in _DSV4_TOPK_OK if b <= topk]\n"
    "            _tgt = max(_lo) if _lo else min(_DSV4_TOPK_OK)\n"
    "            indices = indices[..., :_tgt].contiguous()\n"
    "            if topk_length is not None:\n"
    "                topk_length = topk_length.clamp(max=_tgt)\n"
    "            topk = _tgt\n"
)
open(f, "w").write(s.replace(anchor, anchor + inject, 1))
print("[fix-dspark-sm120-topk] applied: topk rounding installed before decode_dsv4 dispatch")
PY
python3 -c "import ast; ast.parse(open('$F').read()); print('[fix-dspark-sm120-topk] verified (AST valid)')"
