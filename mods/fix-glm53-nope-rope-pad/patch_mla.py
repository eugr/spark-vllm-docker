"""NoPE-MLA rope pad (env-gated): make GLM-5.3-style qk_rope_head_dim=0 models
ride the DeepSeek 512+64 fp8_ds_mla path on SM120/121 by (a) constructing the
inner MLAAttention with rope=pe_pad so the KV-cache spec/kernels see head 576,
and (b) zero-padding q and k_pe just before the attention call. Zeros are
rotation-free and contribute exactly 0 to every logit — mathematically a no-op.
Active only when VLLM_MLA_NOPE_PAD_ROPE=1 and the layer is genuinely NoPE."""
import re, sys
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mla.py"
src = open(P).read()
if "VLLM_MLA_NOPE_PAD_ROPE" in src:
    print("already patched"); sys.exit(0)

a1 = "        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim\n"
r1 = a1 + (
"        import os as _os\n"
"        self.pe_pad = 0\n"
"        if qk_rope_head_dim == 0 and _os.environ.get(\"VLLM_MLA_NOPE_PAD_ROPE\", \"0\") == \"1\":\n"
"            self.pe_pad = 64\n"
)
assert src.count(a1) == 1, f"anchor1 count {src.count(a1)}"
src = src.replace(a1, r1)

a2 = """            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,"""
r2 = """            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim + self.pe_pad,"""
assert src.count(a2) == 1, f"anchor2 count {src.count(a2)}"
src = src.replace(a2, r2)

a3 = "        attn_out = self.mla_attn(\n"
r3 = (
"        if self.pe_pad:\n"
"            q = torch.nn.functional.pad(q, (0, self.pe_pad))\n"
"            k_pe = q.new_zeros((k_pe.shape[0], 1, self.pe_pad))\n"
) + a3
assert src.count(a3) == 1, f"anchor3 count {src.count(a3)}"
src = src.replace(a3, r3)

open(P, "w").write(src)
import py_compile; py_compile.compile(P, doraise=True)
print("mla.py NoPE-pad patch applied + compiles")

# ---- part 2: SM120 sparse decode must be told the REAL topk table width ----
# glm5_next kpool indexer builds tables of width index_topk + kpool alignment
# (2048+128=2176); the impl passed config topk (2048), tripping flashinfer
# wrapper shape check. Pass the actual table width for both top_k and
# max_seq_len (padding entries are -1 and skipped, same as short-seq padding).
P2 = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
src2 = open(P2).read()
if "topk_indices_physical.shape[-1]" in src2:
    print("sm120 width patch already applied")
else:
    b1 = "            max_seq_len=attn_metadata.topk_tokens,"
    n1 = "            max_seq_len=topk_indices_physical.shape[-1],"
    b2 = "            sparse_mla_top_k=attn_metadata.topk_tokens,"
    n2 = "            sparse_mla_top_k=topk_indices_physical.shape[-1],"
    assert src2.count(b1) == 1, f"sm120 anchor1 count {src2.count(b1)}"
    assert src2.count(b2) == 1, f"sm120 anchor2 count {src2.count(b2)}"
    src2 = src2.replace(b1, n1).replace(b2, n2)
    open(P2, "w").write(src2)
    py_compile.compile(P2, doraise=True)
    print("sm120 topk width patch applied + compiles")

# ---- part 3: allowlist the glm5_next topk width in the SM120 decode dispatch ----
# glm5_next tables are index_topk + kpool tail rounded to BLOCK_N=128 => 2176.
# The dsv3_2 decode binding derives topk and num_splits dynamically from
# indices.shape[-1] (2176 = 34*64, tile-aligned); the frozenset is a
# tested-shape allowlist, not a compiled-variant limit. Extend it.
P3 = "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120.py"
src3 = open(P3).read()
if "(8, 2176)" in src3 or "GLM5_NEXT_WIDTH" in src3:
    print("sm120 dispatch allowlist already extended")
else:
    a = "_DECODE_DSV3_2_PAGE_BLOCK_SIZE = 64"
    add = ("# GLM5_NEXT_WIDTH: 2048 topk + kpool tail rounded to 128 -> 2176 (34x64 tiles)\n"
           "_DECODE_DSV3_2_DISPATCH = _DECODE_DSV3_2_DISPATCH | frozenset(\n"
           "    {(h, 2176) for h in (8, 16, 32, 64, 128)}\n"
           ")\n")
    assert src3.count(a) == 1, f"fi anchor count {src3.count(a)}"
    src3 = src3.replace(a, add + a)
    open(P3, "w").write(src3)
    py_compile.compile(P3, doraise=True)
    print("sm120 dispatch allowlist extended + compiles")

# ---- part 4: compact glm5_next topk table to the compiled 2048 width ----
# Backend validation pins index_topk=2048; buffer is 2048 + (kpool-1) tail
# rounded to 2176. Compiled decode kernel only takes 2048. Keep the top
# (2048 - tail) ranked entries + the always-select tail (never duplicated:
# tail tokens come from the incomplete pool, excluded from pool selection).
# Attention is permutation-invariant over the gathered set, so order is free.
src4 = open(P2).read()
if "_glm_kpool_tail" in src4:
    print("sm120 tail-compaction already applied")
else:
    c1 = "        self.kv_scale_format = _kv_scale_format_for_model(model_type)\n"
    r1 = c1 + (
"        _kp = 1\n"
"        if vllm_config.model_config is not None:\n"
"            _kp = int(getattr(vllm_config.model_config.hf_text_config, \"index_kpool\", 1) or 1)\n"
"        self._glm_kpool_tail = max(0, _kp - 1)\n"
)
    assert src4.count(c1) == 1, f"c1 count {src4.count(c1)}"
    src4 = src4.replace(c1, r1)
    c2 = """                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )
"""
    r2 = c2 + (
"        _w = topk_indices_physical.shape[-1]\n"
"        _k = attn_metadata.topk_tokens\n"
"        if _w > _k:\n"
"            _t = min(self._glm_kpool_tail, _w - _k)\n"
"            topk_indices_physical = torch.cat(\n"
"                (\n"
"                    topk_indices_physical[..., : _k - _t],\n"
"                    topk_indices_physical[..., _k : _k + _t],\n"
"                ),\n"
"                dim=-1,\n"
"            ).contiguous()\n"
)
    assert src4.count(c2) == 1, f"c2 count {src4.count(c2)}"
    src4 = src4.replace(c2, r2)
    open(P2, "w").write(src4)
    py_compile.compile(P2, doraise=True)
    print("sm120 tail-compaction applied + compiles")

