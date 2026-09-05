# Third-party derived files

The files under `patches/` are modified copies of third-party sources and
remain under their original licenses. Modifications are documented as
anchored-edit scripts in `patches/generators/` and summarized in README
("The four upstream bugs").

| File | Upstream | License |
|------|----------|---------|
| patches/qwen4_exp_nvfp4.py | sglang (`srt/models/qwen4_exp.py`, lmsysorg/sglang:qwen38flashnext image) | Apache-2.0 |
| patches/qwen_sparse_attn_backend.py | sglang (`srt/layers/attention/qwen_sparse_attn_backend.py`) | Apache-2.0 |
| patches/sparse_attn.py | sglang (`srt/layers/attention/qsa/sparse_attn.py`) | Apache-2.0 |
| patches/flash_fwd.py | flash-attention (`flash_attn/cute/flash_fwd.py`, as shipped in the image) | BSD-3-Clause |

Everything else in this repository is original and MIT-licensed.
