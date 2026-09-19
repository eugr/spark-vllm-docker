# qwen3.8-flash-next-180b-hashk

Bind-in patches for [Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark](https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark),
applied the same way other mods in this repo patch vLLM: `run.sh` copies
full-file replacements over the installed package inside the already-launched
container and backs up each original to `<path>.orig`.

Requires the `lmsysorg/sglang:qwen38flashnext` image (this is an **sglang**
deployment, not vLLM — set `container: lmsysorg/sglang:qwen38flashnext` in the
recipe, or `-t lmsysorg/sglang:qwen38flashnext` on the CLI).

## What it patches

Four upstream sglang / flash-attention bugs that only surface in this
single-Spark, long-context, fallback-heavy configuration (the reference
deployment splits the model across two Sparks with tensor parallelism and
never hits these paths):

| File | Bug |
|---|---|
| `flash_fwd.py` | TMA-O enabled for varlen; ragged epilogue is rank-broken |
| `qwen_sparse_attn_backend.py` | flashinfer trtllm-gen decode kernels are SM100-only, emit garbage on SM121 |
| `sparse_attn.py` | long-prefill sparse kernel feeds fp8-loaded K into `tl.dot` |
| `qwen4_exp_nvfp4.py` → `sglang/srt/models/qwen4_exp.py` | model definition wired for the HashK/NVFP4-packed PLE table |

## HashK artifact (required, not built by this mod)

The default `PLE_MODE=hashk` path needs a 12.8 GB `ple_hashk_R4.pt` artifact
built once from the RadixArk NVFP4 checkpoint (~6 min on GPU, streams the
checkpoint's PLE shards):

```bash
docker run --rm --gpus all \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  -v /path/to/store/artifact:/out --entrypoint python3 \
  lmsysorg/sglang:qwen38flashnext /out/tools/build_hashk_ple.py
```

(`tools/build_hashk_ple.py` comes from the upstream repo, not this one —
clone it separately to run the build.) Then mount the resulting file into the
container at launch:

```bash
./run-recipe.sh qwen3.8-flash-next-180b-hashk --solo \
  -v /path/to/store/artifact/ple_hashk_R4.pt:/patches/ple_hashk_R4.pt
```

## License

`qwen4_exp_nvfp4.py`, `flash_fwd.py`, `qwen_sparse_attn_backend.py`, and
`sparse_attn.py` are modified copies of sglang (Apache-2.0) and
flash-attention (BSD-3) sources, carried over from the upstream repo's
`patches/` directory. See `NOTICE.md` in this directory.
