# glm-5.2

Lets the Triton MLA decode kernel run on DGX Spark (GB10, sm_121) for
GLM-5.2 style models.

## The problem

GLM-5.2 uses MLA attention with a 576-wide KV layout (kv_lora_rank 512 +
rope 64). vLLM's Triton decode kernel only drops to `num_stages=1` when
`BLOCK_DMODEL >= 1024`. At GLM-5.2's `BLOCK_DMODEL=512` it keeps
`num_stages=2`, which needs 102,400 bytes of shared memory per SM. The GB10
has 101,376. The kernel dies with `OutOfResources` at decode time.

## The fix

One line in `vllm/v1/attention/ops/triton_decode_attention.py`:

```
elif not is_hip_ and BLOCK_DMODEL >= 1024:   ->   >= 512
```

`run.sh` makes this edit with an exact-string replacement instead of a git
patch, so it keeps working as vLLM main drifts. It is idempotent and refuses
to patch if the surrounding code has changed in a way it doesn't recognize.

## Notes

With the FLASHINFER_MLA_SPARSE_SM120 attention backend (what vLLM selects
for GLM-5.2 on GB10 as of July 2026), this Triton kernel is not on the hot
path. The fix is kept because other backend selections do route through it,
and it costs nothing. The fix logic originally comes from the bird/GLM-spark
repo; by vLLM v0.23 everything except the threshold change had already
landed upstream.
