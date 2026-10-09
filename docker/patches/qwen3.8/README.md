# Regular-lane Qwen3.8 patches

These frozen source patches are applied before wheel compilation. The `series`
files define application order. Builds do not fetch moving PR heads for this
stack; refresh the files deliberately when updating it.

The vLLM series follows the existing preset selection: ordinary upstream source
builds include it; custom refs, local sources, and explicit PR selections skip it
unless `--apply-preset-vllm-prs` is requested. The FlashInfer series is enabled
for the regular upstream lane with the default FlashInfer ref and no explicit
FlashInfer PRs. Custom FlashInfer refs/PRs skip it.

Both series also check the source repository before applying anything. Neither
can apply to `local-inference-lab/vllm` or `local-inference-lab/flashinfer`, even
if its enable flag is accidentally set. The B12X build script disables both.
Existing explicit `--apply-vllm-pr` and `--apply-flashinfer-pr` behavior remains
available independently.

## Included changes

The vLLM order is:

1. [#59958](https://github.com/vllm-project/vllm/pull/59958), including #56273 and #58815: NVFP4 PLE tables.
2. [#59973](https://github.com/vllm-project/vllm/pull/59973): quantized draft heads.
3. [#59740](https://github.com/vllm-project/vllm/pull/59740) and [#60226](https://github.com/vllm-project/vllm/pull/60226): sparse draft vocabulary, including quantized heads.
4. [#60068](https://github.com/vllm-project/vllm/pull/60068): adaptive draft depth.
5. [#54912](https://github.com/vllm-project/vllm/pull/54912): QSA ring sizing.
6. [#50897](https://github.com/vllm-project/vllm/pull/50897): lookahead-aware prefix hashing.
7. [#54614](https://github.com/vllm-project/vllm/pull/54614): dynamic NVFP4 linear kernels.
8. [#54617](https://github.com/vllm-project/vllm/pull/54617): dynamic NVFP4 MoE.
9. [#60252](https://github.com/vllm-project/vllm/pull/60252): DFlash2 draft cache block sizing.
10. [#59021](https://github.com/vllm-project/vllm/pull/59021) and [#59025](https://github.com/vllm-project/vllm/pull/59025): separate unquantized linear selection and cuDNN BF16 GEMMs.
11. The supplied 32,768-token Flash Next draft vocabulary.
12. [#60646](https://github.com/vllm-project/vllm/pull/60646): CSF-compressed NVFP4 scale loading.
13. Wheel package-data inclusion for the draft vocabulary.

[#55601](https://github.com/vllm-project/vllm/pull/55601) is already in the
validated upstream base. [#59244](https://github.com/vllm-project/vllm/pull/59244)
modified a BF16-only warmup workaround that upstream
[#60434](https://github.com/vllm-project/vllm/pull/60434) subsequently removed.
The series retains the current full-model autotune path, including cuDNN, and
does not restore the obsolete 32-token cap or BF16-only pass.

The FlashInfer order is
[#5863](https://github.com/flashinfer-ai/flashinfer/pull/5863) (canonical NVFP4
W4A4 MoE), then [#5943](https://github.com/flashinfer-ai/flashinfer/pull/5943)
(SM12x CuTe DSL MXFP8 GEMMs). #5943 includes head
`b3568f3498fd562ec621e972de85d153abdc823e`.

## Integration and refresh

The October 8, 2026 series was replayed against upstream vLLM
`ab905a885dfbfc60a2c02286cc9c608c93884de3` and FlashInfer
`fa7c741e0af4493a73352bf240c1bbc802cca288`. These identify the validation bases;
normal builds still follow their selected refs.

On October 9, the #54614 test hunk was refreshed to preserve the quantized-input
shape test added by upstream [#59612](https://github.com/vllm-project/vllm/pull/59612).
Both tests had been inserted at the same location, which caused the October 9
nightly to stop on a test-file conflict. The complete vLLM series was then
replayed successfully against the failed nightly's base
`59ffd73e8597d860730f4a8c4f778f27c75a0dfa` and current main
`75e3b17c055c609f02c64cd4ad3f8692b8170d8e`. The runtime patch content and lane
selection are unchanged. A CPU-only regression in
`tests/test_regular_patch_series.py` checks that both tests survive application.

The supplied conflict resolutions are retained, with adaptations for newer
upstream Mamba cache-boundary and KV-transfer fixes (#60533 and #59197).
Lookahead hashing disables the EAGLE block drop while retaining the resolved
cache-hit alignment and the one-token replay boundary. Both sets of tests/docs
are preserved. Dynamic linear warmup keeps upstream's `round_up=True` and
full-model BF16 tuning. FlashInfer's benchmark retains both cuDNN and SM12x MoE.
The existing SWA workaround recognizes #60252's divisor-aware replacement and
leaves it intact; its behavior on the B12X source is unchanged.

To refresh, compare the referenced PR revisions, reapply the complete sequence
to the intended upstream bases, resolve conflicts, and export the updated
per-step diffs. Remove changes already provided by the selected base. Preserve
both sides of test/doc conflicts; the series applier fails on unresolved
conflicts instead of choosing one side. Recheck source application, wheel
packaging, and the regular/B12X build-selection tests.

No additional wheel-cache metadata is introduced. Rebuild and publish both
regular FlashInfer and vLLM wheels when introducing or updating this series.

## Runtime options

The series supplies capabilities; recipe choices remain explicit. In particular,
`--prefix-match-unit 48`, adaptive draft depth, and dynamic kernel backends are
not enabled globally.

For an MTP drafter using Model Runner V2, set `draft_token_map` to the installed
vocabulary path and `draft_token_map_quantization` to `nvfp4` in
`--speculative-config`. Resolve the path inside the built image with:

```bash
python3 -c 'from importlib.resources import files; print(files("vllm") / "model_executor/layers/draft_vocab_lists/qwen38_flash_next_32k.json")'
```

The JSON is included in the exported vLLM wheel, so the runner does not need the
builder's source checkout.
