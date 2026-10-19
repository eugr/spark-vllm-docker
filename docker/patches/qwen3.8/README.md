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

1. [#59958](https://github.com/vllm-project/vllm/pull/59958) and #58815: host-file-gather PLE tables, using #56273's NVFP4 support already in main.
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
14. The local Flash Next FP4 mod from `mods/draft/flash-next-fp4`:
    opt-in resident CSF expert scales, ModelOpt dispatch, nested scale loading,
    weight-transfer helpers, and post-load host-heap trimming. Its vocabulary
    path is packaged alongside the identical existing vocabulary.
15. `17-flash-next-b12x-loader.patch`: opt-in `--load-format b12x` for Flash
    Next and its embedded MTP draft on Spark, using FlashInfer's checkpoint
    reader and the weight-transfer helpers supplied by patch 16.

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
`15c5c8d21388c118558c8370299204aa4b110ced`.

## Integration and refresh

On October 14, the series was refreshed against vLLM
`355553ac22dca4080929ea8f88a446f15df1f3ba` and FlashInfer
`2895577720741ddfedff118202c89584996b8875`. The complete vLLM series also
replays on the failed nightly base `3f466c852cfd3882820af8dd7963ac775174bce4`.

[#56273](https://github.com/vllm-project/vllm/pull/56273) merged as
`4e0681ed5654c1d8afb3c104d4381afd08529de8`. Patch 01 now contains only the
remaining file-gather integration; merged NVFP4 definitions and tests come
from main. Its header records the merged dependency separately. Existing
`host_file_gather` configuration, sequence-parallel behavior, and loader fixes
are retained. #59958's newer `disk_offload` rewrite is not adopted in this
compatibility fix; its recorded host-file-gather snapshot remains unchanged.

Two other PR snapshots were updated:

- #59973 at `c8ce09384e` selects the NVFP4 draft-head kernel through
  `init_nvfp4_linear_kernel`, retaining the platform/backend selection policy.
- #50897 at `7f7139246a` keeps legacy EAGLE dropping with HiSparse, updates
  Mooncake lookahead tails, and moves the shared gating helper to
  `kv_cache_utils.py`. Its conflicts with newer Mooncake capacity-only workers,
  sleep-mode declarations, offload metrics, and Mamba checkpoint code are
  resolved while preserving those upstream features.

Validation: all 16 vLLM patches and all 17 subsequent Docker source patchers
pass on both vLLM bases. The 2 FlashInfer patches and loader include-order fix
pass on the recorded FlashInfer base. All 151 changed Python files compile;
both draft vocabularies and wheel package-data rules remain intact. The 16
series tests, 28 SWA tests, 100 build-selection checks, 28 CPU loader tests,
nine PLE constructor cases, and three Mooncake integration checks pass.
The new regression applies the decoder integration to the merged NVFP4 code
and checks device, pinned, and file-gather dispatch. Existing upstream PLE
tests remain unchanged. The native-reader test is skipped on CPU; GPU builds
and inference remain for CI. Validation applies to the recorded commits.

Later on October 13, patch 16 was rebased on vLLM
`624a5ddafd96df23fe2b9f6d4b8f52a08b9fe9ea` after
[#59604](https://github.com/vllm-project/vllm/pull/59604) added IQ2/Q8 ModelOpt
dispatch at the resident-CSF insertion point. The reported `modelopt.py`
conflict reproduces on this base. The refreshed patch retains upstream block
quantization dispatch before the opt-in resident-CSF NVFP4 path, preserving
both features. No upstream PR revisions or series selection changed.

The full 16-patch vLLM series replays cleanly and reproduces the prepared source
tree. All 17 Docker source patchers pass, all 129 changed Python files compile,
and both draft vocabularies remain packaged. The 15 series tests, 100
build-selection checks, and 28 CPU loader tests pass; the native-reader test
is skipped. The new regression fails with the previous patch and covers
IQ2/Q8 linear, embedding, and expert dispatch, plus resident-CSF opt-in and
default NVFP4 dispatch. GPU compilation and inference remain for CI.

All 18 patches were refreshed on October 13, 2026 against upstream vLLM
`12b005eefa06f97eeebfae3e14bcac5fa3b81197` and FlashInfer
`74ca44b318a3be013649baabca698099d0862e3c`. Eight referenced PR heads changed:
vLLM #56273, #59973, #59740, #60226, #60068, #50897, and #54614, plus
FlashInfer #5863. All 17 referenced PRs remain open. The patch headers record
their current revisions; existing local integration fixes are retained.
#56273 updates the pinned PLE test for the current prefetch API and trims its
test coverage to loader/lookup validation. The other head updates merge main;
their conflict resolutions are incorporated without duplicating tests or hooks.

The refresh resolves two new source conflicts:

- [#61183](https://github.com/vllm-project/vllm/pull/61183) moved KV-cache
  planning out of `kv_cache_utils.py`, overlapping #50897's hash helpers.
  The helpers remain in that file without restoring the moved planning code.
- [#61082](https://github.com/vllm-project/vllm/pull/61082) added the
  `monokernel` setting at #54617's insertion point. Both that setting and the
  dynamic NVFP4 MoE settings are preserved.

Validation: the complete 16-patch vLLM and 2-patch FlashInfer series replay
cleanly and reproduce the prepared source trees. All 17 subsequent vLLM source
patchers and the FlashInfer loader patch pass; all 153 changed Python files
compile. The 14 series tests, 28 SWA tests, 100 build-selection checks, 28 CPU
loader tests, and nine PLE constructor cases pass. The new series regression
reproduces the old KV-cache patch failure and checks lookahead metadata and
the non-lookahead fallback after application. Both draft vocabularies and their
wheel package-data rules remain intact. The native-reader test is skipped on
CPU; image compilation and GPU inference remain for CI. These results apply
to the recorded commits; builds following moving main can need another refresh.

The host-file-gather implementation in `01-pr59958.patch` now implements the
current PLE base class's `finalize_prefetch` method and single-argument
`start_prefetch` signature. File gather keeps synchronous row staging and
rejects prefetch finalization, as the device backend does. This fixes the
abstract-class construction error before weight loading with both `safetensors`
and `b12x`. Rebuild the regular vLLM wheel and runner to include the correction.
The dependency-free series regression reproduces the original exception, and
the patch includes construction checks for BF16, FP8, and NVFP4 with dummy,
safetensors, and b12x load configurations. All nine constructor cases pass with
CPU tensors and mocked platform services. The full series and all 17 Docker
source patchers replay on the base below; 13 series tests and 28 CPU loader
tests pass. Full checkpoint loading and Spark inference remain unverified.

`17-flash-next-b12x-loader.patch` follows patch 16, bringing the series to 16
vLLM patches and 2 FlashInfer patches. It adapts the loader and transfer hooks
from `local-inference-lab/vllm` commit
`3a947437cf1eb734299de91370c9b37fcdbcc42b` to the regular source interfaces.
The patch contains its implementation and focused tests; FlashInfer remains
on the regular upstream lane. Rebuild the regular vLLM wheel and runner to
include it, using a current compatible regular FlashInfer wheel.

The included tests exercise actual source definitions with CPU tensors without
importing unrelated inference kernels. After applying the series to a source
checkout, run them with Torch, safetensors, NumPy, and pytest installed:

```bash
python -m pytest --noconftest --import-mode=importlib \
  <vllm-source>/tests/model_executor/model_loader/test_flash_next_b12x_loader.py
```

The native-reader case additionally requires Spark CUDA and FlashInfer, and
skips on a CPU-only test environment. Full model loading, output parity, and
startup memory/time comparisons still require GPU validation.

Validation on `fe341a53dda00472c281a2d1f4bd34a74e7eae6e`: the complete
16-patch vLLM series applies cleanly and reproduces the prepared source.
All 17 subsequent Docker source patchers pass, and all 129 changed Python
files parse. The 28 CPU loader checks, 12 series tests, and 100 build-selection
checks pass; the native-reader test is skipped in the CPU environment.
Lint and formatting checks pass for all 14 files in patch 17. No image build
or live inference was performed for this addition.

`16-flash-next-fp4.patch` integrates the supplied runtime mod after the October
12 refresh, bringing the series to 15 vLLM patches and 2 FlashInfer patches.
Its header records the original patch path and SHA-256. All 13 source changes
are retained, including the mod's `setup.py` package-data rule, so the exported
wheel includes them. The runtime mod is unnecessary on images built with this
series. The original mod remains available for the older image it documents.
The source mod directory is ignored by Git; the vendored patch is self-contained
and does not require that directory at build time.

The extended 15-patch vLLM series replays cleanly on
`fe341a53dda00472c281a2d1f4bd34a74e7eae6e`. All 17 subsequent source patchers
pass, and all 122 changed Python files parse. Validation also covers both
vocabulary paths and their wheel package-data rules, 12 series tests (including
default CSF decoding versus resident-scale pass-through), 14 heap-cleanup tests,
and 100 build-selection checks. The original mod's runtime changes remain
present after the Docker patchers. GPU build and inference checks remain for CI.

All 16 patches were refreshed on October 12, 2026 against upstream vLLM
`fe341a53dda00472c281a2d1f4bd34a74e7eae6e` and FlashInfer
`b77de25d433ff5082247b8e306c3aa54857e3ed7`. All 17 referenced PRs remain open;
two revisions changed since the previous refresh:

- #56273 at `5f70b31388895bfd8ced626b9ae89dc00b61f410` validates NVFP4 global
  scales like FP8, supports loaders that copy state without calling
  `load_weights`, and rejects unsupported PLE quantization formats. File-gather
  binding and the shared byte-scale decoder are retained.
- #50897 at `ff08350f20b036b36b3a033001d6f9b4b5448e75` adds the
  `disable_lookahead_block_hashes` opt-out and consolidates EAGLE block dropping
  through `drop_eagle_group_ids`. The Mooncake partial-tail and finished-save
  integrations are retained alongside the updated upstream tests.

The refresh also resolves conflicts introduced by
[#56322](https://github.com/vllm-project/vllm/pull/56322)'s Qwen sequence
parallelism and [#60692](https://github.com/vllm-project/vllm/pull/60692)'s draft
decode-metadata refactor. Pinned and file-gather PLE lookups retain full token
rows until sharding; file-gather rows already contain all vocabulary owners and
must not be summed again by reduce-scatter. The sparse-draft MTP imports are
preserved, and confidence stopping disables fused decoding through the new
`supports_decode_metadata_update` API.

Validation: both exported series apply without conflicts and reproduce the
prepared source trees. All 17 subsequent vLLM source patchers and the FlashInfer
loader patch pass; all 138 changed Python files parse. The 11 series tests,
28 SWA tests, and 100 build-selection checks pass. New CPU regressions cover PLE
storage/sharding combinations and confidence stopping with the new metadata
API; Mooncake save-limit and block-reference checks also pass. The vocabulary
and wheel package-data rule remain intact. GPU compilation and inference remain
for CI. These results apply to the recorded commits; later main revisions may
require another refresh.

On October 11, #59973 and #50897 were rebased after two upstream changes
landed between validation and the nightly build. Their PR revisions are
unchanged; the integration diffs now preserve:

- [#60937](https://github.com/vllm-project/vllm/pull/60937)'s native MRV2 weight
  reload implementation and cache resets, with quantized draft heads refreshed
  through `reset_lora_state` after target reloads and weight transfers.
- [#61012](https://github.com/vllm-project/vllm/pull/61012)'s shared
  `replay_boundaries` helper. Core caching and Mooncake both disable EAGLE
  back-off when lookahead hashes already prove the boundary, and retain the
  upstream sliding-window tail fix when EAGLE block dropping applies.

The complete 14-patch vLLM series and all 17 subsequent source patchers pass on
the failed nightly base `2339bd39d661965b821bfbe652db8f536e0b3dcd` and current main
`b99fdbe1f896abd7f4b133685a37397c3427f7d5`. All 117 changed Python files parse on
both bases, and vocabulary packaging is intact. The 9 series tests (including
CPU regressions for native reload and shared replay boundaries), 28 SWA tests,
and 100 build-selection checks pass. GPU compilation and inference remain for
nightly CI. Validation applies to these commits; builds that track moving main
can encounter later upstream conflicts.

All 16 patches were refreshed on October 10, 2026 against upstream vLLM
`3c062d391d15cfa41d3f321cad24f3030c892a97` and FlashInfer
`34bb66e4d0b420b5a03c99b2b6a1c0831dfaa738`. Each patch header records the full
upstream PR revision(s) and integration base, including bundled dependencies.
The vocabulary and package-data patches remain local additions. These are
validation bases; normal builds still follow their selected refs.

The refresh includes:

- #56273's review fixes, which were missing from #59958's bundled dependency:
  dummy-load handling, byte-based E4M3 scale decoding, 64-bit lookup row indices,
  pinned-row sizing, and an explicit ROCm rejection. The shared NVFP4 row decoder
  serves resident, pinned, and file-gather storage; dummy file-gather tables still
  bind their staging buffers even though checkpoint validation is skipped.
- #50897 at `5f36a6cd20f83e660876f432facbf2ed07a998e7`, including updated Mamba
  prefix-cache expectations, spawned MRV2 tests, and missing test imports/mock
  arguments. The allocation-time cache fix and Mooncake integration below remain.
- #5943's autotuner override fix, preserving the selected timer and first-tactic
  margin, plus parallel SM12x MXFP8 compilation and queued bucket preparation.
- A refreshed #54614 test hunk preserving both its bounded-logits test and the
  autotune cache-key test added by upstream
  [#60085](https://github.com/vllm-project/vllm/pull/60085). The overlapping
  insertion previously stopped regular-lane builds.

Validation: both exported series replay cleanly from the bases above and
reproduce the prepared source trees exactly. All 17 subsequent vLLM source
patchers and the FlashInfer loader patch pass; all 141 changed Python files
parse. The 7 series tests, 28 SWA tests, and 100 build-selection checks pass.
Focused CPU checks cover cache-boundary publication, Mooncake save limits and
references, PLE dummy binding/byte-scale lookup, and autotuner overrides. The
32,768-token vocabulary and its wheel package-data rule are retained. GPU
compilation and inference remain for nightly CI.

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
`75e3b17c055c609f02c64cd4ad3f8692b8170d8e`. That test-only refresh left runtime
patch content and lane selection unchanged. A CPU-only regression in
`tests/test_regular_patch_series.py` checks that both tests survive application.

Later on October 9, #50897 was refreshed to upstream head
`92176bd44606a3cd5857b5fd05b739cbcc434daf`. The old snapshot registered Mamba
tails at allocation but deferred full-block publication until step output.
At a full-block checkpoint, that could publish the same boundary twice and
trip `BlockPool.cache_full_blocks`'s `block_hash_num_tokens` assertion. The
updated PR restores allocation-time caching, removes the publication fence
and special Mamba-tail path, and only publishes the final lookahead block at
request finish. It also includes the upstream multi-module MTP fallback and
unified lookahead hasher.

The full 14-patch vLLM series was replayed against upstream main
`7d0b4e57aac4c4323225b0eee9ae960ce75ffffe`. Integration with the newer
Mooncake fix [#60695](https://github.com/vllm-project/vllm/pull/60695) preserves
explicit boundary puts, original prompt and resumed-prefill lengths, and
references for asynchronous store jobs. Lookahead hashes replace the extra
EAGLE proof margin for partial-tail saves too. Both sides of the overlapping
MTP tests are retained, and the Mooncake tests cover lookahead partial tails.
Focused CPU checks using extracted source methods reproduced the old cache
assertion and passed with the refreshed source; save limits, tail-block
references, and the build-selection tests also passed. Full inference
validation remains with the next nightly CI run.

The supplied conflict resolutions are retained, with adaptations for newer
upstream Mamba cache-boundary and KV-transfer fixes (#60533, #59197, and #60695).
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

`--load-format b12x` is opt-in for Qwen3.8 Flash Next NVFP4 and its embedded
MTP draft (the upstream `Qwen4Exp` classes). This backport supports Spark with
TP=1 or TP=2, without EP, DP, pipeline/context parallelism, EPLB, sleep mode,
or standalone weight reload. Other model architectures are rejected.
`--model-loader-extra-config '{"io_threads":8}'` sets the reader's I/O threads
(1–16, default 8). `read_mode` accepts `auto` or `bounce`, both selecting the
Spark transport; this backport does not expose the fork's shared-GDS path.
Multithread checkpoint routing and non-default safetensors read strategies
are rejected.

Ordinary weights use direct reads into CUDA parameter slices. PLE weights
retain the regular lane's mmap transport so device, pinned-host, and
`engram_config.host_file_gather` tables preserve their existing behavior.
Use the regular PLE configuration; the fork's `VLLM_PLE_TABLE_MEMORY` selector
is not introduced. B12X I/O counters exclude those mmap-loaded PLE bytes.
Both decoded CSF scales and opt-in resident CSF scales are supported; retained
quantized draft-head weights are materialized before their values are used.
The load format does not select inference kernels or change other load formats.

Flash Next FP4 resident compressed expert scales require `VLLM_CSF_RESIDENT=1`
and a CSF checkpoint with per-layer ModelOpt `weight_scale_encoding: csf`
metadata, using `--quantization modelopt_mixed`. This path requires SM12x B12X,
bias-free BF16 SwiGLU experts, and TP without EP, DP, EPLB, pipeline parallelism,
or ubatching. With the option unset, the default loader continues decoding CSF
scales through #60646. The additional post-load heap trim defaults to enabled;
set `VLLM_TRIM_HOST_HEAP_AFTER_LOAD=0` to disable that trim. GPU execution of the
resident-scale path remains unvalidated.

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

The same vocabulary is also packaged at
`vllm/v1/worker/gpu/spec_decode/draft_vocab_lists/qwen38_flash_next_32k.json`
to preserve the Flash Next FP4 mod's documented `draft_token_map` path.
