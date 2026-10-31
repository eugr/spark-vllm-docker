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

1. [#59958](https://github.com/vllm-project/vllm/pull/59958): disk-offloaded PLE tables, using #56273's NVFP4 support already in main. This supersedes the earlier #58815-based host-file-gather implementation.
2. [#59740](https://github.com/vllm-project/vllm/pull/59740) and [#60226](https://github.com/vllm-project/vllm/pull/60226): sparse draft vocabulary, including quantized heads.
3. [#60068](https://github.com/vllm-project/vllm/pull/60068): adaptive draft depth.
4. [#54912](https://github.com/vllm-project/vllm/pull/54912): QSA ring sizing.
5. [#50897](https://github.com/vllm-project/vllm/pull/50897): lookahead-aware prefix hashing.
6. [#54614](https://github.com/vllm-project/vllm/pull/54614): dynamic NVFP4 linear kernels.
7. [#54617](https://github.com/vllm-project/vllm/pull/54617): dynamic NVFP4 MoE.
8. [#60252](https://github.com/vllm-project/vllm/pull/60252): DFlash2 draft cache block sizing.
9. [#59021](https://github.com/vllm-project/vllm/pull/59021) and [#59025](https://github.com/vllm-project/vllm/pull/59025): separate unquantized linear selection and cuDNN BF16 GEMMs.
10. The supplied 32,768-token Flash Next draft vocabulary.
11. [#60646](https://github.com/vllm-project/vllm/pull/60646): CSF-compressed NVFP4 scale loading.
12. Wheel package-data inclusion for the draft vocabulary.
13. The local Flash Next FP4 mod from `mods/draft/flash-next-fp4`:
    opt-in resident CSF expert scales, ModelOpt dispatch, nested scale loading,
    weight-transfer helpers, and post-load host-heap trimming. Its vocabulary
    path is packaged alongside the identical existing vocabulary.
14. `17-flash-next-b12x-loader.patch`: opt-in `--load-format b12x` for Flash
    Next and its embedded MTP draft on Spark, using FlashInfer's checkpoint
    reader and the weight-transfer helpers supplied by patch 16.

[#55601](https://github.com/vllm-project/vllm/pull/55601) is already in the
validated upstream base. Quantized draft heads from
[#59973](https://github.com/vllm-project/vllm/pull/59973) are supplied by upstream
commit `784d1fff0c491b9add10ecd732de8f707a7e7dbe`; the duplicate patch 02 is retired.
[#59244](https://github.com/vllm-project/vllm/pull/59244)
modified a BF16-only warmup workaround that upstream
[#60434](https://github.com/vllm-project/vllm/pull/60434) subsequently removed.
The series retains the current full-model autotune path, including cuDNN, and
does not restore the obsolete 32-token cap or BF16-only pass.

The FlashInfer series contains
[#5943](https://github.com/flashinfer-ai/flashinfer/pull/5943)
(SM12x CuTe DSL MXFP8 GEMMs), at head
`15c5c8d21388c118558c8370299204aa4b110ced`.
Canonical NVFP4 W4A4 MoE from
[#5863](https://github.com/flashinfer-ai/flashinfer/pull/5863) is supplied by
upstream commit `397b9607c83cf201f3c44684c555c3698869b6bd`.

## Integration and refresh

The October 31 refresh follows pipeline 382's patch 06 conflict on vLLM
`87cb835db2bbbaa03e3faa6ae588ba92dce1eea4`. Upstream added a Qwen3.5 multi-module
MTP test and imports shared with patches 04 and 06. Their refreshed test sections
preserve that coverage alongside adaptive draft depth and prefix-cache reuse.

The complete series is validated on main
`564ac0de1b228568cafcbfbe8ccaccc82d6023eb`. Patch 03 retains the refactored
`share_target_lm_head` helper and its validation, deferring quantization until
sparse vocabulary rows are selected when a token map is configured. It also
keeps upstream's MTP step argument and DSpark local-logit path. Patch 06 gives
the new frontend prefix-hash configuration the same effective connector
lookahead depth as the engine hasher. Patch 07 preserves the new batch-invariant
W4A16 test, and patch 17 retains the current embedding imports.

All 15 series entries replay on a fresh archive without historical patch-base
objects: merged patch 01 skips and the other 14 apply, producing the reviewed
tree exactly. All 19 subsequent source patchers and CUTLASS pinning pass, and all
108 changed Python files compile. The 35-test series suite has 33 passes and two
optional tensor-dependency skips; all 107 build checks pass. Regressions cover
both MTP patches together, deferred shared-head quantization and its rejection
rules, matching frontend/engine hash policy, and preservation of upstream's
batch-invariant test. GPU wheel compilation and inference remain for CI. This
refresh does not change the separate B12X startup metadata-exchange failure.

The October 30 refresh fixes pipeline 381's patch 05 failure on vLLM
`deacc3670153845db2f013ea24085d6eaf338874`. Upstream
[#64110](https://github.com/vllm-project/vllm/pull/64110) moved the QSA cache
implementation to `qsa_indexer.py` and renamed its classes. Patch 05 now targets
`QSARawKeyCache` there while retaining the ring-sizing limits and tests.

The full series also targets main
`dc4dd5a2217bc757232e9fa4d8a8b59e207ca068`. Patch 01 matches the merged #59958
implementation, including its GPU synchronization guard, so the series detects
it as already present. Patch 06 preserves the new Mooncake storage layouts from
[#54307](https://github.com/vllm-project/vllm/pull/54307), the consumer decode
replay boundary, and the offload event ordering from
[#63249](https://github.com/vllm-project/vllm/pull/63249).

Validation replays the entire series on a fresh archive without old patch-base
objects: patch 01 skips and the other 14 patches apply. All 19 subsequent source
patchers and CUTLASS pinning pass; all 108 changed Python files compile. The
31-test series suite has 29 passes and two tensor tests skipped because their
optional dependencies are unavailable; all 107 build checks pass. New regressions cover
the renamed QSA cache's ring limits and preservation of Mooncake's consumer
decode replay boundary with lookahead enabled or disabled. Wheel compilation
and inference remain for CI; the separate B12X startup metadata-exchange failure
is outside this source refresh.

The October 29 full PR refresh validates vLLM main
`8084f0624dba24aba6ebac62d8e10cc00730a0fe` and FlashInfer main
`0d5bbf50b3fb3695dd2a4cf9feddabd2ab172412`. Three referenced PR heads changed:

- #59958 at `396e7e0102896dda576dd1660fb64bc2f6ffd232` replaces
  `engram_config.host_file_gather` with `engram_config.disk_offload`. It reads
  checkpoint rows with `preadv`, stages them in fixed buffers, and reports
  truncated files and unsupported weight reloads explicitly. BF16, FP8 and
  NVFP4 storage retain upstream decoding and sequence-parallel behavior.
- #59740 at `178caf8dfb0908542ffb7dbf0f72c42f11b99fc0` uses
  `draft_lm_head_quantization` for sparse rows, fixes the dynamic scorer rank
  at 256, and rejects tensor parallelism for `draft_token_map`. The old
  `draft_token_map_quantization` and `draft_token_map_dynamic_rank` fields are
  removed. #60226's checkpoint-quantized heads remain supported through the
  new API, including padded vocabulary outputs and retained checkpoint weights.
- #59025 at `67b308c2f75f7e26b33055afed0310f771afeee2` limits the cuDNN
  BF16 dispatch to SM12x and output widths at most 128.

All other active PR heads, including FlashInfer #5943, are unchanged. Merged
dependencies remain supplied by main. The refresh resolves pipeline 380 job
692's three MTP import conflicts without restoring the removed fused-expert
helper. It also preserves upstream draft-logit caching, target attention
metadata reuse, Qwen2MoE fused activation quantization, and online shared-expert
name redirection. The lookahead integration test uses upstream's current
`get_tokenizer` API.

Validation uses fresh archives without old patch-base objects: all 15 vLLM
patches and the FlashInfer patch reproduce the prepared source trees exactly.
All 19 subsequent vLLM source patchers, CUTLASS pinning and the FlashInfer
loader patch pass. All 114 changed vLLM and 17 FlashInfer Python files compile
and pass undefined-name checks. The 29 series tests include CPU tensor checks
for PLE checkpoint bytes, stable staging buffers, truncation, sparse logits
and quantized vocabulary padding. Install `torch`, `safetensors` and `numpy`
in an isolated test environment to run those two tensor tests; without them,
the standard-library suite skips them. The loader's CPU tests pass with its
GPU check skipped. Wheel compilation and inference remain for CI; this source
refresh does not resolve the separate B12X memory-budget failures.

On October 29, pipeline 380 stopped at FlashInfer patch 02 on upstream
`82e93625e87b196a9cb0d83fd5d85a29b8e7bdc7`. Upstream now coordinates CUPTI
availability, initialization, and failure handling across tuning ranks. The
autotuner section preserves that coordination, including initialization before
warmup and graph capture, while retaining per-operation timer selection and
profiling repeats. The recorded PR snapshot and all other file sections are
unchanged.

The complete patch applies to the failed revision without old patch-base
objects, all 17 changed Python files compile, and reapplication is detected.
All 23 CPU series tests pass. The new regression reproduces the old conflict,
preserves upstream collective fallback, verifies timer-policy precedence, and
checks that peer initialization failure prevents local warmup. Full wheel
compilation and inference remain for CI.

The next October 27 run, job 685, reached patch 10 on upstream vLLM
`c4f2bb4abbdbad9ed5b0a71acd0bd585056eeab8` and conflicted with the new RDNA
hybrid prefill-dequantization setting. Patch 10 now retains that setting and
its validator alongside the independent unquantized-linear backend override.

All 15 patches and 19 subsequent Docker source patchers replay on fresh
archives of both the failed revision and upstream main
`c56706e88e0a699274e7ed807e072274048d1c33`. CUTLASS pinning passes and all 114
changed Python files compile on both revisions. The 22 CPU series tests and
101 build checks pass; the new regression preserves the RDNA setting and
normalization while adding the unquantized override. Full wheel compilation
and inference remain for CI.

Later on October 27, job 684 stopped at patch 05 on upstream vLLM
`6aa70c4beca4f85c44b9f04c23f46326d9261f14`. Upstream added model imports and
a BF16-indexer regression test in `tests/models/qwen4_exp/test_config.py`.
Patch 05's test section now preserves those additions alongside all three QSA
ring tests. Its runtime patch is unchanged.

All 15 patches replay from a fresh source archive without old patch-base
objects and produce the same tree as the development replay. All 19 subsequent
Docker source patchers and CUTLASS pinning pass; all 114 changed Python files
compile. The 21 CPU series tests and 101 build checks pass. The new regression
reproduces the old import conflict and checks that upstream's BF16-indexer test
survives. Full wheel compilation and inference remain for CI.

On October 27, pipeline 378 stopped at patch 04 on upstream vLLM
`90309f6b742cdb1d6305778b81063918cf61be81`. The draft sampler now dispatches
watermarked and ordinary probabilistic sampling through separate branches.
Patch 04 keeps both branches, including the sharded-logit arguments, and
updates confidence before either sampler can modify the logits in place.
Patch 05's test-import context was also refreshed after upstream removed the
Qwen3.5 model-config import; the QSA ring implementation and tests are retained.

All 15 patches apply directly to a fresh archive of that revision without old
patch-base objects, and the resulting tree matches the development replay.
All 19 subsequent Docker source patchers pass and all 114 changed Python files
compile. The 20 CPU series tests and 101 build checks pass. The new regression
covers watermark, sharded probabilistic, greedy, and local-argmax draft sampling,
including confidence checks before in-place logit modification. Full wheel
compilation and inference remain for CI.

On October 26, patches 03, 04, 06, 07, and 08 were refreshed against pipeline
377's failed vLLM base `fe5fe8ba5fb9412ac6deafd192ae8b7c466b96c5`. Patch 03 keeps
the new Parse MTP validation beside the draft-vocabulary guard. Upstream
[#61589](https://github.com/vllm-project/vllm/pull/61589) now supplies the core
lookahead hashing, including multi-token lookahead and the current KV-event
format. Patch 06 retains the one-token KV-connector extensions and streaming
hash invalidation, using the upstream integer API. HiSparse retains the legacy
block-drop behavior. Patch 07 preserves upstream's disabled B12X Buildkite job
and its AMD router mirror; patch 08 preserves the expanded monokernel settings.

All 15 patches apply directly to a fresh copy of that source tree without old
patch-base objects. All 107 changed Python files compile, all 19 subsequent
Docker source patchers apply, and the 19 CPU series tests and 101 build checks
pass. The regressions cover the Parse MTP guard and local multi-token versus
connector one-token hashing. GPU compilation and inference remain for CI.

On October 25, conflicting sections of vLLM patches 01, 03, 04, 05, 06, and 16
were refreshed after nightly pipeline 376 stopped at patch 01's PLE test imports.
Patch 02 was retired because #59973 is merged; keeping the upstream versions of
its three conflicting files leaves no changes from that patch. The remaining
15 patches replay on both the failed base
`24324ef0f667c633ee9bbb9412343d829f7e6d4e` and upstream main
`0765565f103d5579382ad61994a041654684778b`.

The refresh preserves upstream FLy and heterogeneous-vocabulary validation,
the Model Runner V1 dynamic-depth guard, the speculator's live model-length
property, QSA cache views, and loader configuration access. Streaming updates
retain all emitted tokens except the final sampled token; lookahead hashes
covering that discarded token are invalidated. Only conflicting file sections
were regenerated, and the headers record their refresh base. Strict conflict
handling remains enabled for tests, documentation, and runtime code.

All 118 Python files changed by the series compile on both bases. The 18 CPU
patch-series tests pass, including streaming updates whose emitted tokens are
ahead of computation. GPU compilation and inference remain for CI.

On October 24, the FlashInfer series was refreshed against the failed nightly
base `431c9687f3bb9a048e194c74a178d7cfb02b1585` and replayed on
`03f82fddadaebed69879b4e7d28f3df38c4e28d2`. Patch 01 was retired: its added
lines exactly match the merged #5863 implementation. Patch 02 retains its
recorded PR snapshot while preserving upstream cold-chain and individual
timing, `dataclasses.replace` configuration overrides, and SM107 low-M GEMM
dispatch. Strict patch-conflict handling is unchanged.

Both replays include the loader include-order fix, and all 17 Python files
changed by the remaining FlashInfer patch compile. CPU regression tests cover
tuning-policy preservation and SM107/SM12x dispatch. GPU compilation and
inference remain for CI.

On October 14, the series was refreshed against vLLM
`355553ac22dca4080929ea8f88a446f15df1f3ba` and FlashInfer
`2895577720741ddfedff118202c89584996b8875`. The complete vLLM series also
replays on the failed nightly base `3f466c852cfd3882820af8dd7963ac775174bce4`.

[#56273](https://github.com/vllm-project/vllm/pull/56273) merged as
`4e0681ed5654c1d8afb3c104d4381afd08529de8`. At that point, patch 01 retained
only the file-gather integration; merged NVFP4 definitions and tests came
from main. The October 14 compatibility fix preserved `host_file_gather`,
sequence-parallel behavior, and loader fixes without adopting #59958's newer
`disk_offload` rewrite. The October 29 full PR refresh above supersedes that
snapshot and adopts `disk_offload`.

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
`engram_config.disk_offload` tables preserve their storage behavior.
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

For an MTP drafter using Model Runner V2 and tensor parallel size 1, set
`draft_token_map` to the installed vocabulary path. For an unquantized lm_head,
set `draft_lm_head_quantization` to `nvfp4` in `--speculative-config` to quantize
only the selected rows. Checkpoint-quantized heads retain their own format;
leave that option unset. Resolve the path inside the built image with:

```bash
python3 -c 'from importlib.resources import files; print(files("vllm") / "model_executor/layers/draft_vocab_lists/qwen38_flash_next_32k.json")'
```

The JSON is included in the exported vLLM wheel, so the runner does not need the
builder's source checkout.

The same vocabulary is also packaged at
`vllm/v1/worker/gpu/spec_decode/draft_vocab_lists/qwen38_flash_next_32k.json`
to preserve the Flash Next FP4 mod's documented `draft_token_map` path.
