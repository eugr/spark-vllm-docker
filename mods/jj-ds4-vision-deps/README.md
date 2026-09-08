# mods/jj-ds4-vision-deps

Runtime dependency stack for the DeepSeek-V4-Flash-Vision-Exp recipe (vLLM
`local-inference-lab/vllm` PR [#634](https://github.com/local-inference-lab/vllm/pull/634)
on b12x).

This mod covers the parts that must apply **at container start** and therefore
work against the *existing* `vllm-node-b12x` image: the b12x source overlay
(pure Python/JIT, no rebuild) and the LMCache build/install (dev-lineage
engine-driven source, **not** the old PR #44 tree).

The vision feature itself — **[vLLM PR #634](https://github.com/local-inference-lab/vllm/pull/634)** —
is NOT merged into the `dev/jovian-judgement` branch, so it must be **baked
into the vLLM build** at image-build time. The recipe does that via
`build-and-copy.sh --exp-b12x --rebuild-vllm --apply-vllm-pr 634`
(the `--exp-b12x` profile builds from `local-inference-lab/vllm` ref
`dev/jovian-judgement`; PR #634 is fetched as patch from that same fork via
`pull/634/head` and applied onto the ref — verified clean 3-way apply, 10
commits / 40 files).

The other two vLLM-side deps in the stack — [#553](https://github.com/local-inference-lab/vllm/pull/553)
(engine-driven LMCache + expandable CUDA allocator segments) and
[#671](https://github.com/local-inference-lab/vllm/pull/671) (fused
padded-query output accounting) — are **already present in
`dev/jovian-judgement`** in equivalent form (confirmed in
`vllm/config/vllm.py` `engine_driven` + expandable-segments paths, and the
caller-owned `q_out` kernel contract). They do NOT need `--apply-vllm-pr`;
adding them is a no-op (Docker skips already-applied patches). These two are
not applied by this mod at runtime either — they are build-baked, and an image
built from the branch already carries them.

## What it does (in `run.sh`, idempotent)

1. **b12x overlay** — copies 7 files over the installed `b12x` package
   (found via `import b12x`, not hardcoded). b12x is a pure Python/JIT
   package, so source overlays are enough; kernels recompile on first use.

   | file | source | PR |
   |---|---|---|
   | `attention/_shared/mla/prefill.py` | [#301](https://github.com/local-inference-lab/b12x/pull/301) head | FP8 dual-cache prefill, sparse topk-512 |
   | `attention/_shared/mla/prefill_mg.py` | [#301](https://github.com/local-inference-lab/b12x/pull/301) head | same |
   | `comm/pcie/_cute_intrinsics.py` | [#246](https://github.com/local-inference-lab/b12x/pull/246) head | TP2 peer-push fix |
   | `comm/pcie/_oneshot_cute.py` | [#246](https://github.com/local-inference-lab/b12x/pull/246) head | PIECEWISE shape binding |
   | `comm/pcie/pcie_allreduce.py` | [#246](https://github.com/local-inference-lab/b12x/pull/246) head | TP2 allreduce |
   | `comm/pcie/pcie_oneshot.py` | [#246](https://github.com/local-inference-lab/b12x/pull/246) head | TP2 oneshot |
   | `norm/mhc/_impl.py` | [#306](https://github.com/local-inference-lab/b12x/pull/306) head | `rms_eps=1e-20` allowance |

   (`local-inference-lab/b12x` PRs [#301](https://github.com/local-inference-lab/b12x/pull/301),
   [#246](https://github.com/local-inference-lab/b12x/pull/246),
   [#306](https://github.com/local-inference-lab/b12x/pull/306) — all open.
   #306's exact change is _also_ already present in `master`, but the baked
   image is on `b58f34ea` which predates it, so we include `_impl.py` too.)

   Post-overlay checks: `MHC_SUPPORTED_RMS_EPS` must contain `1e-20`, and all
   7 modules import clean.

2. **LMCache install (dev + 5 follow-up PRs, vendored tarball)** —
   `lmcache-dev5-src.tar.gz` (2.2M) is a vendored, gzip'd copy of the LMCache
   `dev` branch (`7ed46754`) onto which the five follow-up PRs from vLLM
   PR #634's runtime-dependency list —
   [#49](https://github.com/local-inference-lab/LMCache/pull/49),
   [#50](https://github.com/local-inference-lab/LMCache/pull/50),
   [#51](https://github.com/local-inference-lab/LMCache/pull/51),
   [#55](https://github.com/local-inference-lab/LMCache/pull/55),
   [#56](https://github.com/local-inference-lab/LMCache/pull/56) — have been
   cherry-picked (each applies cleanly because all five branch from `dev`
   HEAD). `PR_SOURCE` inside the archive records the `dev` base + every PR
   commit. The archive contains `lmcache/` + `csrc/` + build scaffolding only
   (tests/docs/examples excluded, `__pycache__`/`*.pyc`/`build`/`*.egg-info`
   stripped), matching the #44 tarball convention. `run.sh` extracts it to a
   writable `/tmp` dir (setuptools writes `build/` into the source root, and
   mod trees may be mounted read-only), then installs with
   ```bash
   uv pip install --no-build-isolation --no-deps <staged>
   ```
   `--no-build-isolation` is required: the repo's pyproject build-system pins
   `torch==2.13.0` (matching the image torch 2.13.0+cu130, but build isolation
   would still re-resolve) — skipping build isolation uses the image's own
   torch/toolchain (verified present: torch 2.13.0+cu130, ninja, gcc/g++,
   nvcc, wheel, cuda-python). `--no-deps` avoids re-resolving deps the image
   already provides (redis/cufile are optional storage backends, intentionally
   not pulled; vLLM uses the gpu/disk backend). `SETUPTOOLS_SCM_PRETEND_VERSION`
   is exported because setuptools_scm wants a git tree and the vendored dir is
   not git. Verified afterwards that the engine-driven modules import and the
   follow-up markers are present (`PagedKVTransferWorkspace` = #50,
   `dynamically_pinned` in `scatter_cpu_to_paged_kv` = #55,
   `engine_driven_shm_pool` in `EngineDrivenTransferModule` = #56), i.e. it is
   the dev+5 source, not a dev / 0.5.x wheel LMCache.

   > **#44 is intentionally superseded.** vLLM PR #634's dependency list used
   > to name LMCache `#44` (`release/v0.5.2-glm52-dcp-base` lineage). As of the
   > 2026-09-07 update to #634, #44 is *gone* — replaced by the five dev-lineage
   > PRs #49/#50/#51/#55/#56. The dev engine-driven path is a different
   > implementation from #44's (it has no `EngineDrivenContextPickle`; the
   > context lives in `worker_transfer.py` as `EngineDrivenTransferContext`).
   > Build the dev+5 source, **not** #44 or any `release/v0.5.x` wheel, whose
   > engine-driven path would not match what #634's launcher expects.

3. **flashinfer — no-op.** The baked flashinfer `0.6.18`
   (git commit `18e5811d`) already contains the SM120 sparse-MLA topk-512
   fallback plus the `extra_page_block_size==2` dispatch that PR/commit
   `803c4664` ([flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer))
   adds. Verified: `csrc/sparse_mla_sm120_prefill.cu`
   in `18e5811d` already has the `extra_page_block_size == 2` branch.

## How to rebuild the vendored tarball

`lmcache-dev5-src.tar.gz` is a curated snapshot, not a living checkout. To
reproduce or update it, from a machine with network access:

```bash
# 1. Clone the fork and check out the dev branch base used by this mod.
git clone https://github.com/local-inference-lab/LMCache.git lmcache-dev5
cd lmcache-dev5
git checkout 7ed4675404a31f4ffafd98975899dc83832ba965   # dev base (PR_SOURCE)

# 2. Cherry-pick the five follow-up PRs, in the order recorded in PR_SOURCE.
#    Each is a clean apply because all five branch from dev HEAD.
git fetch origin refs/pull/49/head && git cherry-pick 3e8601aa 31b704d1 a952cedd 5c57de29 0bdb4a3e
git fetch origin refs/pull/50/head && git cherry-pick 94c5514a
git fetch origin refs/pull/51/head && git cherry-pick 24e9a80c 63919a2c
git fetch origin refs/pull/55/head && git cherry-pick 55f50e08 078199fa   # incl. upstream #4830
git fetch origin refs/pull/56/head && git cherry-pick a726d467
# Combined tree HEAD should be 14f4b01306af6c81705128c1ee87c1ee78ca9f81.

# 3. Record provenance. PR_SOURCE is what run.sh/verify.sh rely on to detect a
#    stale or mismatched tree; keep it in the archive.
cat > PR_SOURCE <<'EOF'
# LMCache engine-driven source for the vLLM #634 stack (replaces #44).
# Base: local-inference-lab/LMCache branch `dev` @ 7ed4675404a31f4ffafd98975899dc83832ba965
# Follow-up PRs (all based on dev HEAD, cherry-picked in this order):
#   #49 fs restart-safe keys        -> 3e8601aa 31b704d1 a952cedd 5c57de29 0bdb4a3e
#   #50 reuse paged transfer metadata -> 94c5514a
#   #51 bounded SHM ownership      -> 24e9a80c 63919a2c
#   #55 async-copy lifetime sync   -> 55f50e08 078199fa  (incl. upstream #4830)
#   #56 transport identity report  -> a726d467
# Combined tree HEAD (this archive): 14f4b01306af6c81705128c1ee87c1ee78ca9f81
EOF

# 4. Strip everything not needed at build time (tests/docs/examples, caches,
#    build artifacts), then tar+gzip. Keep lmcache/, csrc/, requirements/,
#    setup_extensions/, pyproject.toml, setup.py, and PR_SOURCE.
rm -rf tests docs examples .git __pycache__ \
       $(find . -name '__pycache__' -o -name '*.pyc' -o -name 'build' -o -name '*.egg-info')
tar czf ../../mods/jj-ds4-vision-deps/lmcache-dev5-src.tar.gz \
    lmcache csrc requirements setup_extensions pyproject.toml setup.py PR_SOURCE
```

The `git fetch origin refs/pull/<n>/head && git cherry-pick ...` step is what
guarantees the archive is exactly the dev+5 tree (not an upstream `dev` or a
`release/v0.5.x` wheel). If any of the five PRs is merged upstream or its commit
hashes change, update `PR_SOURCE` to match and note it in the commit message.

## Provenance / verification

- b12x PR heads: `#301=223f88c2`, `#246=ea76030d`, `#306=3f6896dc`
  ([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x),
  all open, base on master history).
- LMCache: base `dev` @ `7ed46754`, plus the five follow-up PRs
  [#49](https://github.com/local-inference-lab/LMCache/pull/49)/[#50](https://github.com/local-inference-lab/LMCache/pull/50)/[#51](https://github.com/local-inference-lab/LMCache/pull/51)/[#55](https://github.com/local-inference-lab/LMCache/pull/55)/[#56](https://github.com/local-inference-lab/LMCache/pull/56)
  cherry-picked on top (see `PR_SOURCE` for the exact commit list; combined
  tree HEAD `14f4b013`). The `release/v0.5.2-glm52-dcp-base` lineage (#44's
  base) is deprecated for this stack — #634 no longer depends on #44.
- **LMCache follow-up PRs** in the #634 stack (all vendored INTO this tarball):
  #49 (restart-safe fs keys), #50 (reuse paged transfer metadata), #51 (bounded
  shared-memory ownership), #55 (asynchronous-copy lifetime synchronization),
  #56 (transport identity reporting). They are no longer "boundaries to check";
  they are the active LMCache DNA for the #634 engine-driven path.
- **vLLM build-side deps** (baked at image build, not by this mod):
  - [#634](https://github.com/local-inference-lab/vllm/pull/634) — DeepSeek V4
    Vision (+ DSpark drafter, streaming checkpoint loader, qualified LMCache
    profiles). NOT in the branch: **must** be applied via `--apply-vllm-pr 634`.
    Verified clean 3-way apply onto `dev/jovian-judgement` (10 commits / 40
    files).
  - [#553](https://github.com/local-inference-lab/vllm/pull/553) — engine-driven
    LMCache + expandable CUDA allocator segments. Already present in
    `dev/jovian-judgement` (equivalent code in `vllm/config/vllm.py`); no
    `--apply-vllm-pr` needed.
  - [#671](https://github.com/local-inference-lab/vllm/pull/671) — fused
    padded-query output accounting before GPU KV admission. Already present in
    `dev/jovian-judgement` (caller-owned `q_out` kernel contract); no
    `--apply-vllm-pr` needed.
- flashinfer commit `803c4664` ([flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer),
  not a branch head).

## Runtime dependency matrix (as of PR #634 update 2026-09-07)

| Component | Patch | Covered where |
|---|---|---|
| b12x | [#246](https://github.com/local-inference-lab/b12x/pull/246) generation-safe TP2 peer-push + PIECEWISE binding | this mod (run-time overlay) |
| b12x | [#301](https://github.com/local-inference-lab/b12x/pull/301) FP8 V4 dual-cache prefill, sparse topk 512 | this mod (run-time overlay) |
| b12x | [#306](https://github.com/local-inference-lab/b12x/pull/306) `rms_norm_eps=1e-20` specialization | this mod (run-time overlay) |
| vLLM | **[#634](https://github.com/local-inference-lab/vllm/pull/634) Vision** (NOT in branch) | **baked via `--apply-vllm-pr 634`** |
| vLLM | [#553](https://github.com/local-inference-lab/vllm/pull/553) engine-driven LMCache + expandable-CUDA segments | already in `dev/jovian-judgement` |
| vLLM | [#671](https://github.com/local-inference-lab/vllm/pull/671) fused padded-query output accounting | already in `dev/jovian-judgement` |
| LMCache | dev base + **#49/#50/#51/#55/#56** (engine-driven; **replaces #44**) | this mod (vendored install) |
| FlashInfer | `803c4664` SM120 sparse-MLA topk-512 fallback | baked in image flashinfer 0.6.18 (no-op today) |

## Testing

- Overlay logic verified in a temp container (`vllm-node-b12x:latest`):
  `MHC_SUPPORTED_RMS_EPS` goes `(1e-06,1e-05)` → `(1e-20,1e-06,1e-05)`, 7
  modules import clean.
- LMCache dev+5 build path: verified in the same container — `uv pip install
  --no-build-isolation --no-deps` on the vendored tree builds in ~48s, installs
  `0.5.2.post0+jjdev5.7ed4675404...`, and the post-install marker checks pass
  (`PagedKVTransferWorkspace`, `dynamically_pinned`, `engine_driven_shm_pool`).
  `verify.sh` reports ALL PASS; re-running `run.sh` is a no-op.
- End-to-end: exercise via `launch-cluster.sh` with this mod applied (needs
  network + HTTP(S)_PROXY, GPU image). See the recipe's
  `recipes/deepseek-v4-flash-vision-exp.yaml`.
