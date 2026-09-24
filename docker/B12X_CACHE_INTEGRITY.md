# Temporary B12X CuTe cache-integrity patch

An interrupted node startup can leave empty or damaged `.o` objects or `.json`
manifests in the persistent CuTe compile cache. B12X preparation previously
considered a CuTe program available if its object file existed. The later load
could fail with `planned CuTe program has not been compiled` instead of scheduling
the damaged entry for recompilation.

`Dockerfile` applies `patch_b12x_cache_integrity.py --installed` after both the
source and PyPI B12X install paths in the common runner stage. This also covers
regular runners assembled from precompiled vLLM wheels. It is not a runtime mod
and does not require recipe changes. Existing images must be rebuilt to include
the fix; this change does not modify running containers or host caches.

## Behavior

- Preparation validates the object/manifest pair's key, byte count, and SHA-256.
  Missing, malformed, or mismatched metadata is a cache miss, including older
  objects without manifests. Invalid entries are replaced by the existing
  compiler path, which rechecks after acquiring its per-key lock.
- Validation results are bounded and memoized by both files' device, inode,
  size, modification time, and change time. Rewrites and atomic replacements
  trigger revalidation. No whole-cache scan runs at startup.
- Loading validates the private staged object before handing it to CUTLASS.
  Loader mutations cannot alter the canonical cached object.
- Publication fsyncs temporary object and manifest files before replacement,
  then fsyncs their directory entries. New shard directories are also synced.
  The manifest is published last; a partial pair is rejected on the next run.
- The `preparation/` tuning-selection metadata has a different format and is
  not inspected or removed by this patch. Triton caching is unchanged.

Checksums verify stored bytes, not CUDA ABI compatibility or kernel correctness.
The patch does not promise recovery from arbitrary CUDA loader failures when a
checksum-valid artifact is incompatible. Persistent filesystem errors can still
prevent compilation artifacts from being published.

B12X includes its package source fingerprint in cache keys. Installing this
patch therefore causes a one-time recompilation under new keys, even for old
entries that were healthy. No old cache entries are proactively deleted.

## Upstream patch

`b12x-cache-integrity.patch` includes the B12X implementation and standalone
CPU-only regression tests. It was prepared against
`local-inference-lab/b12x@4f3028b19c1d8290dc72b6f483aba40de23eae5a`.
The compiler hunks also apply to the PyPI 1.3.0 wheel. That release has no
`compile_plan.py`, so the build patcher omits only the planner hunk and applies
the shared validation and durable-write changes to its normal compiler path.
From a compatible B12X checkout:

```bash
git apply --check /path/to/vllm-docker/docker/b12x-cache-integrity.patch
git apply /path/to/vllm-docker/docker/b12x-cache-integrity.patch
python3 tests/_lib/test_compile_cache_integrity.py
```

The Docker patcher applies only the `b12x/` hunks to site-packages; it does not
install the upstream test file. It locates B12X without importing GPU packages,
checks all hunks before editing, accepts an identical repeated application, and
fails the build on unknown source layouts instead of silently skipping the fix.
If B12X is not installed, there is nothing to patch. Remove this temporary patch
once the supported source/PyPI versions contain the upstream fix.

## Local validation

```bash
python3 tests/test_b12x_cache_integrity_patch.py
./tests/test_build_and_copy.sh
```

The local suite applies the canonical diff to reviewed upstream source excerpts
and runs its included regression tests. CUDA compilation and loading are mocked;
cache reads, hashing, replacement, locks, and fsync operate on temporary files.
Tests cover valid reuse, interrupted publication, empty/zero-filled/truncated
entries, same-size corruption, malformed manifests, repair after a cached miss,
another writer repairing an entry, durability ordering, and unchanged tuning
metadata. Live GPU startup and power-loss testing are separate validation steps.
