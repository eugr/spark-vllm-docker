# gb10-wfe-waits

On DGX Spark (GB10), multi-node serving keeps several CPU cores at 100% for the whole generation, just waiting:

- the b12x RoCE proxy on every node (`VLLM_ENABLE_ROCE_ALLREDUCE=1`) polls its doorbell word;
- on the head node, `EngineCore` and the local worker poll the shared-memory broadcast queue.

Those cores run at full clock inside the same power and thermal budget as the GPU. This mod makes them wait in
**WFET** instead: the core arms the exclusive monitor on the watched word (`LDXR`) and stays in a low-power wait until
another core or the GPU writes it, an event-stream tick or a 1 ms deadline. In standalone tests the median wake-up was
0.6–0.9 µs after another core's store. It changes how threads wait, not what is computed. An idle server still sleeps:
after a long idle stretch the proxy naps between polls exactly as the stock loop does, so the kernel's idle states can
take the core.

## What it changes (inside each container, head and workers)

1. `b12x/comm/roce/_roce_proxy.c` (standalone b12x or FlashInfer's `flashinfer/experimental/b12x`): the idle doorbell
   wait uses `LDXR` + WFET for its first 8192 waits (8192 nominal 1 ms park budgets, about 8.2 s total, excluding
   loop work and scheduling delays), then naps between polls as the unpatched loop does. b12x builds the proxy from
   this source at startup.
2. `vllm/spinloop*.so`: rebuilt from the bundled `spinloop.cpp` with an aarch64 `LDXR` + WFET branch next to the
   existing AMD `MONITORX`/`MWAITX` one.
3. `vllm/distributed/device_communicators/shm_broadcast.py`: a reader also passes its own read flag to `spinloop()`
   (`check()` reads that byte too).
4. `vllm/envs.py`: `VLLM_USE_SPINLOOP_EXT` defaults to `1`, only if part 2 is in place.

Each part applies completely or is skipped with a message, leaving the existing file unchanged. The mod is
idempotent, keeps the original files as `*.stock`, test-builds the patched proxy, and self-tests the rebuilt
extension before installing it. It does nothing on hosts without FEAT_WFxT. `WFE_WAITS_STRICT=1` makes it exit 1 if
any part was skipped.

## Opt-outs at runtime

| Variable | Effect |
| --- | --- |
| `B12X_ROCE_WFE=0` | RoCE proxy spins again |
| `VLLM_SPINLOOP_WFE=0` | `spinloop` uses its stock `yield` loop |
| `VLLM_USE_SPINLOOP_EXT=0` | stock Python wait in `shm_broadcast` |
| `B12X_ROCE_DEBUG=1` | the proxy prints which wait it uses |

While requests are being served, `top` keeps showing these threads at ~100%: WFET does not deschedule the task. The
difference is at the wall. On an idle server the proxy's nap takes over, as it does without the mod: in a settled
120 s idle window with this mod applied, the mid-window sample's hottest thread in the lane container's process tree
was at 3.0% on the head and 3.2% on the worker, and wall power was 47.45 / 47.34 W against stock's 47.78 / 47.67 W.

## Measured (2× DGX Spark, TP=2, GLM-5.3-Flash W4A16, `spark-vllm-b12x:nightly-20260927`)

CPU uncapped, GPU locked at 1800 MHz, wall power at the plugs, means of two repeats in mirrored order; tool-eval-bench
hardmode (28 scenarios, `--parallel 3`) plus single-stream decode.

| | stock | proxy only | proxy + spinloop (tested runtime patches) |
| --- | ---: | ---: | ---: |
| Head / worker power, benchmark | 120.2 / 101.1 W | 111.7 / 93.8 W | **96.6 / 93.9 W** |
| Head peak temperature | 82 °C | 81.5 °C | **65.5 °C** |
| Energy per output token, benchmark / decode | 5.34 / 9.00 J | 5.28 / 8.82 J | **4.87 / 7.87 J** |

Greedy single-stream decode, stock against the full patch, alternated A Y A Y A Y with a restart per arm: −0.9%, +0.3%,
+0.8%. That small, noisy test did not resolve a consistent slowdown; the same prompt repeated within one arm varies by
2.2%. Proxy and queue errors in the container logs: 0 in every arm. Tested on GB10 only.

These numbers come from images with the same patched files installed directly; the mod itself was then checked by applying it to `spark-vllm-b12x:nightly-20260927` (by digest) with `run.sh` in a `docker build` on a GB10 node. All four parts applied; a second pass with `WFE_WAITS_STRICT=1` reported every part already applied and self-tested; the `.stock` backups are present; `VLLM_USE_SPINLOOP_EXT` defaults to 1 and the extension selects WFET; the patched proxy builds with b12x's own compile command. CPU-only smoke tests in that image: wake latency after another process's store p50 0.77–0.93 µs (stock `yield` loop 0.38–0.43 µs), timeouts, callback exceptions and GIL release all fine; vLLM `MessageQueue` ping-pong, 20,000 round trips, every message verified, both with the new default and with `VLLM_USE_SPINLOOP_EXT=0`. Not run yet: a full multi-node launch through `launch-cluster.sh --apply-mod`, and an image with FlashInfer's b12x (the proxy patcher was only checked against FlashInfer main's source).

Those runs predate the idle nap described above. After it, the mod was measured again on the same two nodes with the
production clock caps on (CPU 3.0 GHz, GPU 1800 MHz), stock image against the mod's image, one lane start and one
repeat per arm: four greedy 512-token replies one at a time gave 29.38 against 29.05 tok/s mean decode (−1.1%, not
resolved by one repeat) at 98.5 / 89.1 against 86.4 / 84.5 W and 62.4 / 57.8 against 52.7 / 51.7 °C peak; the first
request after a 120 s idle window took 0.198 against 0.206 s to its first token. The full-benchmark numbers in the
table were not repeated after the nap.

Stock ran first, then the mod. These single-run power, temperature and first-token differences are descriptive, not
demonstrated savings, equivalence or latency effects; repeatability and thermal/order effects remain uncharacterized.
Both proxy and queue waits changed, so this comparison does not isolate the proxy.

Related: local-inference-lab/vllm#800, vllm-project/vllm#52916, flashinfer-ai/flashinfer#9352.

