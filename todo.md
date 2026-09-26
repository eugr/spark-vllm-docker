# Todo

- Remote compressed NFS store backend for UCM (second `storage_backends`
  target, `Cache|Compress|Posix` over an NFS mount) — user follow-up to the
  local-NVMe store. See `decisions.md` 2026-09-21.
- Runtime verification of `mimo-v2.6-flash-ucm` on the cluster: UCM vLLM
  version-matrix warning vs. image vLLM `main`, UCM + DFlash spec decode, fp8
  KV + connector (untested upstream).
- Bump `uc-manager` pin when 0.8.0 stable ships (adds vLLM 0.28/0.29-era
  patches).
- Carry DFlash spec decode and UCM KV offload onto
  `3x-spark-cluster/mimo-v2.6-flash-pp3` once upstream PP>1 support for spec
  decode / KV connectors is confirmed (add the matching mods then).
- Complete the HF download of MiMo-V2.6-Flash-RL (local snapshot lacks weight
  blobs and `dflash/`) before any real MiMo launch.
- Runtime verification of the PP=3 recipe on three nodes (heterogeneous
  121/121/108 GB memory; no PP>1 prior art for this model).
- Re-check whether `mods/mimo-v2.6-flash` staging is still needed: the official
  vLLM recipe points `--speculative-config model` straight at `<snapshot>/dflash`
  with no sanitized copy, while our mod and tonyd2wild's launcher both fix a
  trailing-comma `dflash/config.json` (keep staging until the checkpoint is
  confirmed fixed; local snapshot currently lacks `dflash/`).
- Runtime verification of `mods/mimo-diffkv-fp8-kv`: fp8 KV pool should drop
  to ~12-14 GiB per the recipe math (his measured 12.6 GiB at 300K) and
  needle tests should pass — development scope forbids real launches.
- Revisit server-side sampling defaults on the MiMo recipes (thinking-off via
  `--default-chat-template-kwargs`, rep-penalty 1.05): decision pending.
- Ops note for restarts: at GMU 0.87+ vLLM's startup probe refuses to launch
  until the previous container's memory (incl. reclaimable page cache) is
  back — his launcher syncs + drops caches first; `mods/drop-caches` exists if
  a launch ever trips this.