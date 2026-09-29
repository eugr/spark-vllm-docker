# Model startup memory profiles

This opt-in mod records one model/recipe run on regular and B12X vLLM images.
It measures each GPU worker, API process, and physical host, and writes a YAML
profile card plus JSONL traces suitable for plotting. It observes the existing
startup sequence, including any built-in heap trimming; it does not collect
garbage, trim allocators, reset peaks, change KV sizing, or run inference itself.

## Launch and retain the measurements

```bash
mkdir -p "$PWD/memory-profiles"
./run-recipe.sh qwen3.8-flash-next-nvfp4-solo --solo \
  --apply-mod mods/memory-profile \
  -v "$PWD/memory-profiles:/memory-profiles" \
  -e VLLM_MEMORY_PROFILE_RUN_ID=qwen-baseline-01 \
  -e VLLM_MEMORY_PROFILE_RECIPE=qwen3.8-flash-next-nvfp4-solo
```

Preview the command with `--dry-run` before a real launch. Apply this mod last
if other mods replace vLLM or alter its startup functions. A recipe can opt in
with `mods: [mods/memory-profile]`; the usual `-v` and `-e` launcher options
still control output and the run label. No existing recipes enable it by default.

Output is grouped as:

```text
memory-profiles/<run-id>/<host-id>/
  manifest.json       # settings, source hashes, container hostname, driver
  host.jsonl          # host RAM, cgroup counters, visible-process PSS samples
  events-<pid>.jsonl  # per-process phase measurements, rank/config metadata
  profile.yaml       # local card, refreshed every five seconds
  monitor.log
```

The model ID comes from resolved vLLM configuration. The recipe label is optional
and user supplied. Every run gets a separate directory; an existing run/host
directory is refused rather than overwritten. Reapplying the same mod in the
same container preserves its original run and uses one sampler.

Bind-mount the output directory: the launcher removes containers when they stop.
Without a mount, collect the artifacts before stopping the container.

## Multiple hosts and aggregation

For a cluster recipe, use the normal discovered cluster and pass the same
`VLLM_MEMORY_PROFILE_RUN_ID` on every node. The launcher distributes the mod and
environment flags. The default host ID is a hash of the Linux boot ID, so ranks
on one physical host share one host record. The output mount path must exist on
each node; it can be separate local storage or a shared filesystem.

On shared storage, or after copying the run directories onto one machine:

```bash
python3 mods/memory-profile/profile_card.py \
  memory-profiles/qwen-baseline-01 \
  --output memory-profiles/qwen-baseline-01.yaml
```

For separate node-local storage, the collector can retrieve files directly from
running containers using existing Docker/SSH access:

```bash
python3 mods/memory-profile/collect.py \
  --host local --host worker-ssh-alias \
  --container vllm_node \
  --run-id qwen-baseline-01 \
  --output memory-profiles/collected-qwen-baseline-01
```

Use the actual container name and repeat `--host` for every node. The collector
only copies profiling files and creates `profile.yaml`; it does not launch or
stop servers or change SSH configuration. Its output directory must be new.
If collection fails, files already collected remain available. Shared storage
needs only one copy of each physical host's files; duplicate host records are
rejected. Different runs, models, or parallel topologies cannot be merged.

A local card on a multi-host run is incomplete until the other ranks are merged.
The card lists expected, recorded, ready, missing and duplicate ranks. Failures
and instrumentation errors remain visible. A ready worker does not by itself
establish API readiness. Offline/headless-only runs retain worker measurements
but have no API readiness marker.

## Readable reports and startup charts

`report.py` reads a YAML or JSON profile card directly; raw JSONL files, vLLM,
CUDA and a running container are not required. Print Markdown to stdout:

```bash
python3 mods/memory-profile/report.py memory-profiles/qwen-baseline-01.yaml
```

Write a Markdown report and an accompanying PNG chart:

```bash
python3 mods/memory-profile/report.py memory-profiles/qwen-baseline-01.yaml \
  --output memory-profiles/qwen-baseline-01.md
```

The chart requires `matplotlib>=3.6`; text output only requires PyYAML. If needed,
install the dependencies in a separate environment:

```bash
python3 -m venv /tmp/memory-profile-report-venv
/tmp/memory-profile-report-venv/bin/python -m pip install PyYAML 'matplotlib>=3.6'
# Use /tmp/memory-profile-report-venv/bin/python to run report.py.
```

- `--plot PATH.png` chooses the chart path; SVG and PDF are also supported.
  It can be used with stdout output. With `--output`, the default chart is the
  Markdown filename with a `.png` extension, linked relative to the report.
- `--no-plot` writes only Markdown, without importing matplotlib.
- `--relative` plots growth instead of absolute GiB. Host RAM uses the pre-vLLM
  baseline; each process counter uses its first available measurement. Tables
  always show absolute measurements. CPU growth curves stay separate because
  the processes' baselines occur at different times.

Each host gets separate panels for system RAM (`MemTotal - MemAvailable`), CPU
PSS stacked by rank/API process, and CUDA allocated/reserved bytes by rank. CPU
PSS has its own scale so smaller CPU allocations remain visible. The CPU stack
carries each process's last observation forward to align asynchronous checkpoints.
Its top edge is an **estimated sum for the recorded processes**, not a simultaneous
host measurement. The total remains unavailable until every contributor has a
sample; missing samples invalidate it until the affected process is measured
again. Gray shading marks those gaps. Known lower bands can remain visible.
Processes absent from an incomplete card, compiler processes and the sampler
are excluded from this sum. The report also lists estimated combined CPU PSS
from rank/API readiness snapshots for each host.

Model-loaded, KV-allocated
and worker-ready checkpoints have distinct markers. Host clocks are independent.

Graphs use the card's startup checkpoints, not continuous samples. Connecting
lines are visual guides; no CPU/GPU values are invented before a process's first
measurement. The API hook runs late in startup, so its CPU band begins then.
CPU step bands show the latest samples, not interpolation between checkpoints.
The sampled host peak is shown as a
reference level; the card does not retain its precise timestamp. Host/CPU/CUDA
categories are never stacked or added together on unified memory systems; only
CPU PSS of different processes on the same host is added.

The report includes coverage/failure flags, readiness and peak footprints,
model/KV storage, KV bytes per 1,000 equivalent capacity tokens, graph deltas,
CPU heap counters, the initial utilization check, and the checkpoint table.

## Hardware requirements and current cluster fit

`capacity.py` produces a concise Markdown requirements report from a YAML or
JSON card. It needs Python and PyYAML, but not vLLM, Torch or matplotlib:

```bash
# Offline: original allocation/context and a smaller cache for one 128K sequence.
python3 mods/memory-profile/capacity.py memory-profiles/qwen-baseline-01.yaml

# On the head: also check current memory on the required nodes.
python3 mods/memory-profile/capacity.py memory-profiles/qwen-baseline-01.yaml \
  --check-host --output memory-profiles/qwen-capacity.md

# Plain console report to stdout (add --check-host for a live assessment).
python3 mods/memory-profile/capacity.py memory-profiles/qwen-baseline-01.yaml \
  --format console
```

The report shows the original context, parallelism, scheduling limits, GPU,
KV dtype and speculation settings. Requirements are **per physical host**:
available RAM needed at launch, and total RAM assuming the recorded background
footprint. The smaller-cache example supplies `--max-model-len`,
`--max-num-seqs 1`, explicit **per-rank** `--kv-cache-memory-bytes`, and an
automatic-sizing equivalent for `--gpu-memory-utilization`. Keep the remaining
recipe settings. Explicit KV bytes control cache size independently of utilization;
the profiled worker still checks utilization during initial memory admission.
To let utilization determine cache size, omit `--kv-cache-memory-bytes`.
This estimates a minimum cache configuration on the **same topology**, not the
minimum number of GPUs or a guarantee that a new configuration will start.
Preserving the original cache allocation on a different RAM size also requires
retuning utilization or setting an explicit KV budget; the total-RAM estimate
does not imply that unchanged automatic-sizing flags reproduce the same cache.

The RAM labels distinguish the host's current state from the model's requirement:

- **Available now** is Linux `MemAvailable`: the kernel's estimate of capacity
  usable without swapping. It includes unused memory plus the reclaimable
  portion of file and kernel caches, while allowing for kernel reserves.
  `MemFree` measures only unused pages. See the [Linux memory counters](https://docs.kernel.org/filesystems/proc.html#meminfo).
- **Required available RAM** is the estimated capacity that must be available
  **before launching** the model, including its startup peak/admission requirement
  and the requested reserve. Compare this number with the live `MemAvailable`.
- **Total RAM** in the requirements table adds the recorded OS/background
  footprint to required available RAM. For example, 52.154 GiB required available
  plus 5.239 GiB background gives 57.393 GiB estimated total RAM.

On Spark this is a shared physical CPU/GPU memory pool. Do not add a separate
VRAM allowance to these host requirements. Swap does not count as available RAM
in this check, and the estimate does not assume that existing workloads stop.

The utilization table distinguishes three targets:

- Preserve the recorded KV allocation and the original `max-num-seqs` setting.
- Provision KV for `max-num-seqs` simultaneous sequences **each at the full
  profiled context**, including estimated block slack.
- Provision the smaller target context (128K by default) for one sequence.

`max-num-seqs` is a scheduling limit. A successfully started server can have less
KV than needed for that many full-length sequences. The report also prints the
recorded group-aware full-context concurrency to make this visible.

For each rank, the calculation is:

```text
sizing_cost = original_requested_bytes - measured_KV_budget
minimum_utilization = (sizing_cost + desired_KV_bytes) / device_total_bytes
```

The common setting uses the largest ratio across ranks and rounds **up** to
0.001. These are estimates with the profiled non-KV overhead retained, not
remeasured minima for different scheduling settings. With `--check-host`, both
the ratios and suggested settings use the checked devices' memory sizes.

The utilization denominator is the device's full reported memory, not the
report's estimated total-RAM requirement. vLLM's cache-sizing budget and whole
host startup RAM account for different costs. The host estimate is:

```text
adjusted_peak = max(pre_KV_peak, post_KV_peak - old_KV + new_KV)
available_RAM = max(adjusted_peak, initial_gate_requirement) + reserve
total_RAM = recorded_background_RAM + available_RAM
```

Peaks above are increments from the pre-vLLM baseline. For example, a 27.47 GiB
sizing budget on a 121.69 GiB device corresponds to utilization about 0.226;
the same configuration can need over 50 GiB of available host RAM during
startup. Increasing utilization to the host-RAM fraction would allocate more KV
in automatic mode. See [vLLM's cache sizing options](https://docs.vllm.ai/en/stable/configuration/engine_args/#--kv-cache-memory-bytes)
for the explicit-KV override behavior.

JSON output includes each rank's sizing cost, desired KV, device total, minimum
ratio, limiting rank, and rounding result in `utilization_estimates`. Host rows
also include `target_kv_breakdown` (full-attention, retained state/window, slack
and null blocks) and `target_host_memory_breakdown` for auditing the arithmetic.

With `--check-host`, a solo card checks only the local machine. A cluster card
uses the saved `.env` and the first required nodes in `CLUSTER_NODES`, like the
launcher; extra nodes are ignored. The script verifies that it is on the
configured head. Use `--config PATH` for another saved configuration. Explicit
mapping is available as `--host local --host worker-ssh-alias`, in rank order.
Hosts are never taken from untrusted labels in the profile card.

The probe reads `/proc/meminfo` and CUDA driver properties without creating a
CUDA context. It uses noninteractive SSH with timeouts for peers. It does not
run discovery, import Torch, launch containers, change clocks, stop workloads,
or modify `.env`. An already-running model counts as occupied memory: the
result answers whether there is room for an **additional launch now**.

Options:

- `--context N`: reduced-cache target; default **131072**, meaning 128K tokens
  for one sequence, including prompt and generated tokens. Extrapolating above
  the profiled context is unsupported.
- `--reserve-gib N`: additional reserve per host; default **4 GiB**. This is
  separate from the recorded loading and warmup peaks.
- `--format markdown|console|json`: output format; default `markdown`. Console
  uses aligned plain-text tables and wrapped paragraphs, without Markdown markup
  or terminal colors. It works with stdout, redirection, pagers and `--output`.
- `--json`: alias for `--format json`.
- `--output PATH`: save the report; otherwise print to stdout.

The live verdicts are **LIKELY FITS**, **DOES NOT FIT**, or **UNKNOWN**, separately
for the original cache/settings, the target context with tuned KV settings,
and a small usable cache at a reduced context. If the original utilization
would create less KV than the recorded allocation, the original-profile test
fails even if a smaller cache could still serve requests. On larger hosts it
also accounts for extra KV allocated by the original utilization setting.
The printed utilization for the target uses the checked hosts' RAM sizes.
Exit codes are `0` for a successful offline report or a likely original-profile
fit, `1` when that live fit fails (even if reduced KV fits), and `2` for unknown,
unsupported, incomplete, or invalid input.

The estimator retains the pre-KV loading/profiling peak. Only the post-KV peak
is adjusted by the change in cache size, and the initial utilization admission
check is evaluated separately. CPU PSS and CUDA counters are never added to
host RAM. Swap and spare memory on another host do not satisfy a rank's needs.

For supported shared-pool layouts, it recovers blocks per full-context request
from the card's block count and group-aware concurrency. Only ordinary full
attention blocks shrink with the target context; the measured Mamba/window
cost stays fixed, with extra block and speculative lookahead slack. This follows
the pool/concurrency accounting in [vLLM's KV cache utilities](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py).
It avoids treating a hybrid model's average bytes/token as an exact marginal
cost. These are conservative estimates, **not exact minima**. Between the
non-KV startup floor and the conservative small-cache bound, the answer remains
unknown because a tighter cache may fit.

Current support is native Linux UMA, one GPU rank per host, TP/PP with
DP=DCP=PCP=1, matching GPU name/architecture, and complete startup coverage.
The source card must use automatic KV sizing: an explicit-KV run skips sizing
cost profiling, so subtracting its fixed cache budget from the initial request
cannot establish a utilization minimum.
Resizing supports verified uniform shared pools containing `FullAttentionSpec`,
`MambaSpec`, and `SlidingWindowSpec`. Other layouts can still reuse the measured
allocation at the same RAM size, but reduced-cache estimates remain unknown.
Discrete GPUs, WSL, changed topology and unsupported layouts need a new or richer
profile. Software versions, image identity, checkpoint availability, transport
and maximum serving workload are assumptions, not verified by this memory check.

## What is measured

- Exact free/total snapshot passed to the initial `gpu_memory_utilization`
  admission check, and the requested bytes if that check succeeds.
- Checkpoints around device initialization, model loading, activation profiling,
  graph estimation/capture, KV allocation, and final warmup.
- PyTorch CUDA allocated/reserved bytes and its existing peak counter, without
  resetting that counter. The sampler never imports Torch or initializes CUDA.
- Unique registered model parameter/buffer storage at readiness, including
  known target/draft model roots. Shared storages are deduplicated.
- Unique KV backing storage, resolved cache groups/block sizes, allocated
  blocks, equivalent token capacity, and effective bytes per 1,000 capacity
  tokens. Aliased hybrid cache views are counted once. Capacity is obtained
  from vLLM's group-aware calculation when available; otherwise it is null.
  Repeated cache-group layouts are grouped by count in the card. The raw
  `configured_tensor_bytes` value sums logical descriptors, which can alias;
  only the deduplicated storage is a physical footprint.
- Per-process CPU PSS/RSS, anonymous/file/shared pages, glibc heap statistics,
  and PyTorch pinned-host allocator counters when supported. API startup and
  both sides of the existing startup GC/trim hook are included.
- Host `MemTotal - MemAvailable`, sampled startup peak and increment from the
  pre-vLLM baseline, readiness footprint, and cgroup counters in the raw trace.
  Compiler subprocesses contribute to the visible-process PSS samples.
- Selected resolved memory settings, model revision when exposed by the model
  config, package versions, GPU properties, source hashes and selected allocator
  environment settings. Command lines, requests, API keys, and arbitrary
  environment/configuration dumps are not recorded.

## Settings

Pass these with the launcher's `-e` option:

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_MEMORY_PROFILE_DIR` | `/memory-profiles` | Absolute container output path |
| `VLLM_MEMORY_PROFILE_RUN_ID` | UTC timestamp plus random suffix | Shared run ID; set explicitly for clusters |
| `VLLM_MEMORY_PROFILE_RECIPE` | unset | Recipe label for the card |
| `VLLM_MEMORY_PROFILE_IMAGE` | unset | Optional image digest/identity, supplied by the caller |
| `VLLM_MEMORY_PROFILE_HOST_ID` | hash of Linux boot ID | Optional unique physical-host label |
| `VLLM_MEMORY_PROFILE_INTERVAL` | `0.5` | Host sampling interval, seconds |
| `VLLM_MEMORY_PROFILE_PROCESS_INTERVAL` | `5` | PSS sampling interval, seconds |
| `VLLM_MEMORY_PROFILE_DURATION` | `3600` | Maximum sampler lifetime, seconds |
| `VLLM_MEMORY_PROFILE_SYNC` | `1` | Synchronize CUDA at phase checkpoints; `0` avoids those waits |

Container-created run directories can be owned by root; write merged cards into
your own output directory as in the example above. The generated local YAML
and traces are readable by the host user.

To stop a sampler, create `STOP` in its host output directory (using
`docker exec <container> touch <host-output-directory>/STOP` if needed). The sampler
writes a final local card and exits. In-process hooks remain until that container
exits. Later host samples remain in JSONL; YAML startup peaks exclude samples
after local readiness. The sampler must cover startup to mark the card complete.

## Interpretation and limits

The YAML is a standalone versioned profile card; the recipe runner does not yet
use it for automatic admission or tuning. It records measured startup behavior,
not a guarantee that a different host or a maximum serving workload will fit.
Reprofile meaningful changes in model revision, TP/PP/DP, context, batch size,
speculation, quantization, graph configuration, kernel versions, or cache state.

On Spark, CPU and GPU share physical memory: **do not add host RAM to CUDA
counters**, and do not add the same host observation once per rank. Host totals
include unrelated activity. PSS and cgroup counters can omit GPU-backed pages;
process-namespace PSS includes the small profiling observer. Host peak samples
are lower bounds, and host baselines are taken independently on each node.

Model storage covers registered tensors; native backend weights, driver memory,
and workspaces can be outside that inventory. Non-KV and other-allocation values
are derived residuals. Graph capture reports device-wide memory deltas, not an
isolated graph-object size. A hybrid model's KV bytes per 1,000 equivalent capacity
tokens is not a universal marginal cost per user token.

The mod introduces sampling, synchronization and storage-inventory overhead, so
startup timing is not a benchmark. It never introduces distributed collectives.
Runtime observation failures warn and preserve the original vLLM operation or
exception; source layouts that lack the required hooks are rejected before
patching. Regular and B12X V1 worker layouts, including their V2 model runners,
are supported. Other engines/backends are not covered by this mod.

Run the CPU-only regression suite with:

```bash
python3 tests/test_memory_profile_mod.py
python3 tests/test_memory_capacity.py
```
