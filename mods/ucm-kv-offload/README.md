# UCM KV cache offload mod

Runtime mod that installs [unified-cache-management (UCM)](https://github.com/ModelEngine-Group/unified-cache-management)
and writes its connector config so vLLM offloads KV cache blocks to local
storage. Used by `recipes/mimo-v2.6-flash-ucm.yaml`; the serve flag and the
patch-hook environment variable live in that recipe because mods cannot set
the serve process environment.

## What it does

- Installs `uc-manager[cu130]==0.7.0` from PyPI with `uv` (pinned; skipped when
  already present), then verifies `import ucm`. The image is CUDA 13.0 /
  Python 3.12, which matches the published `cu130` cp312 wheels.
- Writes `/workspace/ucm_config.yaml` with an uncompressed `Cache|Posix`
  pipeline store: GPU -> host pinned buffers -> POSIX files under
  `storage_backends`.
- Creates the store directory.

The default store path `/root/.cache/vllm/ucm-kv` sits inside the host-mounted
vLLM cache directory (`launch-cluster.sh` mounts `~/.cache/vllm` by default),
so offloaded KV persists across container recreation with no extra `-v` flag.
With `--no-cache-dirs` the store still works but lives in the container layer.

## Environment overrides (container env, e.g. launch-cluster.sh `-e`)

- `UCM_KV_DIR`: store directory (default `/root/.cache/vllm/ucm-kv`).
  To use a dedicated mount: `-v "$HOME/ucm-kv:/ucm-kv" -e UCM_KV_DIR=/ucm-kv`.
- `UCM_KV_CAPACITY_GB`: advertised posix store capacity (default `10240`).
- `UCM_CONFIG_FILE`: config path written by the mod (default
  `/workspace/ucm_config.yaml`); the recipe's `--kv-transfer-config` hardcodes
  the default, so only override both together.

## Known limits

- UCM 0.7.0 ships version-specific patches for vLLM up to 0.28.0; the image
  builds vLLM `main`, so UCM logs a warning and relies on the stock
  `KVConnectorBase_V1` API beyond that. Runtime verification on the cluster is
  a follow-up (see `todo.md`).
- UCM + DFlash speculative decoding + fp8 KV is an untested combination
  upstream.

UCM's own log files land under `log/` in the directory where `vllm serve`
starts; override with `UCM_LOG_PATH`.