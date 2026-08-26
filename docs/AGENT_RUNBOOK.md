# Operational Agent Runbook

Use this runbook when the user asks an agent to **use** this repository: prepare
an inference host, inspect available recipes, configure a cluster, build or
download artifacts, launch a recipe, or check a live vLLM server.

This is not a development guide. Do not edit recipes, scripts, mods, Dockerfiles,
or tests to make an operational command succeed unless the user separately asks
for a repository change. Report source-level problems instead.

## Operating Principles

- Work from the repository root.
- Read `README.md` and `recipes/README.md`; read `docs/NETWORKING.md` before any
  multi-node setup.
- Read changelog in `README.md` because it can contain guidance regarding specific models.
- Inspect before changing host or cluster state.
- Use autodiscovery and the saved `.env` as the default cluster workflow. Do not
  pass node IPs with `-n` unless the user explicitly instructs you to do so or
  the network topology prevents autodiscovery from working.
- Use a recipe dry-run before every real setup or launch.
- Only build images, download models, scan networks, copy to remote nodes, or
  launch containers when the user's request includes that action.
- Preserve existing containers, images, caches, models, `.env` files, and cluster
  configuration. Do not prune, overwrite, stop, or remove them unless requested.
- Never expose credentials or the contents of `.env` in chat, logs, diffs, or
  commands. In particular, `--show-env` can display configured tokens.

## 1. Inspect the Host

Start with read-only checks:

```bash
pwd
uname -m
python3 --version
python3 -c 'import yaml; print(yaml.__version__)'
command -v docker
docker info
nvidia-smi
df -h .
git status --short
```

Recipe execution needs Python 3.10 or newer and PyYAML. Real inference normally
needs a DGX Spark or compatible Linux/aarch64 NVIDIA host, a working NVIDIA
driver, Docker with GPU access, enough disk space for images and model weights,
and network access.

`run-recipe.sh` attempts to install PyYAML with pip when it is missing. Installing
a dependency changes the host and may require network access; do it only as part
of an authorized environment-setup request and follow the host's Python package
policy.

For a cluster, also confirm that the interfaces and passwordless SSH setup match
`docs/NETWORKING.md`. Do not change network interfaces, GPU clocks, SSH settings,
or firewall rules without a specific request.

## 2. Select and Inspect a Recipe

List top-level recipes:

```bash
./run-recipe.sh --list
```

Find topology-specific and other nested recipes, which `--list` does not show:

```bash
rg --files recipes -g '*.yaml' | sort
```

In general, solo and cluster recipes that work on 2x+ clusters
go to `recipes` directory, however recipes that requres 3x or more will be nested
in `recipes/3x-spark-cluster`, `recipes/4x-spark-cluster`, etc.

Inspect the selected YAML before running it. Confirm at least:

- `model` and `container`
- `cluster_only` or `solo_only`
- required node count implied by the path, description, and parallelism flags
- `mods`, `build_args`, environment settings, port, and memory defaults
- whether the model or image requires authenticated access

Do not choose a large model or topology on the user's behalf when the requested
target is ambiguous; recipe choice materially affects downloads, storage, and
cluster use.

## 3. Configure the Target

Solo operation usually does not require cluster configuration.

For cluster operation, reuse an existing valid `.env`. On first setup, or when
the user asks to rediscover the cluster, run:

```bash
./run-recipe.sh --discover
```

Discovery detects the topology and interfaces, scans for SSH-reachable GB10
peers, asks for confirmation, and saves `CLUSTER_NODES`, `COPY_HOSTS`, and
network settings to `.env`. Subsequent recipe commands use that configuration
automatically.

Do not pass `-n`, `--nodes`, or literal node IPs during the normal workflow.
Manual node lists are a fallback only when the user explicitly requests them or
the topology cannot be discovered correctly. Do not infer or invent addresses.

An explicitly selected `--config PATH` is also acceptable when the user already
has a configuration file for the target cluster. Do not overwrite an existing
`.env`. `.env.example` documents the public fields. Real configuration files may
contain `CONTAINER_*` secrets; `COPY_HOSTS` may intentionally differ from
`CLUSTER_NODES` for mesh topologies.

## 4. Preview Without Changing State

For a solo-capable recipe:

```bash
./run-recipe.sh RECIPE_NAME --dry-run --solo
```

For a cluster recipe, first complete autodiscovery and then let the runner use
the saved `.env`:

```bash
./run-recipe.sh RECIPE_NAME_OR_PATH --dry-run
```

If the user selected a non-default configuration file:

```bash
./run-recipe.sh RECIPE_NAME_OR_PATH \
  --config PATH_TO_ENV \
  --dry-run
```

Review the discovered node count and the generated vLLM command, image, mods,
mode, tensor/pipeline parallelism, model length, memory settings, port, and extra
arguments. Do not proceed if they conflict with the requested hardware or
outcome. Run discovery explicitly before an unattended cluster preview; if no
configuration exists, the recipe runner can start interactive autodiscovery.

## 5. Execute the Requested Operation

Full solo setup—prepare the image, download the model, and launch:

```bash
./run-recipe.sh RECIPE_NAME --solo --setup
```

Full cluster setup using the autodiscovered `.env`:

```bash
./run-recipe.sh RECIPE_NAME_OR_PATH --setup
```

Pass `--config PATH_TO_ENV` only when the user selected a non-default saved
configuration.

Useful narrower operations are:

```bash
./run-recipe.sh RECIPE_NAME --solo --build-only
./run-recipe.sh RECIPE_NAME --solo --download-only
```

Use `--force-build` or `--force-download` only when the user asks to replace or
refresh an existing artifact. `--setup` can pull or compile large images,
download large model repositories, copy artifacts over SSH, and launch
containers. Allow enough time and disk space, and communicate progress during
long operations.

Multi-node launches default to the no-Ray backend. Add `--ray` only when the
selected recipe or user request requires it. Do not add undocumented overrides
just to get a launch past validation.

### Environment, volumes, and recipe overrides

Both `run-recipe.sh` and `launch-cluster.sh` accept repeatable `-e VAR=VALUE`
options for container environment variables and repeatable
`-v LOCAL_PATH:CONTAINER_PATH` options for Docker volume mappings:

```bash
./run-recipe.sh RECIPE_NAME \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/models:/models" \
  --setup

./launch-cluster.sh \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  -v "$HOME/models:/models" \
  exec vllm serve /models/MODEL_NAME --port 8000 -tp 2
```

Place launcher options before `exec`. In cluster mode, a volume mapping is
applied on every launched node, so the host path must be valid and appropriate
on each node.

For gated Hugging Face models, export the token so host-side downloads can use
it, and pass it into the container with `-e`:

```bash
export HF_TOKEN="YOUR_TOKEN"
./run-recipe.sh RECIPE_NAME -e HF_TOKEN="$HF_TOKEN" --setup

# Or for a direct launch
./launch-cluster.sh -e HF_TOKEN="$HF_TOKEN" exec vllm serve MODEL_NAME -tp 2
```

Never print or record the expanded token. Confirm that debugging or dry-run
output will not be shared before placing a secret in a command argument.

`run-recipe.sh` has first-class overrides for `--port`, `--host`, `--tp` /
`--tensor-parallel`, `--gpu-mem` / `--gpu-memory-utilization`, and
`--max-model-len`. Put arbitrary additional vLLM arguments after `--`:

```bash
./run-recipe.sh RECIPE_NAME \
  --port 9000 \
  --gpu-mem 0.85 \
  -- --max-num-seqs 16 --enforce-eager
```

Arguments after `--` are appended to the generated `vllm serve` command. If an
option appears more than once, vLLM uses the last occurrence; the runner warns
when it detects a duplicate of a first-class override. Prefer the first-class
option unless deliberately relying on last-one-wins behavior.

## 6. Direct Launch and Distribution Workflows

### Run `vllm serve` directly

Recipes are preferred when one exists, but a normal vLLM command can be run
through the cluster launcher:

```bash
./launch-cluster.sh exec vllm serve MODEL_NAME \
  --host 0.0.0.0 \
  --port 8000 \
  -tp 2
```

Do not pass node IPs for the normal cluster workflow. `launch-cluster.sh` loads
the default `.env` when present and performs autodiscovery when it is absent. Use
`./launch-cluster.sh --setup` to force discovery and save the result before a
later launch. Use `--solo` for a deliberately single-node command.

Pass model and inference options, including the intended `-tp`, `-pp`, or `-dp`
parallelism, but do not add multiprocessing orchestration parameters to the
`vllm serve` command. In particular, omit:

- `--distributed-executor-backend`
- `--nnodes`
- `--node-rank`
- `--master-addr` and `--master-port`
- `--headless`

The launcher derives the required node count from the parallelism flags. In its
default no-Ray mode it starts the worker and head commands with the correct
node/rank/master parameters. With launcher option `--ray`, it starts Ray and
adds `--distributed-executor-backend ray` when needed. If a non-default
coordination port is required, pass `--master-port` to `launch-cluster.sh`
before `exec`, not inside the vLLM arguments.

Some models may require non-default or 3rd party container image. In this case, 
you need to specify the container using `-t` argument. For example, if B12X kernels
are utilized, one needs to specify `-t vllm-node-b12x`.

The launch script supports 3rd-party containers, such as vLLM official or NVIDIA NGC, 
however their usage is recommended only to advanced users or reserved to special cases.

### Distribute the image and model

When operating without recipe `--setup`, distribute the selected image and
model explicitly:

```bash
# Prepare the image and copy it to all COPY_HOSTS concurrently
./build-and-copy.sh -c --copy-parallel

# If recipe/vLLM command requires B12X kernels, prepare and copy b12x image
./build-and-copy.sh --exp-b12x -c --copy-parallel

# Download/reuse the model and copy it to all COPY_HOSTS concurrently
./hf-download.sh MODEL_NAME -c --copy-parallel
```

`-c` / `--copy-to` opts into distribution. With no addresses after it, each
script uses `COPY_HOSTS` from `.env` and falls back to autodiscovery. Do not add
addresses unless instructed or autodiscovery cannot support the topology.
`--copy-parallel` copies to all resolved hosts concurrently. To distribute an
already prepared local image without pulling or building it again, use:

```bash
./build-and-copy.sh --no-build -c --copy-parallel
```

You can also distribute any non-default/3rd-party container the same way by providing `-t` argument:

```bash
docker pull vllm/vllm-openai:nightly
./build-and-copy.sh -t vllm/vllm-openai:nightly --no-build -c --copy-parallel
```

Recipe `--setup` already coordinates image and model distribution; do not repeat
these lower-level commands afterward unless verification shows a missing or
mismatched artifact.

### Troubleshoot model-copy permissions

If `hf-download.sh` or `rsync` reports permission errors below `~/.cache`, check
ownership and write access on the head and every worker. If the cache is supposed
to belong to the login user, run the following locally on **each node**:

```bash
sudo chown -R "$USER" "$HOME/.cache"
```

This is a broad, privileged, recursive ownership change. Run it only after
confirming that root-owned cache files are the cause and the user authorizes the
repair. If `HF_HOME` points elsewhere, repair that configured cache path instead.
Do not run the command automatically over SSH or hide it inside a normal model
download.

## 7. Verify the Result

Use the selected recipe's host and port. For the common local port 8000:

```bash
docker ps
curl --fail --show-error http://127.0.0.1:8000/v1/models
```

If the server is still starting, inspect bounded logs for the launched container
instead of repeatedly restarting it:

```bash
docker logs --tail 100 vllm_node
```

The container name can be changed with `--name`; do not assume `vllm_node` when
an override or existing configuration is present. A successful container start
is not sufficient verification: confirm that the API becomes ready and reports
the expected served model.

## 8. Handle Failures Conservatively

- Re-run the same recipe with `--dry-run` and the same mode, config, overrides,
  and node count to capture the intended command.
- Check the exact image, model access, available disk/memory, Docker/NVIDIA
  status, port conflicts, and SSH reachability relevant to the error.
- For cluster failures, compare every node's image and connectivity before
  relaunching.
- Do not edit project source, weaken safety settings, prune Docker data, delete
  caches, or rebuild with experimental flags unless the user authorizes that
  separate action.
- Preserve the failing logs and report the concrete command phase and error.

## Operational Handoff

Report:

- recipe name and path
- solo or cluster mode and node count (avoid disclosing private details unless
  needed)
- selected image, model, and material overrides
- whether image preparation, download, distribution, and launch completed
- API endpoint and readiness result
- any remaining error or manual prerequisite

Do not include tokens, `.env` contents, authentication headers, or unrelated
machine configuration.
