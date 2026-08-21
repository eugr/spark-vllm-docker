# Agent Development Guide

Use this guide when the user asks an agent to inspect, fix, extend, review, or
test this repository. For real host preparation, downloads, builds, or recipe
launches, use `docs/AGENT_RUNBOOK.md` only when the user requests those
operational actions.

## Development Scope

This project is a collection of Bash and Python orchestration scripts for
running vLLM on one or more NVIDIA DGX Spark systems. It is not a Python package
that needs to be installed. Development and most tests do not require a GPU,
Docker daemon, model weights, or a live cluster.

Do not prepare the host for inference or run a real deployment merely to test a
source change. Use mocked tests and explicit recipe dry-runs unless the task also
includes operational verification.

## Start Here

- Read `README.md` for the supported architecture and public workflows.
- Read `recipes/README.md` before creating or changing a recipe.
- Read `docs/NETWORKING.md` before changing cluster networking behavior.
- Check for a `README.md` in any `mods/<name>/` directory you change.
- Run commands from the repository root.
- Inspect `git status --short` before editing. Preserve unrelated user changes
  and local files.

## Development Environment

Recipe validation and tests require:

- Bash and standard GNU/Linux command-line tools
- Python 3.10 or newer
- PyYAML
- `bc` for `tests/test_recipes.sh`

Check them with:

```bash
python3 --version
python3 -c 'import yaml; print(yaml.__version__)'
command -v bc
```

`run-recipe.sh` attempts to install PyYAML when it is missing. During
development, avoid changing the host Python installation; use an isolated
environment if dependencies are unavailable:

```bash
python3 -m venv /tmp/spark-vllm-agent-venv
source /tmp/spark-vllm-agent-venv/bin/activate
python -m pip install PyYAML
```

## Repository Map

- `run-recipe.sh` / `run-recipe.py`: supported recipe entry point and schema
  implementation
- `recipes/`: declarative model launch configurations
- `launch-cluster.sh`: solo and multi-node container orchestration
- `build-and-copy.sh`: image selection, building, and distribution
- `hf-download.sh`: Hugging Face download and model distribution
- `autodiscover.sh`: host and network topology discovery
- `mods/`: patches or files applied inside launched containers
- `tests/`: mocked integration and focused behavior tests

The implementation in `run-recipe.py` is authoritative if recipe documentation
and code disagree.

## Safe Development Commands

```bash
# Inventory; this lists only top-level recipes
./run-recipe.sh --list

# Validate a solo-capable recipe
./run-recipe.sh qwen3.5-35b-a3b-fp8 --dry-run --solo

# Validate a nested four-node recipe without reading local .env settings
./run-recipe.sh recipes/4x-spark-cluster/minimax-m2.5.yaml \
  --config /dev/null \
  --dry-run -n 10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4
```

Always give automated dry-runs either `--solo` or an explicit dummy `-n` node
list. Otherwise the runner may use `.env` or start network autodiscovery.
Cluster dry-runs still read interface settings from `.env` even with `-n`; use
`--config /dev/null` when validating independently of local configuration.
Nested recipes must be passed by path. The dummy `-n` addresses above are
synthetic test inputs only; operational workflows use autodiscovery by default.

Do not use `--setup`, `--build-only`, `--download-only`, `--force-*`, or
`--discover` for development validation. Do not invoke `build-and-copy.sh` or
`hf-download.sh` unless the task explicitly requires that operational action.

## Creating or Changing Recipes

A recipe currently requires:

```yaml
recipe_version: "1"
name: Human-readable name
container: vllm-node
command: |
  vllm serve org/model --port {port} --host {host}
```

Common optional fields are `description`, `model`, `mods`, `build_args`,
`defaults`, `env`, `cluster_only`, and `solo_only`.

- Put general recipes in `recipes/<name>.yaml`; reserve directories such as
  `3x-spark-cluster/`, `4x-spark-cluster/`, and `8x-spark-cluster/` for recipes
  tied to that topology.
- Use repository-relative mod paths such as `mods/fix-name`.
- Back command placeholders with `defaults` or supported runner overrides. Quote
  YAML values where YAML type coercion could change intent.
- Omit `--distributed-executor-backend`, `--nnodes`, `--node-rank`,
  `--master-addr`, `--master-port`, and `--headless` from new recipe and launch
  commands. The runner and `launch-cluster.sh` supply them automatically.
- Do not make both `cluster_only` and `solo_only` true.
- Prefer `vllm-node` unless a documented alternate build is required.
- Update recipe documentation and `tests/expected_commands.sh` when a documented
  command changes.
- Do not edit `recipes/backups/` unless the task specifically targets archived
  alternatives.
- Validate the declared mode and node count. A two-node dry-run does not validate
  a recipe whose command requires four or eight nodes.

## Tests

For recipe or runner changes, run the CI-equivalent checks:

```bash
./tests/test_recipes.sh -v
./tests/test_launch_cluster_image_sync.sh
./tests/test_launch_cluster_vllm_pr.sh
```

Then dry-run every changed recipe, including nested recipes, with an explicit
mode and intended node count.

Run focused checks for other touched areas:

```bash
# build-and-copy.sh or Dockerfile behavior
./tests/test_build_and_copy.sh

# targeted Python patch behavior
python3 tests/test_vllm_flashinfer_b12x_patch.py

# shell syntax for an edited script
bash -n path/to/script.sh
```

Some mods have dedicated `tests/test_*_mod.sh` scripts. Run the matching test
when editing those mods. These suites use fixtures and mocks; do not substitute a
real deployment for them.

## Implementation Conventions

- Bash is the primary orchestration language. Preserve `#!/bin/bash`, quote
  expansions, and use arrays for commands rather than assembling shell strings.
- Python supports 3.10+. Prefer `pathlib.Path`, argument lists for subprocesses,
  and existing helpers in `run-recipe.py`.
- Keep mods targeted and repeatable. `mods/<name>/run.sh` should apply cleanly or
  detect that it is already applied.
- Preserve executable bits on shell and Python entry points.
- Keep changes scoped. Do not embed credentials, machine-specific IPs, cache
  paths, `.env` values, or generated launch output in tracked files.

## Development Completion Checklist

- The worktree contains only the intended changes plus pre-existing user work.
- Changed recipes load and produce the expected dry-run commands.
- Relevant focused tests pass, or missing dependencies are reported.
- Documentation, recipe fields, mod paths, and command expectations agree.
- No secrets or local `.env` contents appear in the diff or test output.
