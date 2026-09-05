#!/bin/bash
#
# test_stack.sh - Tests for run-stack.py (multi-recipe co-location).
#
# Uses --dry-run and validation paths only; never launches containers.
# Suitable for CI.
#
# Usage:
#   ./tests/test_stack.sh        # run all tests
#   ./tests/test_stack.sh -v     # verbose
#

set +e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_STACK="$PROJECT_DIR/run-stack.py"
VERBOSE="${1:-}"

# Shared harness: colors, counters, log_*/assert_contains helpers
source "$SCRIPT_DIR/lib.sh"

TMPDIR_STACK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_STACK"' EXIT

# An empty .env: forces solo-local resolution regardless of the developer's
# own .env (this repo's real .env sets CLUSTER_NODES). Created once up front
# so every solo-regression test below can force it explicitly.
: > "$TMPDIR_STACK/empty.env"

echo "========================================"
echo "run-stack.py tests"
echo "========================================"

# ---- 1. --list surfaces the example stack ----
log_test "--list includes example-dual-stack"
OUT="$("$RUN_STACK" --list 2>&1)"
assert_contains "example-dual-stack listed" "$OUT" "example-dual-stack"

# ---- 2. example-dual-stack dry-run: order, names, ports ----
log_test "example-dual-stack --dry-run launch plan"
OUT="$("$RUN_STACK" example-dual-stack --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
log_verbose "$OUT"
## --config is forwarded to each run-recipe.py invocation (test 23 covers this
## explicitly), so it lands between --gpu-mem and the -- extra_args separator;
## check the recipe-args prefix and the extra_args suffix as two substrings
## (same style as test 6) instead of one contiguous string.
assert_contains "entry 1 is qwen (largest first)" "$OUT" \
    "1. ./run-recipe.py qwen3.6-35b-a3b-nvfp4 --solo --name qwen_nvfp4 -d --port 8000 --gpu-mem 0.45"
assert_contains "entry 2 is nemotron" "$OUT" \
    "2. ./run-recipe.py nemotron-3-nano-nvfp4 --solo --name nemotron_nano -d --port 8001 --gpu-mem 0.4 --config"
assert_contains "extra_args after --" "$OUT" "-- --load-format instanttensor"
assert_contains "health gate on port 8000" "$OUT" "http://localhost:8000/health"
assert_contains "health gate on port 8001" "$OUT" "http://localhost:8001/health"
# Order check: qwen must appear before nemotron in the output.
if [[ "$OUT" == *"qwen_nvfp4"*"nemotron_nano"* ]]; then
    log_pass "load order preserved (qwen before nemotron)"
else
    log_fail "load order preserved (qwen before nemotron)"
fi

# ---- 3. duplicate port aborts before launching ----
log_test "duplicate port is rejected"
cat > "$TMPDIR_STACK/dup-port.yaml" <<'EOF'
name: dup-port
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
  - {recipe: nemotron-3-nano-nvfp4, container_name: b, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/dup-port.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"port 8000"* && "$OUT" == *"already used"* ]]; then
    log_pass "duplicate port errors with nonzero exit"
else
    log_fail "duplicate port errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 4. duplicate container_name aborts ----
log_test "duplicate container_name is rejected"
cat > "$TMPDIR_STACK/dup-name.yaml" <<'EOF'
name: dup-name
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
  - {recipe: nemotron-3-nano-nvfp4, container_name: a, port: 8001}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/dup-name.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"duplicate container_name"* ]]; then
    log_pass "duplicate container_name errors with nonzero exit"
else
    log_fail "duplicate container_name errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 5. port defaults from the recipe when omitted ----
log_test "omitted port inherits recipe defaults.port"
cat > "$TMPDIR_STACK/default-port.yaml" <<'EOF'
name: default-port
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: only}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/default-port.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
# qwen3.6-35b-a3b-nvfp4 recipe declares defaults.port: 8000
assert_contains "default port 8000 applied" "$OUT" "--port 8000"

# ---- 6. mods resolve to absolute paths; extra_args land after `--` ----
log_test "mods + extra_args rendering"
cat > "$TMPDIR_STACK/mods.yaml" <<'EOF'
name: mods
recipes:
  - recipe: qwen3.6-35b-a3b-nvfp4
    container_name: q
    port: 8000
    mods: [mods/gpu-mem-util-gb]
    extra_args: ["--gpu-memory-utilization-gb", 60]
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/mods.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
log_verbose "$OUT"
assert_contains "mod resolved to absolute path" "$OUT" \
    "--apply-mod $PROJECT_DIR/mods/gpu-mem-util-gb"
assert_contains "extra_args after --" "$OUT" \
    "-- --gpu-memory-utilization-gb 60"

# ---- 7. nonexistent recipe is rejected ----
log_test "nonexistent recipe is rejected"
cat > "$TMPDIR_STACK/bad-recipe.yaml" <<'EOF'
name: bad-recipe
recipes:
  - {recipe: this-recipe-does-not-exist, container_name: a, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/bad-recipe.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"not found"* ]]; then
    log_pass "nonexistent recipe errors with nonzero exit"
else
    log_fail "nonexistent recipe errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 8. malformed YAML gives a clean error, not a traceback ----
log_test "malformed YAML is rejected cleanly"
printf 'recipes: [\n  {broken' > "$TMPDIR_STACK/malformed.yaml"
OUT="$("$RUN_STACK" "$TMPDIR_STACK/malformed.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"Error:"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "malformed YAML errors without a traceback"
else
    log_fail "malformed YAML errors without a traceback (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 8b. stack YAML whose top level is not a mapping is rejected cleanly ----
log_test "non-mapping stack YAML is rejected cleanly"
printf -- '- just\n- a list\n' > "$TMPDIR_STACK/list-stack.yaml"
OUT="$("$RUN_STACK" "$TMPDIR_STACK/list-stack.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"Error:"* && "$OUT" == *"mapping"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "list-shaped stack YAML errors without a traceback"
else
    log_fail "list-shaped stack YAML errors without a traceback (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 9. invalid container_name is rejected ----
log_test "invalid container_name is rejected"
cat > "$TMPDIR_STACK/bad-name.yaml" <<'EOF'
name: bad-name
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: "has space", port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/bad-name.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"not a valid Docker container name"* ]]; then
    log_pass "invalid container_name errors with nonzero exit"
else
    log_fail "invalid container_name errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 10. unsupported stack_version is rejected ----
log_test "unsupported stack_version is rejected"
cat > "$TMPDIR_STACK/future.yaml" <<'EOF'
stack_version: "99"
name: future
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/future.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"unsupported stack_version"* ]]; then
    log_pass "unsupported stack_version errors with nonzero exit"
else
    log_fail "unsupported stack_version errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 11. non-integer health_timeout names the field ----
log_test "non-integer health_timeout is rejected"
cat > "$TMPDIR_STACK/bad-timeout.yaml" <<'EOF'
name: bad-timeout
health_timeout: soon
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/bad-timeout.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"health_timeout"* ]]; then
    log_pass "bad health_timeout errors naming the field"
else
    log_fail "bad health_timeout errors naming the field (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 12. --dry-run conflicts with --stop ----
log_test "--dry-run cannot be combined with --stop"
OUT="$("$RUN_STACK" example-dual-stack --stop --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"not allowed with"* ]]; then
    log_pass "--stop --dry-run is rejected"
else
    log_fail "--stop --dry-run is rejected (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 13. derived container names keep dotted recipe names intact ----
log_test "shorthand entry derives the full recipe name"
cat > "$TMPDIR_STACK/shorthand.yaml" <<'EOF'
name: shorthand
recipes:
  - qwen3.6-35b-a3b-nvfp4
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/shorthand.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
assert_contains "derived name is not truncated at the first dot" "$OUT" \
    "--name qwen3.6-35b-a3b-nvfp4"

# ---- 14. --dry-run reflects --setup ----
log_test "--dry-run --setup shows the --setup flag"
OUT="$("$RUN_STACK" example-dual-stack --dry-run --setup \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"; RC=$?
# Scope the needle to the generated command lines: argparse's own
# "unrecognized arguments: --setup" error would satisfy a bare-string match,
# turning this test false-green if the flag were removed from the parser.
PLAN_LINES="$(echo "$OUT" | grep './run-recipe.py')"
if [[ $RC -eq 0 && -n "$PLAN_LINES" && "$PLAN_LINES" == *"--setup"* ]]; then
    log_pass "planned run-recipe.py commands include --setup"
else
    log_fail "planned run-recipe.py commands include --setup (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 15. missing recipe field is rejected ----
log_test "entry without 'recipe' is rejected"
cat > "$TMPDIR_STACK/no-recipe.yaml" <<'EOF'
name: no-recipe
recipes:
  - {container_name: a, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/no-recipe.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"must have a 'recipe'"* ]]; then
    log_pass "missing recipe field errors"
else
    log_fail "missing recipe field errors (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- fixtures reused by several tests below: a cluster .env and a
# cluster-capable stack (placement resolution replaces the old stack-wide
# Mode: SOLO/CLUSTER banner tests; see P1-P6 for placement coverage) ----
cat > "$TMPDIR_STACK/cluster.env" <<'EOF'
CLUSTER_NODES="10.0.0.1,10.0.0.2"
LOCAL_IP="10.0.0.1"
EOF
cat > "$TMPDIR_STACK/clusterable.yaml" <<'EOF'
name: clusterable
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
EOF

# ---- 19. master_port auto-assigns and can be overridden ----
log_test "master_port auto-assignment and override"
cat > "$TMPDIR_STACK/mports.yaml" <<'EOF'
name: mports
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8001}
  - {recipe: qwen3.5-35b-a3b-fp8, container_name: c, port: 8002, master_port: 29600}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/mports.yaml" --dry-run \
        --config "$TMPDIR_STACK/cluster.env" 2>&1)"
log_verbose "$OUT"
assert_contains "entry 1 auto master port" "$OUT" "--master-port 29501"
assert_contains "entry 2 auto master port" "$OUT" "--master-port 29502"
assert_contains "entry 3 explicit override wins" "$OUT" "--master-port 29600"

# ---- 20. duplicate master_port rejected ----
log_test "duplicate master_port is rejected"
cat > "$TMPDIR_STACK/dup-mport.yaml" <<'EOF'
name: dup-mport
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, master_port: 29501}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8001, master_port: 29501}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/dup-mport.yaml" --dry-run \
        --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"master_port 29501 duplicates"* ]]; then
    log_pass "duplicate master_port errors with nonzero exit"
else
    log_fail "duplicate master_port errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 21. master_port colliding with an API port rejected ----
log_test "master_port colliding with an API port is rejected"
cat > "$TMPDIR_STACK/mport-clash.yaml" <<'EOF'
name: mport-clash
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, master_port: 8001}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8001}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/mport-clash.yaml" --dry-run \
        --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"collides with an API port"* ]]; then
    log_pass "master_port/API port collision errors"
else
    log_fail "master_port/API port collision errors (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 22. non-integer master_port names the field ----
log_test "non-integer master_port is rejected"
cat > "$TMPDIR_STACK/bad-mport.yaml" <<'EOF'
name: bad-mport
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, master_port: soon}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/bad-mport.yaml" --dry-run \
        --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"master_port"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "bad master_port errors naming the field"
else
    log_fail "bad master_port errors naming the field (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 22b. an explicit master_port equal to the 29501 default survives a
# differing .env MASTER_PORT (launch-cluster.sh honors any passed value; the
# .env applies only when no --master-port is passed at all) ----
log_test "explicit master_port 29501 coexists with a differing .env MASTER_PORT"
cat > "$TMPDIR_STACK/mport-override.env" <<'EOF'
CLUSTER_NODES="10.0.0.1,10.0.0.2"
LOCAL_IP="10.0.0.1"
MASTER_PORT=29500
EOF
cat > "$TMPDIR_STACK/mport-override.yaml" <<'EOF'
name: mport-override
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8001, master_port: 29501}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/mport-override.yaml" --dry-run \
        --config "$TMPDIR_STACK/mport-override.env" 2>&1)"; RC=$?
if [[ $RC -eq 0 && "$OUT" == *"--master-port 29500"* && "$OUT" == *"--master-port 29501"* ]]; then
    log_pass ".env base auto-assigns 29500; explicit 29501 passes through"
else
    log_fail ".env base auto-assigns 29500; explicit 29501 passes through (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 22e. a nonexistent --config path is an error, not an empty env ----
log_test "nonexistent --config path is rejected"
OUT="$("$RUN_STACK" example-cluster-stack --dry-run \
        --config /nonexistent/typo.env 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"Error:"* && "$OUT" == *"/nonexistent/typo.env"* \
      && "$OUT" != *"=== Stack:"* ]]; then
    log_pass "typo'd --config errors instead of silently resolving"
else
    log_fail "typo'd --config errors instead of silently resolving (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 22f. out-of-range master_port is rejected (0 would be silently dropped) ----
log_test "master_port 0 is rejected"
cat > "$TMPDIR_STACK/mport-zero.yaml" <<'EOF'
name: mport-zero
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8001, master_port: 0}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/mport-zero.yaml" --dry-run \
        --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"'master_port' must be between 1 and 65535"* \
      && "$OUT" != *"Traceback"* ]]; then
    log_pass "master_port 0 errors instead of silently colliding"
else
    log_fail "master_port 0 errors instead of silently colliding (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 22g. YAML boolean port is rejected (true would become --port 1) ----
log_test "boolean port is rejected"
cat > "$TMPDIR_STACK/port-bool.yaml" <<'EOF'
name: port-bool
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: true}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/port-bool.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"'port' must be an integer"* \
      && "$OUT" != *"Traceback"* ]]; then
    log_pass "boolean port errors instead of becoming --port 1"
else
    log_fail "boolean port errors instead of becoming --port 1 (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 23. cluster mode emits -n and --master-port, never --solo ----
log_test "cluster dry-run emits -n and omits --solo"
OUT="$("$RUN_STACK" "$TMPDIR_STACK/clusterable.yaml" --dry-run \
        --config "$TMPDIR_STACK/cluster.env" 2>&1)"
log_verbose "$OUT"
assert_contains "node list passed explicitly" "$OUT" "-n 10.0.0.1,10.0.0.2"
assert_contains "master port passed" "$OUT" "--master-port 29501"
# The explicit --config must be forwarded to each run-recipe.py invocation so
# every entry resolves the same .env the stack itself used.
CONFIG_LINES="$(echo "$OUT" | grep './run-recipe.py')"
if [[ "$CONFIG_LINES" == *"--config"*"cluster.env"* ]]; then
    log_pass "--config forwarded on the run-recipe.py plan line"
else
    log_fail "--config forwarded on the run-recipe.py plan line"
    log_verbose "$CONFIG_LINES"
fi
# Scope the --solo check to the generated command lines only. A blanket
# `$OUT != *--solo*` would break the moment any banner or hint text mentions
# the flag, which is unrelated to what gets executed.
PLAN_LINES="$(echo "$OUT" | grep './run-recipe.py')"
if [[ -n "$PLAN_LINES" && "$PLAN_LINES" != *"--solo"* ]]; then
    log_pass "cluster plan commands contain no --solo"
else
    log_fail "cluster plan commands contain no --solo"
    log_verbose "$PLAN_LINES"
fi

# (Former tests 24-26 covered stack-wide mode contradictions; per-entry
# placement coverage now lives in P11-P13.)

# ---- 26b. malformed recipe YAML fails loudly instead of bypassing mode validation ----
log_test "malformed recipe YAML is rejected, not silently defaulted"
printf 'cluster_only: [\n  broken' > "$TMPDIR_STACK/broken-recipe.yaml"
cat > "$TMPDIR_STACK/broken-ref.yaml" <<EOF
name: broken-ref
recipes:
  - {recipe: $TMPDIR_STACK/broken-recipe.yaml, container_name: a, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/broken-ref.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"could not be read"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "malformed recipe YAML errors instead of defaulting to no mode restriction"
else
    log_fail "malformed recipe YAML errors instead of defaulting to no mode restriction (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 26b2. recipe YAML that parses to a non-mapping fails like unreadable YAML ----
log_test "list-shaped recipe YAML is rejected, not silently defaulted"
printf -- '- just\n- a list\n' > "$TMPDIR_STACK/list-recipe.yaml"
cat > "$TMPDIR_STACK/list-ref.yaml" <<EOF
name: list-ref
recipes:
  - {recipe: $TMPDIR_STACK/list-recipe.yaml, container_name: a, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/list-ref.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"could not be read"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "list-shaped recipe YAML errors instead of bypassing mode validation"
else
    log_fail "list-shaped recipe YAML errors instead of bypassing mode validation (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- 26c. a recipe's cluster_only/solo_only flag must NOT block --status ----
log_test "--status is not blocked by a recipe mode flag"
cat > "$TMPDIR_STACK/co.yaml" <<'EOF'
name: co
recipes:
  - {recipe: deepseek-v4-flash, container_name: a, port: 8000}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/co.yaml" --status \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"; RC=$?
if [[ $RC -eq 0 && "$OUT" == *"deepseek-v4-flash"* && "$OUT" != *"cluster_only"* ]]; then
    log_pass "--status inspects a contradictory stack instead of refusing"
else
    log_fail "--status inspects a contradictory stack instead of refusing (rc=$RC)"
    log_verbose "$OUT"
fi

# (Former test 27 covered stop_command; P15 below covers all its placement
# forms plus --config forwarding.)

# ---- P1. placement: all distributes across the .env cluster ----
log_test "placement all -> distributed"
cat > "$TMPDIR_STACK/cluster.env" <<'EOF'
CLUSTER_NODES="10.0.0.1,10.0.0.2"
LOCAL_IP="10.0.0.1"
EOF
cat > "$TMPDIR_STACK/p-all.yaml" <<'EOF'
name: p-all
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: all}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-all.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"
log_verbose "$OUT"
assert_contains "distributed emits -n" "$OUT" "-n 10.0.0.1,10.0.0.2"
assert_contains "distributed emits master-port" "$OUT" "--master-port 29501"

# ---- P2. placement: all, no cluster -> solo local (byte-identical solo form) ----
log_test "placement all, no cluster -> solo local"
: > "$TMPDIR_STACK/empty.env"
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-all.yaml" --dry-run --config "$TMPDIR_STACK/empty.env" 2>&1)"
log_verbose "$OUT"
assert_contains "solo local emits --solo" "$OUT" \
    "./run-recipe.py qwen3.6-35b-a3b-nvfp4 --solo --name a -d --port 8000"

# ---- P2a. 'all' means every available node, so on a single-node host it is
# the expected placement, not a fault: no warning. Covers both spellings --
# the omitted key AND an explicit 'placement: all' -- on the read-only
# sub-commands (up/--stop have side effects and are exercised elsewhere). ----
log_test "no-cluster fallback reports placement without warning"
cat > "$TMPDIR_STACK/p-default.yaml" <<'EOF'
name: p-default
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000}
EOF
for STACK_YAML in "$TMPDIR_STACK/p-default.yaml" "$TMPDIR_STACK/p-all.yaml"; do
    for MODE in --dry-run --status; do
        LABEL="$(basename "$STACK_YAML" .yaml) $MODE"
        OUT="$("$RUN_STACK" "$STACK_YAML" "$MODE" \
                --config "$TMPDIR_STACK/empty.env" 2>&1)"
        log_verbose "$OUT"
        assert_contains "$LABEL names the placement target" "$OUT" "-> this host"
        if [[ "$OUT" != *"WARNING"* ]]; then
            log_pass "$LABEL emits no fallback warning"
        else
            log_fail "$LABEL emits no fallback warning"
            log_verbose "$OUT"
        fi
    done
done

# ---- P3. placement: <local IP> -> solo on this host ----
log_test "placement local IP -> solo local"
cat > "$TMPDIR_STACK/p-local.yaml" <<'EOF'
name: p-local
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: 10.0.0.1}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-local.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"
assert_contains "local placement emits --solo" "$OUT" \
    "./run-recipe.py qwen3.6-35b-a3b-nvfp4 --solo --name a -d --port 8000"

# ---- P4. placement: <remote IP> -> --placement-node, never --solo ----
log_test "placement remote IP -> --placement-node"
cat > "$TMPDIR_STACK/p-remote.yaml" <<'EOF'
name: p-remote
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: 10.0.0.2}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-remote.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"
log_verbose "$OUT"
assert_contains "remote placement emits --placement-node" "$OUT" \
    "./run-recipe.py qwen3.6-35b-a3b-nvfp4 --placement-node 10.0.0.2 --name a -d --port 8000"
PLAN_LINES="$(echo "$OUT" | grep './run-recipe.py')"
if [[ -n "$PLAN_LINES" && "$PLAN_LINES" != *"--solo"* ]]; then
    log_pass "remote placement plan contains no --solo"
else log_fail "remote placement plan contains no --solo"; log_verbose "$PLAN_LINES"; fi
assert_contains "remote entry health-polls its node, not localhost" "$OUT" "http://10.0.0.2:8000/health"

# ---- P5. placement naming an unknown node is rejected at load time ----
log_test "unknown placement IP is rejected"
cat > "$TMPDIR_STACK/p-bad.yaml" <<'EOF'
name: p-bad
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: 10.9.9.9}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-bad.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"10.9.9.9"* && "$OUT" == *"placement"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "unknown placement node errors, naming the IP"
else log_fail "unknown placement node errors, naming the IP (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P6. mixed stack: one distributed, one pinned remote (distinct ports) ----
log_test "mixed placement in one stack"
cat > "$TMPDIR_STACK/p-mixed.yaml" <<'EOF'
name: p-mixed
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: big, port: 8000, placement: all}
  - {recipe: gemma4-26b-a4b, container_name: small, port: 8001, placement: 10.0.0.2}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-mixed.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"
log_verbose "$OUT"
assert_contains "big is distributed" "$OUT" "qwen3.6-35b-a3b-nvfp4 -n 10.0.0.1,10.0.0.2"
assert_contains "small is pinned remote" "$OUT" "gemma4-26b-a4b --placement-node 10.0.0.2"

# ---- P16. a non-string placement (e.g. a list for a subset) is rejected cleanly ----
log_test "non-string placement is rejected without a traceback"
cat > "$TMPDIR_STACK/p-listplace.yaml" <<'EOF'
name: p-listplace
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: [10.0.0.1, 10.0.0.2]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-listplace.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"placement"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "non-string placement errors cleanly, naming the field"
else
    log_fail "non-string placement errors cleanly, naming the field (rc=$RC)"; log_verbose "$OUT"
fi

# ---- P7. same API port on different nodes is allowed ----
log_test "same port on different nodes is allowed"
cat > "$TMPDIR_STACK/p-ports-ok.yaml" <<'EOF'
name: p-ports-ok
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: 10.0.0.1}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8000, placement: 10.0.0.2}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-ports-ok.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -eq 0 ]]; then log_pass "same port, different nodes accepted"
else log_fail "same port, different nodes accepted (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P8. same API port on the SAME node is rejected ----
log_test "duplicate port on the same node is rejected"
cat > "$TMPDIR_STACK/p-ports-bad.yaml" <<'EOF'
name: p-ports-bad
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: 10.0.0.2}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8000, placement: 10.0.0.2}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-ports-bad.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"port 8000"* && "$OUT" == *"10.0.0.2"* ]]; then
    log_pass "duplicate port on one node rejected"
else log_fail "duplicate port on one node rejected (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P9. single-node entries may reuse a master_port value (unused) ----
log_test "single-node entries need no distinct master_port"
cat > "$TMPDIR_STACK/p-mp-single.yaml" <<'EOF'
name: p-mp-single
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: 10.0.0.1, master_port: 29501}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8001, placement: 10.0.0.2, master_port: 29501}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-mp-single.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -eq 0 ]]; then log_pass "single-node master_port reuse accepted"
else log_fail "single-node master_port reuse accepted (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P10. two DISTRIBUTED entries sharing a master_port is rejected ----
log_test "duplicate master_port among distributed entries is rejected"
cat > "$TMPDIR_STACK/p-mp-dist.yaml" <<'EOF'
name: p-mp-dist
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: a, port: 8000, placement: all, master_port: 29501}
  - {recipe: gemma4-26b-a4b, container_name: b, port: 8001, placement: all, master_port: 29501}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-mp-dist.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"master_port"* ]]; then log_pass "duplicate distributed master_port rejected"
else log_fail "duplicate distributed master_port rejected (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P11. cluster_only recipe pinned to a single node is rejected ----
log_test "cluster_only + single-node placement rejected"
cat > "$TMPDIR_STACK/p-co.yaml" <<'EOF'
name: p-co
recipes:
  - {recipe: deepseek-v4-flash, container_name: a, port: 8000, placement: 10.0.0.2}
EOF
# Exit code 3 is the CI contract: the workflow retries a mode-conflicting
# stack without a cluster only when it sees 3, so pin the exact code here.
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-co.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -eq 3 && "$OUT" == *"cluster_only"* && "$OUT" == *"deepseek-v4-flash"* ]]; then
    log_pass "cluster_only + single-node rejected with exit code 3"
else log_fail "cluster_only + single-node rejected with exit code 3 (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P12. solo_only recipe that resolves distributed is rejected ----
log_test "solo_only + distributed rejected"
cat > "$TMPDIR_STACK/p-so.yaml" <<'EOF'
name: p-so
recipes:
  - {recipe: openai-gpt-oss-120b, container_name: a, port: 8000, placement: all}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-so.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -eq 3 && "$OUT" == *"solo_only"* && "$OUT" == *"openai-gpt-oss-120b"* ]]; then
    log_pass "solo_only + distributed rejected with exit code 3"
else log_fail "solo_only + distributed rejected with exit code 3 (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P13. solo_only recipe pinned to a single node is fine ----
log_test "solo_only + single-node accepted"
cat > "$TMPDIR_STACK/p-so-ok.yaml" <<'EOF'
name: p-so-ok
recipes:
  - {recipe: openai-gpt-oss-120b, container_name: a, port: 8000, placement: 10.0.0.2}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-so-ok.yaml" --dry-run --config "$TMPDIR_STACK/cluster.env" 2>&1)"; RC=$?
if [[ $RC -eq 0 ]]; then log_pass "solo_only + single-node accepted"
else log_fail "solo_only + single-node accepted (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P14. --status is NOT blocked by a contradiction (LOCAL-resolving; no ssh) ----
log_test "--status not blocked by a placement contradiction (local)"
cat > "$TMPDIR_STACK/p-co-local.yaml" <<'EOF'
name: p-co-local
recipes:
  - {recipe: deepseek-v4-flash, container_name: a, port: 8000, placement: all}
EOF
# empty.env -> no cluster -> resolves single-LOCAL, so --status does only local
# docker inspect + localhost health, never an ssh.
OUT="$("$RUN_STACK" "$TMPDIR_STACK/p-co-local.yaml" --status --config "$TMPDIR_STACK/empty.env" 2>&1)"; RC=$?
if [[ $RC -eq 0 && "$OUT" == *"deepseek-v4-flash"* && "$OUT" != *"cluster_only"* ]]; then
    log_pass "--status inspects a contradictory (local) stack instead of refusing"
else log_fail "--status inspects a contradictory (local) stack instead of refusing (rc=$RC)"; log_verbose "$OUT"; fi

# ---- P15. stop_command renders per placement, and forwards --config ----
log_test "stop_command distributed/local/remote argv"
OUT="$(python3 - "$PROJECT_DIR" <<'PY'
import sys, importlib.util
from pathlib import Path
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("rs", root / "run-stack.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
P = rs.Placement
def entry(p): return {"container_name": "a", "placement": p, "master_port": 29501}
print("DIST:", rs.stop_command(entry(P("distributed", ["10.0.0.1","10.0.0.2"], None, ""))))
print("LOCAL:", rs.stop_command(entry(P("single", [], None, ""))))
print("REMOTE:", rs.stop_command(entry(P("single", [], "10.0.0.2", "")), "/x/.env"))
PY
)"
log_verbose "$OUT"
assert_contains "distributed stop passes -n" "$OUT" "'-n', '10.0.0.1,10.0.0.2', '--name', 'a', 'stop'"
assert_contains "local stop keeps --solo" "$OUT" "'--solo', '--name', 'a', 'stop'"
assert_contains "remote stop passes --placement-node" "$OUT" "'--placement-node', '10.0.0.2'"
assert_contains "remote stop forwards --config" "$OUT" "'--config', '/x/.env'"
if [[ "$OUT" != *"REMOTE:"*"--solo"* ]]; then log_pass "remote stop omits --solo"
else log_fail "remote stop omits --solo"; log_verbose "$OUT"; fi
# The load-bearing invariant: a distributed stop must NEVER pass --solo, or
# launch-cluster.sh's cleanup skips its ssh worker loop and strands worker
# containers. Require the DIST line to be present AND free of --solo (so this
# cannot pass vacuously when stop_command errors before printing).
DIST_LINE="$(echo "$OUT" | grep '^DIST:')"
if [[ -n "$DIST_LINE" && "$DIST_LINE" != *"--solo"* ]]; then
    log_pass "distributed stop omits --solo (no worker stranding)"
else
    log_fail "distributed stop omits --solo (no worker stranding)"; log_verbose "$DIST_LINE"
fi

# ---- V1. absolute host paths pass through; -v lands before the `--` ----
log_test "volumes: absolute host path renders as -v before extra_args"
cat > "$TMPDIR_STACK/vol-abs.yaml" <<'EOF'
name: vol-abs
recipes:
  - recipe: qwen3.6-35b-a3b-nvfp4
    container_name: q
    port: 8000
    volumes: ["/data/models:/models", "/data/out:/out:ro"]
    extra_args: ["--load-format", "instanttensor"]
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-abs.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
log_verbose "$OUT"
assert_contains "absolute host path unchanged" "$OUT" "-v /data/models:/models"
assert_contains ":ro option suffix preserved" "$OUT" "-v /data/out:/out:ro"
# -v must precede the `--` separator or run-recipe.py would hand it to vLLM.
if [[ "$OUT" == *"-v /data/models:/models"*"-- --load-format"* ]]; then
    log_pass "-v precedes the -- extra_args separator"
else
    log_fail "-v precedes the -- extra_args separator"; log_verbose "$OUT"
fi

# ---- V2. relative host paths resolve against the repo root (as mods do) ----
log_test "volumes: relative host path resolves to an absolute path"
cat > "$TMPDIR_STACK/vol-rel.yaml" <<'EOF'
name: vol-rel
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["data/models:/models"]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-rel.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
log_verbose "$OUT"
assert_contains "relative host path made absolute" "$OUT" \
    "-v $PROJECT_DIR/data/models:/models"

# ---- V3. a bare name (no slash) stays a named Docker volume ----
log_test "volumes: bare name is left as a named Docker volume"
cat > "$TMPDIR_STACK/vol-named.yaml" <<'EOF'
name: vol-named
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["hfcache:/root/.cache/huggingface"]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-named.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
log_verbose "$OUT"
assert_contains "named volume not path-resolved" "$OUT" \
    "-v hfcache:/root/.cache/huggingface"
if [[ "$OUT" != *"$PROJECT_DIR/hfcache"* ]]; then
    log_pass "named volume not rewritten to a repo-relative path"
else
    log_fail "named volume not rewritten to a repo-relative path"; log_verbose "$OUT"
fi

# ---- V4. a relative CONTAINER path is rejected (Docker requires absolute) ----
log_test "volumes: relative container path is rejected"
cat > "$TMPDIR_STACK/vol-badtarget.yaml" <<'EOF'
name: vol-badtarget
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["/data/models:models"]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-badtarget.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"container path"* && "$OUT" == *"absolute"* ]]; then
    log_pass "relative container path errors with nonzero exit"
else
    log_fail "relative container path errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- V5. a mapping with no container path is rejected ----
log_test "volumes: mapping without a container path is rejected"
cat > "$TMPDIR_STACK/vol-nocolon.yaml" <<'EOF'
name: vol-nocolon
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["/data/models"]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-nocolon.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"host:container"* ]]; then
    log_pass "missing container path errors with nonzero exit"
else
    log_fail "missing container path errors with nonzero exit (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- V6. volumes are forwarded per entry, not leaked across entries ----
log_test "volumes: applied only to the entry that declares them"
cat > "$TMPDIR_STACK/vol-scope.yaml" <<'EOF'
name: vol-scope
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["/data/a:/a"]}
  - {recipe: nemotron-3-nano-nvfp4, container_name: n, port: 8001}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-scope.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
log_verbose "$OUT"
ENTRY2="$(echo "$OUT" | grep 'run-recipe.py nemotron-3-nano-nvfp4')"
if [[ -n "$ENTRY2" && "$ENTRY2" != *"-v "* ]]; then
    log_pass "entry without volumes gets no -v"
else
    log_fail "entry without volumes gets no -v"; log_verbose "$ENTRY2"
fi

# ---- V7. a non-list 'volumes:' is rejected cleanly, not with a traceback ----
log_test "volumes: non-list value is rejected cleanly"
cat > "$TMPDIR_STACK/vol-notlist.yaml" <<'EOF'
name: vol-notlist
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: "/data/models:/models"}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-notlist.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"Error:"* && "$OUT" == *"must be a list"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "non-list volumes errors without a traceback"
else
    log_fail "non-list volumes errors without a traceback (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- V8. '~' expands; an unknown ~user is rejected cleanly, not a traceback ----
log_test "volumes: ~ expands and an unknown ~user is rejected cleanly"
cat > "$TMPDIR_STACK/vol-tilde.yaml" <<'EOF'
name: vol-tilde
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["~/models:/models"]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-tilde.yaml" --dry-run \
        --config "$TMPDIR_STACK/empty.env" 2>&1)"
assert_contains "~ expands to \$HOME" "$OUT" "-v $HOME/models:/models"
cat > "$TMPDIR_STACK/vol-baduser.yaml" <<'EOF'
name: vol-baduser
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["~nosuchuser42/models:/models"]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-baduser.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"Error:"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "unknown ~user errors without a traceback"
else
    log_fail "unknown ~user errors without a traceback (rc=$RC)"
    log_verbose "$OUT"
fi

# ---- V9. shell metacharacters are refused at load time ----
# launch-cluster.sh expands $DOCKER_ARGS unquoted and re-parses the docker line
# through bash -c/ssh, so a space or a $ in a mapping would word-split or
# execute rather than mount. Refuse it here with a readable message instead.
log_test "volumes: whitespace / shell metacharacters are rejected"
cat > "$TMPDIR_STACK/vol-space.yaml" <<'EOF'
name: vol-space
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["/data/my models:/models"]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-space.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"Error:"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "host path with a space is rejected"
else
    log_fail "host path with a space is rejected (rc=$RC)"; log_verbose "$OUT"
fi
cat > "$TMPDIR_STACK/vol-meta.yaml" <<'EOF'
name: vol-meta
recipes:
  - {recipe: qwen3.6-35b-a3b-nvfp4, container_name: q, port: 8000, volumes: ["/data/$(id -u):/models"]}
EOF
OUT="$("$RUN_STACK" "$TMPDIR_STACK/vol-meta.yaml" --dry-run 2>&1)"; RC=$?
if [[ $RC -ne 0 && "$OUT" == *"Error:"* && "$OUT" != *"Traceback"* ]]; then
    log_pass "host path with shell metacharacters is rejected"
else
    log_fail "host path with shell metacharacters is rejected (rc=$RC)"; log_verbose "$OUT"
fi

echo "========================================"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}  Failed: ${RED}${TESTS_FAILED}${NC}"
echo "========================================"
[[ $TESTS_FAILED -eq 0 ]]
