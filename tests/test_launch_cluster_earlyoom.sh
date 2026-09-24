#!/bin/bash
# Exercise earlyoom bootstrap scripts with fake Docker, SSH, and package tools.
# The container PATH is isolated so no host package manager can be invoked.
set -euo pipefail

PROJECT_DIR="$(dirname "$(dirname "$(realpath "$0")")")"
TMP_BASE="$(mktemp -d)"
TEST_INDEX=0

cleanup() {
    local pid_file
    for pid_file in "$TMP_BASE"/case-*/containers/*/pid; do
        [[ -f "$pid_file" ]] || continue
        kill "$(cat "$pid_file")" 2>/dev/null || true
    done
    rm -rf "$TMP_BASE"
}
trap cleanup EXIT

fail() {
    echo "[FAIL] $*" >&2
    cat "$OUTPUT_LOG" "$COMMAND_LOG" >&2
    exit 1
}

setup_fixture() {
    TEST_INDEX=$((TEST_INDEX + 1))
    CASE_DIR="$TMP_BASE/case-$TEST_INDEX"
    FIXTURE_DIR="$CASE_DIR/project"
    FAKE_BIN_DIR="$CASE_DIR/bin"
    COMMAND_LOG="$CASE_DIR/commands.log"
    OUTPUT_LOG="$CASE_DIR/output.log"
    mkdir -p "$FIXTURE_DIR/mod" "$FAKE_BIN_DIR"
    cp "$PROJECT_DIR/launch-cluster.sh" "$PROJECT_DIR/autodiscover.sh" "$FIXTURE_DIR/"
    touch "$FIXTURE_DIR/test.env" "$FIXTURE_DIR/mod/run.sh" "$COMMAND_LOG"

    cat > "$CASE_DIR/earlyoom" <<'EARLYOOM'
#!/bin/bash
if [[ "$(< "$CONTAINER_DIR/mode")" == startup-fail ]]; then
    exit 1
fi
printf '%s earlyoom' "$TEST_NODE" >> "$COMMAND_LOG"
printf ' <%s>' "$@" >> "$COMMAND_LOG"
printf '\n' >> "$COMMAND_LOG"
printf 'earlyoom\n' > "$CONTAINER_DIR/comm"
exec /bin/sleep 60
EARLYOOM

    cat > "$CASE_DIR/apt-get" <<'APT'
#!/bin/bash
set -euo pipefail
mode="$(< "$CONTAINER_DIR/mode")"
echo "$TEST_NODE apt-get $* DEBIAN_FRONTEND=${DEBIAN_FRONTEND:-}" >> "$COMMAND_LOG"
if [[ "$mode" == "$1-fail" ]]; then
    echo "mock apt-get $1 failed" >&2
    exit 100
fi
if [[ "$1" == install && "$mode" != no-binary ]]; then
    /bin/cp "$CASE_DIR/earlyoom" "$CONTAINER_DIR/bin/earlyoom"
    /bin/chmod +x "$CONTAINER_DIR/bin/earlyoom"
    # Make the binary visible before package configuration finishes.
    /bin/sleep 0.1
    echo "$TEST_NODE installation complete" >> "$COMMAND_LOG"
fi
APT

    cat > "$FAKE_BIN_DIR/docker" <<'DOCKER'
#!/bin/bash
set -euo pipefail
export TEST_NODE="${TEST_NODE:-head}"
export CONTAINER_DIR="$CASE_DIR/containers/$TEST_NODE"
printf '%s docker' "$TEST_NODE" >> "$COMMAND_LOG"
printf ' %q' "$@" >> "$COMMAND_LOG"
printf '\n' >> "$COMMAND_LOG"

container_script() {
    local script="$1"
    script="${script//\/proc\/1\/comm/$CONTAINER_DIR/comm}"
    script="${script//\/tmp\/vllm-spark-earlyoom-ready/$CONTAINER_DIR/ready}"
    shift
    PATH="$CONTAINER_DIR/bin" exec /bin/bash -c "$script" "$@"
}

case "$1" in
    ps) exit 0 ;;
    image) echo sha256:test-image ;;
    run)
        while [[ "$1" != "$TEST_IMAGE" ]]; do shift; done
        shift
        if [[ "$1" == bash ]]; then
            [[ "$2" == -c ]]
            container_script "$3" "${@:4}" > "$CONTAINER_DIR/output.log" 2>&1 &
            echo "$!" > "$CONTAINER_DIR/pid"
        else
            [[ "$*" == 'sleep infinity' ]]
        fi
        ;;
    exec)
        if [[ "$2" == --user ]]; then
            [[ "$3" == 0 && "$4" == -w && "$5" == / ]]
            [[ "$7" == bash && "$8" == -c ]]
            container_script "$9"
        elif [[ "$*" == *'chmod +x run.sh'* ]]; then
            echo "$TEST_NODE mod applied" >> "$COMMAND_LOG"
        elif [[ "$*" == *'vllm serve'* || "$*" == *'ray start'* ]]; then
            echo "$TEST_NODE workload started" >> "$COMMAND_LOG"
        fi
        ;;
    stop)
        if [[ -f "$CONTAINER_DIR/pid" ]]; then
            kill "$(< "$CONTAINER_DIR/pid")" 2>/dev/null || true
            rm -f "$CONTAINER_DIR/pid"
        fi
        ;;
esac
DOCKER

    cat > "$FAKE_BIN_DIR/ssh" <<'SSH'
#!/bin/bash
set -euo pipefail
while [[ "$1" == -o ]]; do shift 2; done
shift
if [[ "$*" == 'mkdir -p /tmp/vllm_mod_pkg_'* || "$*" == 'rm -rf /tmp/vllm_mod_pkg_'* ]]; then
    exit 0
fi
TEST_NODE=worker /bin/bash -c "$*"
SSH

    cat > "$FAKE_BIN_DIR/sleep" <<'SLEEP'
#!/bin/bash
/bin/sleep 0.02
SLEEP
    cat > "$FAKE_BIN_DIR/scp" <<'SCP'
#!/bin/bash
exit 0
SCP
    chmod +x "$FAKE_BIN_DIR/"* "$CASE_DIR/apt-get" "$CASE_DIR/earlyoom"
}

setup_container() {
    local node="$1"; local mode="$2"
    local container_dir="$CASE_DIR/containers/$node"
    mkdir -p "$container_dir/bin"
    echo "$mode" > "$container_dir/mode"
    echo bash > "$container_dir/comm"
    ln -s /bin/bash "$container_dir/bin/bash"
    ln -s /bin/touch "$container_dir/bin/touch"
    cp "$FAKE_BIN_DIR/sleep" "$container_dir/bin/sleep"
    if [[ "$mode" != no-apt ]]; then
        cp "$CASE_DIR/apt-get" "$container_dir/bin/apt-get"
    fi
    if [[ "$mode" == present || "$mode" == startup-fail ]]; then
        cp "$CASE_DIR/earlyoom" "$container_dir/bin/earlyoom"
    fi
}

run_launch() {
    (
        cd "$FIXTURE_DIR"
        export PATH="$FAKE_BIN_DIR:$PATH"
        export CASE_DIR COMMAND_LOG
        export TEST_IMAGE="${TEST_IMAGE:-third-party:test}"
        export LOCAL_IP=10.0.0.1
        unset VLLM_SPARK_EXTRA_DOCKER_ARGS VLLM_SPARK_EARLYOOM_ARGS
        ./launch-cluster.sh --config "$FIXTURE_DIR/test.env" \
            --eth-if eth0 --ib-if ib0 --no-cache-dirs -t "$TEST_IMAGE" -d "$@"
    ) > "$OUTPUT_LOG" 2>&1
}

assert_contains() {
    grep -Fq -- "$2" "$1" || fail "Expected $1 to contain: $2"
}

assert_not_contains() {
    if grep -Fq -- "$2" "$1"; then
        fail "Expected $1 not to contain: $2"
    fi
}

assert_before() {
    local first second
    first="$(grep -nF -- "$1" "$COMMAND_LOG" | head -1 | cut -d: -f1)"
    second="$(grep -nF -- "$2" "$COMMAND_LOG" | head -1 | cut -d: -f1)"
    [[ -n "$first" && -n "$second" && "$first" -lt "$second" ]] \
        || fail "Expected '$1' before '$2'"
}

setup_fixture
setup_container head no-apt
run_launch --solo exec vllm serve test-model || fail "launch without earlyoom failed"
assert_contains "$COMMAND_LOG" 'third-party:test sleep infinity'
assert_not_contains "$OUTPUT_LOG" 'Preparing earlyoom'
assert_not_contains "$COMMAND_LOG" 'docker exec --user'
echo '[PASS] no earlyoom flag leaves the idle command unchanged'

for image in vllm-node vllm-node-b12x third-party:test; do
    setup_fixture
    setup_container head present
    TEST_IMAGE="$image" run_launch --solo --earlyoom start || fail "preinstalled earlyoom failed"
    assert_contains "$COMMAND_LOG" 'head earlyoom <-M> <524288,102400> <-s> <100> <-r> <60>'
    assert_not_contains "$COMMAND_LOG" 'head apt-get'
done
echo '[PASS] preinstalled earlyoom skips installation regardless of image name'

setup_fixture
setup_container head missing
setup_container worker missing
run_launch --nodes 10.0.0.1,10.0.0.2 \
    --earlyoom-args '-M 786432,196608 -s 100 -r 120' \
    --apply-mod "$FIXTURE_DIR/mod" exec vllm serve test-model -tp 2 \
    || fail "cluster installation failed"
for node in head worker; do
    assert_contains "$COMMAND_LOG" "$node apt-get update"
    assert_contains "$COMMAND_LOG" "$node apt-get install -y --no-install-recommends earlyoom DEBIAN_FRONTEND=noninteractive"
    assert_contains "$COMMAND_LOG" "$node earlyoom <-M> <786432,196608> <-s> <100> <-r> <120>"
    assert_before "$node installation complete" "$node earlyoom"
    assert_before "$node earlyoom" "$node mod applied"
    assert_before "$node mod applied" "$node workload started"
done
echo '[PASS] missing earlyoom installs on every node before mods and workloads, with custom args'

for failure in update-fail install-fail no-apt no-binary startup-fail; do
    setup_fixture
    setup_container head "$failure"
    if run_launch --solo --earlyoom --apply-mod "$FIXTURE_DIR/mod" exec vllm serve test-model; then
        fail "launch succeeded despite $failure"
    fi
    case "$failure" in
        update-fail) diagnostic='apt-get update failed while preparing earlyoom' ;;
        install-fail) diagnostic='apt-get could not install earlyoom' ;;
        no-apt) diagnostic='apt-get is unavailable' ;;
        no-binary) diagnostic='earlyoom is still unavailable' ;;
        startup-fail) diagnostic='earlyoom did not start' ;;
    esac
    assert_contains "$OUTPUT_LOG" "$diagnostic"
    assert_contains "$OUTPUT_LOG" "image 'third-party:test'"
    assert_contains "$OUTPUT_LOG" 'Restart the launch without --earlyoom'
    assert_contains "$OUTPUT_LOG" 'without --earlyoom-args'
    assert_contains "$COMMAND_LOG" 'head docker stop vllm_node'
    assert_not_contains "$COMMAND_LOG" 'mod applied'
    assert_not_contains "$COMMAND_LOG" 'workload started'
    if [[ "$failure" == update-fail ]]; then
        assert_not_contains "$COMMAND_LOG" 'head apt-get install'
    fi
    echo "[PASS] $failure stops the launch and prints recovery instructions"
done

setup_fixture
setup_container head present
setup_container worker install-fail
if run_launch --nodes 10.0.0.1,10.0.0.2 --ray --earlyoom exec vllm serve test-model -tp 2; then
    fail 'launch succeeded despite worker installation failure'
fi
assert_contains "$OUTPUT_LOG" "on 10.0.0.2."
assert_contains "$OUTPUT_LOG" 'Restart the launch without --earlyoom'
assert_contains "$COMMAND_LOG" 'head docker stop vllm_node'
assert_contains "$COMMAND_LOG" 'worker docker stop vllm_node'
assert_not_contains "$COMMAND_LOG" 'workload started'
echo '[PASS] worker installation failure aborts before Ray/vLLM and cleans up the cluster'

echo "All earlyoom tests passed ($TEST_INDEX fixtures)."
