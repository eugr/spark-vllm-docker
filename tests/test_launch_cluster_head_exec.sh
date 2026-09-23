#!/bin/bash
#
# Focused behavior tests for launch-cluster.sh head command dispatch.
#
# In no-ray multi-node runs the worker ranks are always started detached, with
# their output piped to the container log. The head used to run in the launcher
# foreground through a pty, which lost its output when the launcher terminal went
# away. These tests pin the detached dispatch, the wait that keeps the launcher's
# cleanup semantics, and the exit status propagation. All Docker and SSH
# operations are handled by fake commands.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TMP_BASE="$(mktemp -d)"
TEST_INDEX=0
TESTS_PASSED=0

cleanup() {
    rm -rf "$TMP_BASE"
}
trap cleanup EXIT

pass() {
    echo "[PASS] $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo "[FAIL] $1" >&2
    if [[ -f "${OUTPUT_LOG:-}" ]]; then
        echo "--- output ---" >&2
        sed -n '1,220p' "$OUTPUT_LOG" >&2
    fi
    if [[ -f "${TEST_LOG:-}" ]]; then
        echo "--- command log ---" >&2
        sed -n '1,220p' "$TEST_LOG" >&2
    fi
    exit 1
}

setup_fixture() {
    TEST_INDEX=$((TEST_INDEX + 1))
    CASE_DIR="$TMP_BASE/case-$TEST_INDEX"
    FIXTURE_DIR="$CASE_DIR/project"
    FAKE_BIN_DIR="$CASE_DIR/bin"
    STATE_DIR="$CASE_DIR/state"
    TEST_LOG="$CASE_DIR/commands.log"
    OUTPUT_LOG="$CASE_DIR/output.log"

    mkdir -p "$FIXTURE_DIR" "$FAKE_BIN_DIR" "$STATE_DIR"
    cp "$PROJECT_DIR/launch-cluster.sh" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/autodiscover.sh" "$FIXTURE_DIR/"
    touch "$FIXTURE_DIR/test.env"
    : > "$TEST_LOG"
    : > "$OUTPUT_LOG"

    cat > "$FAKE_BIN_DIR/docker" <<'DOCKER'
#!/bin/bash
set -uo pipefail
echo "docker $*" >> "$TEST_LOG"
state_dir="$STATE_DIR"
mkdir -p "$state_dir"

case "${1:-}" in
    ps)
        # Report the container only once the head command has been dispatched and
        # until it is stopped. Before dispatch the launcher checks for a
        # pre-existing cluster, and a running container would make it skip cleanup.
        if [[ -f "$state_dir/dispatched" && ! -f "$state_dir/stopped" && "${CONTAINER_GONE:-false}" != "true" ]]; then
            echo "${CONTAINER_NAME:-vllm_node}"
        fi
        exit 0
        ;;
    stop)
        touch "$state_dir/stopped"
        exit 0
        ;;
    logs)
        # Emulate `docker logs -f`: block until the launcher kills us.
        if [[ "${LOGS_BLOCK:-true}" == "true" ]]; then
            while true; do /bin/sleep 1; done
        fi
        exit 0
        ;;
    image)
        if [[ "${2:-}" == "inspect" ]]; then
            echo "sha256:test"
        fi
        exit 0
        ;;
    exec)
        shift
        detached="false"
        while [[ "${1:-}" == -* ]]; do
            [[ "$1" == "-d" ]] && detached="true"
            shift
        done
        shift || true   # container name
        case "${1:-}" in
            rm)
                exit 0
                ;;
            cat)
                polls=0
                [[ -f "$state_dir/polls" ]] && polls=$(wc -l < "$state_dir/polls")
                polls=$((polls + 1))
                echo "$polls" >> "$state_dir/polls"
                if [[ "$polls" -ge "${SENTINEL_AFTER_POLLS:-3}" ]]; then
                    printf '%s\n' "${HEAD_EXIT_STATUS:-0}"
                    exit 0
                fi
                exit 1
                ;;
        esac
        if [[ "$detached" == "true" ]]; then
            touch "$state_dir/dispatched"
            printf '%s\n' "$*" > "$state_dir/dispatch-args"
        fi
        exit 0
        ;;
esac

exit 0
DOCKER

    cat > "$FAKE_BIN_DIR/ssh" <<'SSH'
#!/bin/bash
set -uo pipefail
echo "ssh $*" >> "$TEST_LOG"

if [[ "$*" == *"docker ps --format"* ]]; then
    # Report a running container only after the head command was dispatched.
    # A pre-existing cluster makes launch-cluster.sh skip cleanup.
    if [[ -f "$STATE_DIR/dispatched" ]]; then
        echo "${CONTAINER_NAME:-vllm_node}"
        exit 0
    fi
    exit 1
fi

if [[ "$*" == *"docker image inspect"* ]]; then
    echo "sha256:test"
fi

exit 0
SSH

    cat > "$FAKE_BIN_DIR/sleep" <<'SLEEP'
#!/bin/bash
exit 0
SLEEP

    chmod +x "$FAKE_BIN_DIR/docker" "$FAKE_BIN_DIR/ssh" "$FAKE_BIN_DIR/sleep"
}

run_launch() {
    (
        cd "$FIXTURE_DIR"
        PATH="$FAKE_BIN_DIR:$PATH" \
            TEST_LOG="$TEST_LOG" \
            STATE_DIR="$STATE_DIR" \
            LOCAL_IP="10.0.0.1" \
            SENTINEL_AFTER_POLLS="${SENTINEL_AFTER_POLLS:-3}" \
            HEAD_EXIT_STATUS="${HEAD_EXIT_STATUS:-0}" \
            CONTAINER_GONE="${CONTAINER_GONE:-false}" \
            ./launch-cluster.sh \
                --config "$FIXTURE_DIR/test.env" \
                --nodes "10.0.0.1,10.0.0.2" \
                --eth-if eth0 \
                --ib-if ib0 \
                --no-cache-dirs \
                "$@" exec vllm serve test-model --port 8000 -tp 2
    ) > "$OUTPUT_LOG" 2>&1
}

assert_output_contains() {
    local pattern="$1"
    grep -Eq "$pattern" "$OUTPUT_LOG" || fail "Expected output to match: $pattern"
}

assert_log_contains() {
    local pattern="$1"
    grep -Eq "$pattern" "$TEST_LOG" || fail "Expected command log to match: $pattern"
}

assert_log_not_contains() {
    local pattern="$1"
    if grep -Eq "$pattern" "$TEST_LOG"; then
        fail "Expected command log not to match: $pattern"
    fi
}

log_line_number() {
    local pattern="$1"
    grep -nE "$pattern" "$TEST_LOG" | head -1 | cut -d: -f1 || true
}

test_head_runs_detached_like_workers() {
    setup_fixture
    run_launch || fail "no-ray exec launch failed"

    assert_log_contains '^docker exec -d vllm_node bash -c .*>> /proc/1/fd/1 2>&1; echo \$\? > /tmp/\.launch-cluster-head-exit'
    assert_log_contains '^ssh .*docker exec -d vllm_node bash -c'
    assert_log_not_contains 'docker exec -it'
    assert_log_not_contains 'docker exec -i '
    pass "head command is dispatched detached with container-log output"
}

test_launcher_waits_for_head_sentinel() {
    setup_fixture
    SENTINEL_AFTER_POLLS=3 run_launch || fail "no-ray exec launch failed"

    local polls
    polls=$(grep -cE '^docker exec vllm_node cat /tmp/\.launch-cluster-head-exit' "$TEST_LOG" || true)
    [[ "$polls" -ge 3 ]] || fail "Expected at least 3 sentinel polls, saw $polls"
    pass "launcher waits for the head command to report completion ($polls polls)"
}

test_cleanup_still_runs_after_head_exit() {
    setup_fixture
    run_launch || fail "no-ray exec launch failed"

    local stop_line poll_line
    stop_line=$(log_line_number '^docker stop vllm_node')
    poll_line=$(log_line_number '^docker exec vllm_node cat /tmp/\.launch-cluster-head-exit')
    [[ -n "$stop_line" ]] || fail "Expected cleanup to stop the head container"
    [[ -n "$poll_line" ]] || fail "Expected the launcher to poll the sentinel"
    [[ "$stop_line" -gt "$poll_line" ]] || \
        fail "Expected cleanup to stop the container only after the head command exited"
    assert_output_contains 'Stopping cluster\.\.\.'
    pass "no-ray exec exits into the existing cleanup path"
}

test_head_exit_status_is_propagated() {
    setup_fixture
    local status=0
    HEAD_EXIT_STATUS=7 run_launch || status=$?
    [[ "$status" -eq 7 ]] || fail "Expected launcher exit status 7, got $status"
    pass "head command exit status is propagated to the launcher"
}

test_daemon_mode_dispatches_without_waiting() {
    setup_fixture
    run_launch -d || fail "daemon mode launch failed"

    assert_log_contains '^docker exec -d vllm_node bash -c'
    assert_log_not_contains '^docker exec vllm_node cat'
    assert_log_not_contains '^docker stop vllm_node'
    assert_output_contains 'Command dispatched in background \(Daemon mode\)'
    pass "daemon mode still dispatches and returns without waiting"
}

test_missing_container_fails_the_wait() {
    setup_fixture
    if CONTAINER_GONE=true run_launch; then
        fail "launch unexpectedly succeeded after the container disappeared"
    fi
    assert_output_contains 'is no longer running; head command did not report completion'
    pass "a disappearing container ends the wait with a failure"
}

test_interrupt_still_stops_cluster() {
    setup_fixture

    # Keep the launcher in the wait loop, interrupt it, and check that the EXIT
    # trap still stops the cluster without reporting a missing completion.
    (
        cd "$FIXTURE_DIR"
        PATH="$FAKE_BIN_DIR:$PATH" \
            TEST_LOG="$TEST_LOG" \
            STATE_DIR="$STATE_DIR" \
            LOCAL_IP="10.0.0.1" \
            SENTINEL_AFTER_POLLS=100000 \
            exec ./launch-cluster.sh \
                --config "$FIXTURE_DIR/test.env" \
                --nodes "10.0.0.1,10.0.0.2" \
                --eth-if eth0 \
                --ib-if ib0 \
                --no-cache-dirs \
                exec vllm serve test-model --port 8000 -tp 2
    ) > "$OUTPUT_LOG" 2>&1 &
    local launcher_pid=$!

    local waited=0
    while [[ ! -f "$STATE_DIR/dispatched" && "$waited" -lt 2000 ]]; do
        sleep 1
        waited=$((waited + 1))
    done
    [[ -f "$STATE_DIR/dispatched" ]] || fail "head command was never dispatched"
    sleep 1
    kill -INT "$launcher_pid" 2>/dev/null || true
    local status=0
    wait "$launcher_pid" 2>/dev/null || status=$?

    [[ "$status" -ne 0 ]] || fail "interrupted launcher unexpectedly exited with status 0"
    assert_log_contains '^docker stop vllm_node'
    assert_output_contains 'Stopping cluster\.\.\.'
    if grep -q 'no longer running' "$OUTPUT_LOG"; then
        fail "interrupt reported a missing head completion after cleanup"
    fi
    pass "interrupting the launcher still stops the cluster cleanly"
}

test_head_runs_detached_like_workers
test_launcher_waits_for_head_sentinel
test_cleanup_still_runs_after_head_exit
test_head_exit_status_is_propagated
test_daemon_mode_dispatches_without_waiting
test_missing_container_fails_the_wait
test_interrupt_still_stops_cluster

echo "All $TESTS_PASSED launch-cluster head exec tests passed."
