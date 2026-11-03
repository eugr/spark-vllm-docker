#!/bin/bash
#
# Focused launch-cluster.sh image consistency and failure diagnostic tests.
# All Docker and SSH operations are handled by fake commands.

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
    TEST_LOG="$CASE_DIR/commands.log"
    OUTPUT_LOG="$CASE_DIR/output.log"

    mkdir -p "$FIXTURE_DIR" "$FAKE_BIN_DIR"
    cp "$PROJECT_DIR/launch-cluster.sh" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/autodiscover.sh" "$FIXTURE_DIR/"
    mkdir -p "$FIXTURE_DIR/docker"
    cp "$PROJECT_DIR/docker/image_identity.py" "$FIXTURE_DIR/docker/"
    LOCAL_INSPECT_FILE="$PROJECT_DIR/tests/fixtures/image-identity/classic.json"
    REMOTE_INSPECT_FILE="$PROJECT_DIR/tests/fixtures/image-identity/containerd.json"
    touch "$FIXTURE_DIR/test.env"
    : > "$TEST_LOG"
    : > "$OUTPUT_LOG"

    cat > "$FAKE_BIN_DIR/docker" <<'DOCKER'
#!/bin/bash
set -euo pipefail
echo "docker $*" >> "$TEST_LOG"

if [[ "${1:-}" == "ps" ]]; then
    [[ "${EXISTING_CONTAINER:-false}" != true ]] || echo vllm_node
    exit 0
fi

if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
    if [[ "${LOCAL_IMAGE_MISSING:-false}" == "true" ]]; then
        exit 1
    fi
    if [[ "${3:-}" == "--format" ]]; then
        echo "${LOCAL_IMAGE_ID:-sha256:head}"
    else
        cat "$LOCAL_INSPECT_FILE"
        exit "${LOCAL_INSPECT_STATUS:-0}"
    fi
    exit 0
fi

if [[ "${1:-}" == "exec" && ( "${2:-}" == -i || "${2:-}" == -it ) ]]; then
    exec python3 -c '
import os, pathlib, sys, time
log = os.environ["TEST_LOG"]
pathlib.Path(log + ".head-pid").write_text(str(os.getpid()))
pathlib.Path(log + ".launcher-pid").write_text(str(os.getppid()))
if os.environ.get("READ_HEAD_STDIN") == "true":
    pathlib.Path(log + ".stdin").write_text(sys.stdin.read())
time.sleep(float(os.environ.get("HEAD_EXEC_DELAY", "0")))
sys.exit(int(os.environ.get("HEAD_EXEC_STATUS", "0")))'
fi

if [[ "${1:-}" == "logs" ]]; then
    echo 'earlyoom: head memory report'
    exit "${HEAD_LOG_STATUS:-0}"
fi

exit 0
DOCKER

    cat > "$FAKE_BIN_DIR/ssh" <<'SSH'
#!/bin/bash
set -euo pipefail
echo "ssh $*" >> "$TEST_LOG"

if [[ "$*" == *"docker ps --format"* ]]; then
    if [[ "${EXISTING_CONTAINER:-false}" == true ]]; then
        echo vllm_node
        exit 0
    fi
    exit 1
fi

if [[ "$*" == *"docker image inspect"* ]]; then
    if [[ "${REMOTE_IMAGE_MISSING:-false}" == "true" ]]; then
        exit 1
    fi
    if [[ "$*" == *"--format"* ]]; then
        echo "${REMOTE_IMAGE_ID:-sha256:head}"
    else
        cat "$REMOTE_INSPECT_FILE"
        exit "${REMOTE_INSPECT_STATUS:-0}"
    fi
fi

if [[ "$*" == *"docker logs"* ]]; then
    echo 'Worker_TP1: original startup exception'
    exit "${WORKER_LOG_STATUS:-0}"
fi

if [[ "$*" == *"docker exec vllm_node bash -c "* ]]; then
    # Model an attached SSH/docker client. The process itself stays alive until
    # the remote command exits, or until the launcher terminates the client.
    export MOCK_WORKER_FAIL=true
    if [[ -n "${FAIL_WORKER_NODE:-}" && "$*" != *"$FAIL_WORKER_NODE docker exec "* ]]; then
        export MOCK_WORKER_FAIL=false
    fi
    exec python3 -c '
import os, pathlib, sys, time
pathlib.Path(os.environ["TEST_LOG"] + ".worker-" + str(os.getpid())).touch()
failing = os.environ["MOCK_WORKER_FAIL"] == "true"
time.sleep(float(os.environ.get("WORKER_EXEC_DELAY", "60")) if failing else 60)
sys.exit(int(os.environ.get("WORKER_EXEC_STATUS", "0")) if failing else 0)'
fi

exit 0
SSH

    cat > "$FAKE_BIN_DIR/sleep" <<'SLEEP'
#!/bin/bash
exit 0
SLEEP
    cat > "$FAKE_BIN_DIR/scp" <<'SCP'
#!/bin/bash
echo "scp $*" >> "$TEST_LOG"
exit 0
SCP

    chmod +x "$FAKE_BIN_DIR/docker" "$FAKE_BIN_DIR/ssh" "$FAKE_BIN_DIR/sleep" "$FAKE_BIN_DIR/scp"
}

run_launch() {
    (
        cd "$FIXTURE_DIR"
        PATH="$FAKE_BIN_DIR:$PATH" \
            TEST_LOG="$TEST_LOG" \
            LOCAL_INSPECT_FILE="$LOCAL_INSPECT_FILE" \
            REMOTE_INSPECT_FILE="$REMOTE_INSPECT_FILE" \
            LOCAL_IP="10.0.0.1" \
            ./launch-cluster.sh \
                --config "$FIXTURE_DIR/test.env" \
                --nodes "10.0.0.1,10.0.0.2" \
                --eth-if eth0 \
                --ib-if ib0 \
                --no-cache-dirs \
                -d "$@" start
    ) > "$OUTPUT_LOG" 2>&1
}

run_exec() {
    (
        cd "$FIXTURE_DIR"
        action_args=(exec vllm serve test-model --tensor-parallel-size "${EXEC_TEST_TP:-2}")
        if [[ -n "${EXEC_TEST_SCRIPT:-}" ]]; then
            action_args=(--launch-script "$EXEC_TEST_SCRIPT")
        fi
        PATH="$FAKE_BIN_DIR:$PATH" \
            TEST_LOG="$TEST_LOG" \
            LOCAL_INSPECT_FILE="$LOCAL_INSPECT_FILE" \
            REMOTE_INSPECT_FILE="$REMOTE_INSPECT_FILE" \
            LOCAL_IP="10.0.0.1" \
            exec ./launch-cluster.sh \
                --config "$FIXTURE_DIR/test.env" \
                --nodes "${EXEC_TEST_NODES:-10.0.0.1,10.0.0.2}" \
                --eth-if eth0 --ib-if ib0 --no-cache-dirs \
                "$@" "${action_args[@]}"
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

test_matching_images_launch_cluster() {
    setup_fixture
    REMOTE_IMAGE_ID="sha256:head" run_launch || fail "launch failed for matching image IDs"
    assert_output_contains 'Docker image consistency check passed\.'
    assert_output_contains '\[WORKER\] 10\.0\.0\.2: sha256:head \(match\)'
    assert_log_contains '^docker run '
    pass "matching image IDs allow cluster launch"
}

test_unlimited_memlock_on_every_node() {
    local mode
    local mode_args=()
    for mode in privileged non-privileged; do
        setup_fixture
        mode_args=()
        [[ "$mode" == privileged ]] || mode_args=(--non-privileged)
        run_launch "${mode_args[@]}" || fail "$mode launch failed"
        assert_log_contains '^docker run .*--ulimit memlock=-1:-1 '
        assert_log_contains '^ssh .*docker run .*--ulimit memlock=-1:-1 '
    done
    pass "head and workers have unlimited memlock in both privilege modes"
}

test_mismatched_image_aborts_before_launch() {
    setup_fixture
    REMOTE_INSPECT_FILE="$CASE_DIR/different.json"
    sed 's/2222222222/3333333333/g' "$LOCAL_INSPECT_FILE" > "$REMOTE_INSPECT_FILE"
    if REMOTE_IMAGE_ID="sha256:worker" run_launch; then
        fail "launch unexpectedly succeeded for mismatched image IDs"
    fi
    assert_output_contains 'Docker image mismatch on worker node \(10\.0\.0\.2\)'
    assert_output_contains 'Head:   sha256:head'
    assert_output_contains 'Worker: sha256:worker'
    assert_output_contains 'Cluster launch aborted because image'
    assert_log_not_contains '^docker run '
    pass "mismatched image ID aborts before containers start"
}

test_missing_worker_image_aborts_before_launch() {
    setup_fixture
    if REMOTE_IMAGE_MISSING="true" run_launch; then
        fail "launch unexpectedly succeeded with a missing worker image"
    fi
    assert_output_contains "Could not inspect image 'vllm-node' on worker node \\(10\\.0\\.0\\.2\\)"
    assert_output_contains 'image may be missing or inaccessible'
    assert_log_not_contains '^docker run '
    pass "missing worker image aborts before containers start"
}

test_mixed_image_stores_launch_cluster() {
    setup_fixture
    REMOTE_IMAGE_ID="sha256:manifest" run_launch || fail "launch failed for equivalent image content"
    assert_output_contains 'Docker image consistency check passed\.'
    assert_output_contains '\[WORKER\] 10\.0\.0\.2: sha256:manifest \(matching image content\)'
    assert_log_contains '^docker image inspect sha256:head$'
    assert_log_contains '^ssh .* docker image inspect sha256:manifest$'
    assert_log_contains '^docker run '
    pass "different IDs with matching content allow mixed-store cluster launch"
}

test_changed_config_aborts_before_launch() {
    setup_fixture
    REMOTE_INSPECT_FILE="$CASE_DIR/different-config.json"
    sed 's/"bash"/"python3"/' "$LOCAL_INSPECT_FILE" > "$REMOTE_INSPECT_FILE"
    if REMOTE_IMAGE_ID="sha256:worker" run_launch; then
        fail "launch unexpectedly succeeded for different runtime configurations"
    fi
    assert_output_contains 'Docker image mismatch on worker node'
    assert_log_not_contains '^docker run '
    pass "matching layers with different runtime config abort launch"
}

test_unreadable_content_aborts_before_launch() {
    for side in local remote; do
        for failure in malformed command; do
            setup_fixture
            if [[ "$failure" == "malformed" ]]; then
                if [[ "$side" == "local" ]]; then
                    LOCAL_INSPECT_FILE=/dev/null
                else
                    REMOTE_INSPECT_FILE=/dev/null
                fi
                if REMOTE_IMAGE_ID="sha256:worker" run_launch; then
                    fail "launch succeeded with malformed $side metadata"
                fi
            elif [[ "$side" == "local" ]]; then
                if LOCAL_INSPECT_STATUS=1 REMOTE_IMAGE_ID="sha256:worker" run_launch; then
                    fail "launch succeeded despite failed local inspection"
                fi
            else
                if REMOTE_INSPECT_STATUS=1 REMOTE_IMAGE_ID="sha256:worker" run_launch; then
                    fail "launch succeeded despite failed remote inspection"
                fi
            fi
            assert_output_contains 'Cluster launch aborted because image'
            assert_log_not_contains '^docker run '
        done
    done
    pass "malformed metadata and failed inspections cannot allow a mismatched ID"
}

test_missing_head_image_aborts_before_launch() {
    setup_fixture
    if LOCAL_IMAGE_MISSING="true" run_launch; then
        fail "launch unexpectedly succeeded with a missing head image"
    fi
    assert_output_contains "Could not inspect image 'vllm-node' on head node"
    assert_log_not_contains '^docker run '
    pass "missing head image aborts before containers start"
}

test_failed_exec_captures_logs_before_cleanup() {
    setup_fixture
    local status=0
    HEAD_EXEC_STATUS=42 run_exec || status=$?
    [[ "$status" == 42 ]] || fail "head exit status was not preserved: $status"
    assert_output_contains 'earlyoom: head memory report'
    assert_output_contains 'Worker_TP1: original startup exception'
    assert_log_contains '^docker logs --timestamps --tail 200 vllm_node$'
    assert_log_contains '^ssh .*ConnectTimeout=5 .*docker logs --timestamps --tail 200 vllm_node$'
    local worker_log_line first_stop_line
    worker_log_line=$(grep -nE '^ssh .*docker logs ' "$TEST_LOG" | cut -d: -f1)
    first_stop_line=$(grep -nE '^docker stop ' "$TEST_LOG" | head -1 | cut -d: -f1)
    [[ "$worker_log_line" -lt "$first_stop_line" ]] || fail "containers stopped before worker log collection"
    pass "failed foreground exec captures head and worker logs before cleanup"
}

test_log_errors_preserve_exec_status_and_cleanup() {
    setup_fixture
    local status=0
    HEAD_EXEC_STATUS=42 HEAD_LOG_STATUS=1 WORKER_LOG_STATUS=255 run_exec || status=$?
    [[ "$status" == 42 ]] || fail "log collection replaced head exit status: $status"
    assert_output_contains 'Could not collect head container logs'
    assert_output_contains 'Could not collect worker container logs'
    assert_log_contains '^docker stop vllm_node$'
    assert_log_contains '^ssh .*docker stop vllm_node$'
    pass "unavailable logs preserve the original failure and cluster cleanup"
}

test_successful_and_daemon_exec_do_not_dump_logs() {
    setup_fixture
    run_exec || fail "successful foreground exec failed"
    assert_log_not_contains 'docker logs '
    setup_fixture
    HEAD_EXEC_STATUS=42 run_exec -d || fail "daemon dispatch failed"
    assert_log_not_contains 'docker logs '
    assert_log_not_contains 'docker stop '
    pass "successful foreground exec and daemon dispatch do not dump logs"
}

test_solo_failure_captures_only_local_logs() {
    setup_fixture
    local status=0
    HEAD_EXEC_STATUS=42 run_exec --solo || status=$?
    [[ "$status" == 42 ]] || fail "solo exit status was not preserved: $status"
    assert_output_contains 'earlyoom: head memory report'
    assert_log_not_contains '^ssh '
    pass "solo failure captures local container logs without contacting peers"
}

assert_exec_clients_stopped() {
    local file pid
    for file in "$TEST_LOG.head-pid" "$TEST_LOG".worker-*; do
        [[ -e "$file" ]] || continue
        if [[ "$file" == *.head-pid ]]; then pid=$(< "$file"); else pid=${file##*.worker-}; fi
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            fail "launcher left client $pid running"
        fi
    done
}

test_worker_exit_fails_without_waiting_for_head() {
    local code expected status started
    for code in 17 0 255; do
        setup_fixture
        status=0
        expected=$code
        [[ "$expected" -ne 0 ]] || expected=1
        started=$SECONDS
        HEAD_EXEC_DELAY=60 WORKER_EXEC_DELAY=0.1 WORKER_EXEC_STATUS=$code run_exec || status=$?
        [[ "$status" == "$expected" ]] || fail "worker status $code returned $status"
        [[ $((SECONDS - started)) -lt 10 ]] || fail "worker failure waited for the head"
        assert_output_contains "Worker rank 1 on 10.0.0.2 exited with status $code"
        assert_output_contains 'Worker_TP1: original startup exception'
        assert_log_contains '^ssh -n .*ServerAliveInterval=5 .*ServerAliveCountMax=3 .*docker exec vllm_node '
        assert_log_contains '^docker stop vllm_node$'
        assert_exec_clients_stopped
    done
    pass "worker failures, clean early exits and SSH disconnects promptly fail the launch"
}

test_second_worker_failure_identifies_rank_and_node() {
    setup_fixture
    local status=0
    EXEC_TEST_NODES=10.0.0.1,10.0.0.2,10.0.0.3 EXEC_TEST_TP=3 \
        FAIL_WORKER_NODE=10.0.0.3 HEAD_EXEC_DELAY=60 WORKER_EXEC_DELAY=0.1 \
        WORKER_EXEC_STATUS=23 run_exec || status=$?
    [[ "$status" == 23 ]] || fail "second worker status was not preserved: $status"
    assert_output_contains 'Worker rank 2 on 10.0.0.3 exited with status 23'
    assert_exec_clients_stopped
    pass "a later worker failure identifies its rank and node and reaps all clients"
}

test_supervised_head_preserves_stdin_and_status() {
    setup_fixture
    READ_HEAD_STDIN=true HEAD_EXEC_DELAY=0.1 run_exec <<< 'head input' || fail 'head launch failed'
    [[ "$(< "$TEST_LOG.stdin")" == 'head input' ]] || fail 'head stdin was lost to background execution'
    assert_exec_clients_stopped
    pass "supervised foreground head retains stdin and reaps worker clients on success"
}

test_recipe_launch_script_worker_failure() {
    setup_fixture
    local status=0
    printf '#!/bin/bash\nvllm serve test-model --tensor-parallel-size 2\n' > "$FIXTURE_DIR/recipe.sh"
    EXEC_TEST_SCRIPT="$FIXTURE_DIR/recipe.sh" HEAD_EXEC_DELAY=60 WORKER_EXEC_DELAY=0.1 \
        WORKER_EXEC_STATUS=31 run_exec || status=$?
    [[ "$status" == 31 ]] || fail "recipe script worker status was lost: $status"
    assert_output_contains 'Worker rank 1 on 10.0.0.2 exited with status 31'
    assert_log_contains '^scp '
    assert_log_contains '^ssh -n .*docker exec vllm_node bash -c /workspace/exec-script.sh'
    assert_exec_clients_stopped
    pass "recipe-generated launch scripts receive the same worker failure supervision"
}

test_failure_preserves_existing_cluster() {
    setup_fixture
    local status=0
    EXISTING_CONTAINER=true HEAD_EXEC_DELAY=60 WORKER_EXEC_DELAY=0.1 \
        WORKER_EXEC_STATUS=19 run_exec || status=$?
    [[ "$status" == 19 ]] || fail "existing-cluster worker status was lost: $status"
    assert_output_contains 'Skipping cleanup'
    assert_log_not_contains 'docker stop '
    assert_exec_clients_stopped
    pass "worker failure reaps clients without stopping pre-existing containers"
}

test_signal_stops_supervisor_and_clients() {
    setup_fixture
    local launch_pid status=0 i launcher_pid
    HEAD_EXEC_DELAY=60 run_exec &
    launch_pid=$!
    for ((i=0; i<200; i++)); do
        [[ ! -s "$TEST_LOG.launcher-pid" ]] || break
        /bin/sleep 0.02
    done
    [[ -s "$TEST_LOG.launcher-pid" ]] || fail 'head did not start for signal test'
    launcher_pid=$(< "$TEST_LOG.launcher-pid")
    kill -TERM "$launcher_pid"
    wait "$launch_pid" || status=$?
    [[ "$status" == 143 ]] || fail "SIGTERM exit status was not preserved: $status"
    assert_log_contains '^docker stop vllm_node$'
    assert_exec_clients_stopped
    pass "SIGTERM terminates supervision, reaps clients and cleans up owned containers"
}

test_matching_images_launch_cluster
test_unlimited_memlock_on_every_node
test_mismatched_image_aborts_before_launch
test_missing_worker_image_aborts_before_launch
test_mixed_image_stores_launch_cluster
test_changed_config_aborts_before_launch
test_unreadable_content_aborts_before_launch
test_missing_head_image_aborts_before_launch
test_failed_exec_captures_logs_before_cleanup
test_log_errors_preserve_exec_status_and_cleanup
test_successful_and_daemon_exec_do_not_dump_logs
test_solo_failure_captures_only_local_logs
test_worker_exit_fails_without_waiting_for_head
test_second_worker_failure_identifies_rank_and_node
test_supervised_head_preserves_stdin_and_status
test_recipe_launch_script_worker_failure
test_failure_preserves_existing_cluster
test_signal_stops_supervisor_and_clients

echo "All $TESTS_PASSED launch-cluster image consistency and diagnostic tests passed."
