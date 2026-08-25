#!/bin/bash
#
# Focused behavior tests for launch-cluster.sh image consistency checks.
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
    touch "$FIXTURE_DIR/test.env"
    : > "$TEST_LOG"
    : > "$OUTPUT_LOG"

    cat > "$FAKE_BIN_DIR/docker" <<'DOCKER'
#!/bin/bash
set -euo pipefail
echo "docker $*" >> "$TEST_LOG"

if [[ "${1:-}" == "ps" ]]; then
    exit 0
fi

if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
    if [[ "${LOCAL_IMAGE_MISSING:-false}" == "true" ]]; then
        exit 1
    fi
    echo "${LOCAL_IMAGE_ID:-sha256:head}"
    exit 0
fi

exit 0
DOCKER

    cat > "$FAKE_BIN_DIR/ssh" <<'SSH'
#!/bin/bash
set -euo pipefail
echo "ssh $*" >> "$TEST_LOG"

if [[ "$*" == *"docker ps --format"* ]]; then
    exit 1
fi

if [[ "$*" == *"docker image inspect"* ]]; then
    if [[ "${REMOTE_IMAGE_MISSING:-false}" == "true" ]]; then
        exit 1
    fi
    echo "${REMOTE_IMAGE_ID:-sha256:head}"
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
            LOCAL_IP="10.0.0.1" \
            ./launch-cluster.sh \
                --config "$FIXTURE_DIR/test.env" \
                --nodes "10.0.0.1,10.0.0.2" \
                --eth-if eth0 \
                --ib-if ib0 \
                --no-cache-dirs \
                -d start
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

test_mismatched_image_aborts_before_launch() {
    setup_fixture
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

test_matching_images_launch_cluster
test_mismatched_image_aborts_before_launch
test_missing_worker_image_aborts_before_launch

echo "All $TESTS_PASSED launch-cluster image consistency tests passed."
