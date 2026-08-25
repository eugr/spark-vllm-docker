#!/bin/bash
#
# Focused tests for launch-time vLLM PR application. Docker and patch downloads
# are mocked; the generated mod runner uses a real git apply against a fixture.

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
        sed -n '1,240p' "$OUTPUT_LOG" >&2
    fi
    if [[ -f "${COMMAND_LOG:-}" ]]; then
        echo "--- command log ---" >&2
        sed -n '1,240p' "$COMMAND_LOG" >&2
    fi
    exit 1
}

setup_fixture() {
    TEST_INDEX=$((TEST_INDEX + 1))
    CASE_DIR="$TMP_BASE/case-$TEST_INDEX"
    FIXTURE_DIR="$CASE_DIR/project"
    FAKE_BIN_DIR="$CASE_DIR/bin"
    FAKE_CONTAINER_ROOT="$CASE_DIR/container"
    COMMAND_LOG="$CASE_DIR/commands.log"
    LAYER_LOG="$CASE_DIR/layers.log"
    OUTPUT_LOG="$CASE_DIR/output.log"
    FAKE_PR_DIFF="$CASE_DIR/pr.diff"

    mkdir -p \
        "$FIXTURE_DIR/mod-a" \
        "$FIXTURE_DIR/mod-b" \
        "$FAKE_BIN_DIR" \
        "$FAKE_CONTAINER_ROOT/site-packages/vllm"
    cp "$PROJECT_DIR/launch-cluster.sh" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/autodiscover.sh" "$FIXTURE_DIR/"
    touch "$FIXTURE_DIR/test.env"
    printf 'old_value = 1\n' > "$FAKE_CONTAINER_ROOT/site-packages/vllm/runtime_test.py"
    : > "$COMMAND_LOG"
    : > "$LAYER_LOG"
    : > "$OUTPUT_LOG"

    cat > "$FIXTURE_DIR/mod-a/run.sh" <<'MOD_A'
#!/bin/bash
echo "[mod-a] applied"
MOD_A
    cat > "$FIXTURE_DIR/mod-b/run.sh" <<'MOD_B'
#!/bin/bash
echo "[mod-b] applied"
MOD_B
    chmod +x "$FIXTURE_DIR/mod-a/run.sh" "$FIXTURE_DIR/mod-b/run.sh"

    cat > "$FAKE_PR_DIFF" <<'DIFF'
diff --git a/vllm/runtime_test.py b/vllm/runtime_test.py
--- a/vllm/runtime_test.py
+++ b/vllm/runtime_test.py
@@ -1 +1 @@
-old_value = 1
+new_value = 2
diff --git a/tests/test_runtime_test.py b/tests/test_runtime_test.py
--- a/tests/test_runtime_test.py
+++ b/tests/test_runtime_test.py
@@ -1 +1 @@
-old test
+new test
DIFF

    cat > "$FAKE_BIN_DIR/curl" <<'CURL'
#!/bin/bash
set -euo pipefail
echo "curl $*" >> "$COMMAND_LOG"
destination=""
while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "-o" ]]; then
        destination="$2"
        break
    fi
    shift
done
[[ -n "$destination" ]]
cp "$FAKE_PR_DIFF" "$destination"
CURL

    cat > "$FAKE_BIN_DIR/docker" <<'DOCKER'
#!/bin/bash
set -euo pipefail
echo "docker $*" >> "$COMMAND_LOG"

if [[ "${1:-}" == "ps" ]]; then
    if [[ "${EXISTING_CONTAINER:-false}" == "true" ]]; then
        echo "vllm_node"
    fi
    exit 0
fi

if [[ "${1:-}" == "image" && "${2:-}" == "inspect" ]]; then
    echo "sha256:test-image"
    exit 0
fi

if [[ "${1:-}" == "cp" ]]; then
    source_path="${2%/.}"
    container_path="${3#*:}"
    destination="$FAKE_CONTAINER_ROOT$container_path"
    mkdir -p "$destination"
    cp -a "$source_path/." "$destination/"
    exit 0
fi

if [[ "${1:-}" == "exec" && "$*" == *" mkdir -p /workspace/mods/"* ]]; then
    container_path="${*: -1}"
    mkdir -p "$FAKE_CONTAINER_ROOT$container_path"
    exit 0
fi

if [[ "${1:-}" == "exec" && "$*" == *"chmod +x run.sh && ./run.sh"* ]]; then
    mod_name="$(printf '%s\n' "$*" | sed -n 's#.*cd /workspace/mods/\([^ ]*\).*#\1#p')"
    [[ -n "$mod_name" ]]
    echo "$mod_name" >> "$LAYER_LOG"
    (
        cd "$FAKE_CONTAINER_ROOT/workspace/mods/$mod_name"
        VLLM_PACKAGE_DIR="$FAKE_CONTAINER_ROOT/site-packages/vllm" ./run.sh
    )
    exit 0
fi

exit 0
DOCKER

    cat > "$FAKE_BIN_DIR/sleep" <<'SLEEP'
#!/bin/bash
exit 0
SLEEP

    chmod +x "$FAKE_BIN_DIR/curl" "$FAKE_BIN_DIR/docker" "$FAKE_BIN_DIR/sleep"
}

run_launch() {
    (
        cd "$FIXTURE_DIR"
        PATH="$FAKE_BIN_DIR:$PATH" \
            COMMAND_LOG="$COMMAND_LOG" \
            LAYER_LOG="$LAYER_LOG" \
            FAKE_PR_DIFF="$FAKE_PR_DIFF" \
            FAKE_CONTAINER_ROOT="$FAKE_CONTAINER_ROOT" \
            EXISTING_CONTAINER="${EXISTING_CONTAINER:-false}" \
            ./launch-cluster.sh \
                --config "$FIXTURE_DIR/test.env" \
                --solo \
                --eth-if eth0 \
                --ib-if ib0 \
                --no-cache-dirs \
                "$@" \
                -d start
    ) > "$OUTPUT_LOG" 2>&1
}

assert_output_contains() {
    local pattern="$1"
    grep -Eq "$pattern" "$OUTPUT_LOG" || fail "Expected output to match: $pattern"
}

assert_log_not_contains() {
    local pattern="$1"
    if grep -Eq "$pattern" "$COMMAND_LOG"; then
        fail "Expected command log not to match: $pattern"
    fi
}

test_runtime_pr_is_applied_in_cli_layer_order() {
    setup_fixture
    run_launch \
        --apply-mod "$FIXTURE_DIR/mod-a" \
        --apply-vllm-pr 12345 \
        --apply-mod "$FIXTURE_DIR/mod-b" \
        || fail "runtime PR launch failed"

    grep -q '^new_value = 2$' "$FAKE_CONTAINER_ROOT/site-packages/vllm/runtime_test.py" \
        || fail "runtime PR did not patch the installed vLLM fixture"
    [[ "$(sed -n '1p' "$LAYER_LOG")" == "mod-a" ]] || fail "mod-a was not first"
    [[ "$(sed -n '2p' "$LAYER_LOG")" == vllm-pr-12345-* ]] || fail "runtime PR was not second"
    [[ "$(sed -n '3p' "$LAYER_LOG")" == "mod-b" ]] || fail "mod-b was not third"
    [[ "$(grep -c '^curl ' "$COMMAND_LOG")" -eq 1 ]] || fail "PR diff was not fetched exactly once"
    assert_output_contains 'Validated vLLM PR #12345 for runtime application: 1 package path\(s\), 1 test/docs path\(s\) ignored\.'
    assert_output_contains '\[vllm-pr #12345\] Applied successfully\.'
    pass "runtime PR is fetched once and applied in CLI layer order"
}

test_build_only_pr_is_rejected_before_container_start() {
    setup_fixture
    cat > "$FAKE_PR_DIFF" <<'DIFF'
diff --git a/csrc/runtime_test.cu b/csrc/runtime_test.cu
--- a/csrc/runtime_test.cu
+++ b/csrc/runtime_test.cu
@@ -1 +1 @@
-old kernel
+new kernel
DIFF

    if run_launch --apply-vllm-pr 23456; then
        fail "native-code PR unexpectedly succeeded"
    fi
    assert_output_contains 'vLLM PR #23456 is not runtime-only'
    assert_output_contains 'Use build-and-copy\.sh --apply-vllm-pr 23456 instead\.'
    assert_log_not_contains '^docker run '
    pass "native/build PR is rejected before containers start"
}

test_repeated_pr_is_downloaded_once_and_idempotent() {
    setup_fixture
    run_launch --apply-vllm-pr 12345 --apply-vllm-pr 12345 \
        || fail "repeated runtime PR launch failed"

    grep -q '^new_value = 2$' "$FAKE_CONTAINER_ROOT/site-packages/vllm/runtime_test.py" \
        || fail "repeated runtime PR did not leave the expected patched source"
    [[ "$(grep -c '^curl ' "$COMMAND_LOG")" -eq 1 ]] \
        || fail "repeated PR was downloaded more than once"
    assert_output_contains '\[vllm-pr #12345\] Patch is already applied; skipping\.'
    pass "repeated runtime PR reuses its download and applies idempotently"
}

test_existing_cluster_refuses_unverifiable_runtime_pr() {
    setup_fixture
    if EXISTING_CONTAINER=true run_launch --apply-vllm-pr 34567; then
        fail "runtime PR unexpectedly succeeded against an existing container"
    fi
    assert_output_contains 'cluster containers are already running'
    assert_output_contains '\-\-apply-vllm-pr cannot be verified or applied'
    assert_log_not_contains '^curl '
    pass "existing cluster refuses an unapplied runtime PR request"
}

test_invalid_pr_number_is_rejected() {
    setup_fixture
    if run_launch --apply-vllm-pr '123;touch-bad'; then
        fail "invalid PR number unexpectedly succeeded"
    fi
    assert_output_contains 'requires a positive integer PR number'
    assert_log_not_contains '^curl '
    pass "invalid PR number is rejected before launch"
}

test_runtime_pr_is_applied_in_cli_layer_order
test_build_only_pr_is_rejected_before_container_start
test_repeated_pr_is_downloaded_once_and_idempotent
test_existing_cluster_refuses_unverifiable_runtime_pr
test_invalid_pr_number_is_rejected

echo "All $TESTS_PASSED launch-cluster runtime vLLM PR tests passed."
