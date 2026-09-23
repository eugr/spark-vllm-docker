#!/bin/bash
#
# lib.sh - Shared harness for the test suites (sourced, not executed).
#
# Callers set VERBOSE (to "-v" for verbose output) before sourcing.
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

log_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
}

log_verbose() {
    if [[ "$VERBOSE" == "-v" ]]; then
        echo "       $1"
    fi
    return 0
}

# assert_contains <description> <haystack> <needle>
assert_contains() {
    if [[ "$2" == *"$3"* ]]; then
        log_pass "$1"
    else
        log_fail "$1"
        log_verbose "expected to find: $3"
        log_verbose "in output:"
        log_verbose "$2"
    fi
}
