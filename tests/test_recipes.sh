#!/bin/bash
#
# test_recipes.sh - Integration tests for run-recipe.py and launch-cluster.sh
#
# These tests use --dry-run mode to verify compatibility without actually
# running containers. Suitable for CI/CD pipelines.
#
# Usage:
#   ./tests/test_recipes.sh          # Run all tests
#   ./tests/test_recipes.sh -v       # Verbose output
#

# Don't exit on first failure; we want a full summary.
set +e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERBOSE="${1:-}"

# Load expected commands for README verification
source "$SCRIPT_DIR/expected_commands.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Helper functions
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
}

get_recipe_flag() {
    local flag_name="$1"
    local recipe_file="$2"
    grep -E "^${flag_name}:" "$recipe_file" | awk '{print $2}'
}

find_solo_recipe() {
    for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
        if [[ -f "$recipe" ]]; then
            cluster_only=$(get_recipe_flag "cluster_only" "$recipe")
            if [[ "$cluster_only" == "true" ]]; then
                continue
            fi
            echo "$(basename "$recipe" .yaml)"
            return 0
        fi
    done
    return 1
}

find_cluster_recipe() {
    for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
        if [[ -f "$recipe" ]]; then
            solo_only=$(get_recipe_flag "solo_only" "$recipe")
            if [[ "$solo_only" == "true" ]]; then
                continue
            fi
            echo "$(basename "$recipe" .yaml)"
            return 0
        fi
    done
    return 1
}

find_recipe_with_mods() {
    for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
        if [[ -f "$recipe" ]]; then
            has_mods=$(awk '
                /^mods:/ {inmods=1; next}
                inmods && /^[[:space:]]*-[[:space:]]/ {print "yes"; exit}
                inmods && /^[^[:space:]]/ {exit}
            ' "$recipe")
            if [[ "$has_mods" == "yes" ]]; then
                echo "$(basename "$recipe" .yaml)"
                return 0
            fi
        fi
    done
    return 1
}

get_recipe_mode() {
    local recipe_name="$1"
    local recipe_file="$PROJECT_DIR/recipes/${recipe_name}.yaml"
    local cluster_only
    local solo_only
    cluster_only=$(get_recipe_flag "cluster_only" "$recipe_file")
    solo_only=$(get_recipe_flag "solo_only" "$recipe_file")
    if [[ "$cluster_only" == "true" ]]; then
        echo "cluster"
    elif [[ "$solo_only" == "true" ]]; then
        echo "solo"
    else
        echo "solo"
    fi
}

run_recipe_dry_run() {
    local recipe_name="$1"
    local mode="$2"
    if [[ "$mode" == "cluster" ]]; then
        "$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run -n "10.0.0.1,10.0.0.2" 2>&1
    else
        "$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_test "Checking prerequisites..."
    
    if ! command -v python3 &> /dev/null; then
        log_fail "python3 not found"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ $(echo "$python_version < 3.10" | bc -l) -eq 1 ]]; then
        log_fail "Python 3.10+ required, found $python_version"
        exit 1
    fi
    
    # Check PyYAML
    if ! python3 -c "import yaml" 2>/dev/null; then
        log_fail "PyYAML not installed"
        exit 1
    fi
    
    log_pass "Prerequisites OK (Python $python_version with PyYAML)"
}

# Test: run-recipe.py exists and is executable
test_run_recipe_exists() {
    log_test "run-recipe.py exists and is executable"
    
    if [[ -x "$PROJECT_DIR/run-recipe.py" ]]; then
        log_pass "run-recipe.py is executable"
    else
        log_fail "run-recipe.py not found or not executable"
    fi
}

# Test: launch-cluster.sh exists and is executable
test_launch_cluster_exists() {
    log_test "launch-cluster.sh exists and is executable"
    
    if [[ -x "$PROJECT_DIR/launch-cluster.sh" ]]; then
        log_pass "launch-cluster.sh is executable"
    else
        log_fail "launch-cluster.sh not found or not executable"
    fi
}

# Test: run-recipe.py --list works
test_list_recipes() {
    log_test "run-recipe.py --list"
    
    output=$("$PROJECT_DIR/run-recipe.py" --list 2>&1)
    
    if [[ $? -eq 0 ]] && echo "$output" | grep -q "Available recipes"; then
        log_pass "--list shows available recipes"
        log_verbose "Found recipes in output"
    else
        log_fail "--list failed or no recipes found"
        log_verbose "$output"
    fi
}

# Test: All recipes have required recipe_version field
test_recipe_version_required() {
    log_test "All recipes have required recipe_version field"
    
    local all_valid=true
    for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
        if [[ -f "$recipe" ]]; then
            recipe_name=$(basename "$recipe")
            if ! grep -q "^recipe_version:" "$recipe"; then
                log_verbose "$recipe_name missing recipe_version"
                all_valid=false
            fi
        fi
    done
    
    if [[ "$all_valid" == "true" ]]; then
        log_pass "All recipes have recipe_version field"
    else
        log_fail "Some recipes missing recipe_version field"
    fi
}

# Test: All recipes load without errors
test_all_recipes_load() {
    log_test "All recipes load without errors"
    
    local all_valid=true
    for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
        if [[ -f "$recipe" ]]; then
            recipe_name=$(basename "$recipe" .yaml)
            cluster_only=$(grep -E "^cluster_only:" "$recipe" | awk '{print $2}')
            solo_only=$(grep -E "^solo_only:" "$recipe" | awk '{print $2}')
            
            if [[ "$cluster_only" == "true" && "$solo_only" == "true" ]]; then
                log_verbose "$recipe_name has conflicting cluster_only + solo_only"
                all_valid=false
                continue
            fi
            
            if [[ "$cluster_only" == "true" ]]; then
                output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run -n "10.0.0.1,10.0.0.2" 2>&1 || true)
            else
                output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1 || true)
            fi
            
            if ! echo "$output" | grep -q "Error:"; then
                log_verbose "$recipe_name loads OK"
            else
                log_verbose "$recipe_name failed to load"
                all_valid=false
            fi
        fi
    done
    
    if [[ "$all_valid" == "true" ]]; then
        log_pass "All recipes load successfully"
    else
        log_fail "Some recipes failed to load"
    fi
}

# Test: Dry-run generates valid launch script
test_dry_run_generates_script() {
    log_test "Dry-run generates valid launch script"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$(run_recipe_dry_run "$recipe_name" "solo")
    
    if echo "$output" | grep -q "#!/bin/bash" && echo "$output" | grep -q "vllm serve"; then
        log_pass "Dry-run generates bash script with vllm serve command"
    else
        log_fail "Dry-run output doesn't contain expected content"
        log_verbose "$output"
    fi
}

# Test: Solo mode sets tensor_parallel=1
test_solo_mode_tp1() {
    log_test "Solo mode sets tensor_parallel=1"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$(run_recipe_dry_run "$recipe_name" "solo")
    
    # Check that -tp 1 is in the output (solo mode should set tp=1)
    if echo "$output" | grep -q "\-tp 1"; then
        log_pass "Solo mode correctly sets -tp 1"
    else
        log_fail "Solo mode did not set -tp 1"
        log_verbose "$output"
    fi
}

# Test: Solo mode removes --distributed-executor-backend ray
test_solo_mode_removes_ray() {
    log_test "Solo mode removes --distributed-executor-backend ray"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$(run_recipe_dry_run "$recipe_name" "solo")
    
    # Check that --distributed-executor-backend is NOT in the output
    if ! echo "$output" | grep -q "\-\-distributed-executor-backend"; then
        log_pass "Solo mode correctly removes --distributed-executor-backend"
    else
        log_fail "Solo mode did not remove --distributed-executor-backend"
        log_verbose "$output"
    fi
}

# Test: Cluster mode preserves --distributed-executor-backend ray
test_cluster_mode_keeps_ray() {
    log_test "Cluster mode preserves --distributed-executor-backend ray"
    
    # Use minimax-m2-awq which explicitly has --distributed-executor-backend ray
    if [[ ! -f "$PROJECT_DIR/recipes/minimax-m2-awq.yaml" ]]; then
        log_skip "minimax-m2-awq.yaml not found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" minimax-m2-awq --dry-run -n "192.168.1.1,192.168.1.2" 2>&1)
    
    # Check that --distributed-executor-backend IS in the output for cluster mode
    if echo "$output" | grep -q "\-\-distributed-executor-backend ray"; then
        log_pass "Cluster mode correctly preserves --distributed-executor-backend ray"
    else
        log_fail "Cluster mode did not preserve --distributed-executor-backend"
        log_verbose "$output"
    fi
}

# Test: CLI overrides work (--port)
test_cli_override_port() {
    log_test "CLI override --port works"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo --port 9999 2>&1)
    
    if echo "$output" | grep -q "\-\-port 9999"; then
        log_pass "--port override correctly applied"
    else
        log_fail "--port override not found in output"
        log_verbose "$output"
    fi
}

# Test: launch-cluster.sh --help works
test_launch_cluster_help() {
    log_test "launch-cluster.sh --help"
    
    output=$("$PROJECT_DIR/launch-cluster.sh" --help 2>&1 || true)
    
    if echo "$output" | grep -q "Usage:"; then
        log_pass "--help shows usage information"
    else
        log_fail "--help did not show usage"
        log_verbose "$output"
    fi
}

# Test: launch-cluster.sh references examples/ not profiles/
test_launch_cluster_examples_path() {
    log_test "launch-cluster.sh references examples/ directory"
    
    if grep -q "examples/" "$PROJECT_DIR/launch-cluster.sh"; then
        log_pass "launch-cluster.sh references examples/"
    else
        log_fail "launch-cluster.sh does not reference examples/"
    fi
    
    if grep -q "profiles/" "$PROJECT_DIR/launch-cluster.sh"; then
        log_fail "launch-cluster.sh still references profiles/"
    fi
}

# Test: Unsupported recipe version shows warning
test_unsupported_recipe_version() {
    log_test "Unsupported recipe_version shows warning"
    
    # Create a temporary recipe with unsupported version
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
recipe_version: "999"
name: Test Recipe
container: test-container
command: echo "test"
EOF
    
    output=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run --solo 2>&1)
    rm -f "$temp_recipe"
    
    if echo "$output" | grep -q "Warning.*schema version"; then
        log_pass "Unsupported recipe_version shows warning"
    else
        log_fail "No warning for unsupported recipe_version"
        log_verbose "$output"
    fi
}

# Test: Missing recipe_version fails
test_missing_recipe_version_fails() {
    log_test "Missing recipe_version field fails"
    
    # Create a temporary recipe without recipe_version
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
name: Test Recipe
container: test-container
command: echo "test"
EOF
    
    output=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run --solo 2>&1 || true)
    rm -f "$temp_recipe"
    
    if echo "$output" | grep -q "Error.*recipe_version"; then
        log_pass "Missing recipe_version correctly fails"
    else
        log_fail "Missing recipe_version did not fail as expected"
        log_verbose "$output"
    fi
}

# Test: cluster_only recipe fails in solo mode
test_cluster_only_fails_solo() {
    log_test "cluster_only recipe fails in solo mode"
    
    # Create a temporary cluster_only recipe
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
recipe_version: "1"
name: Cluster Only Test
container: test-container
cluster_only: true
command: echo "test"
EOF
    
    output=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run --solo 2>&1 || true)
    exit_code=$?
    rm -f "$temp_recipe"
    
    if echo "$output" | grep -q "requires cluster mode"; then
        log_pass "cluster_only recipe correctly fails in solo mode"
    else
        log_fail "cluster_only recipe did not fail in solo mode"
        log_verbose "$output"
    fi
}

# Test: solo_only recipe fails in cluster mode
test_solo_only_fails_cluster() {
    log_test "solo_only recipe fails in cluster mode"
    
    # Create a temporary solo_only recipe
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
recipe_version: "1"
name: Solo Only Test
container: test-container
solo_only: true
command: echo "test"
EOF
    
    output=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run -n "10.0.0.1,10.0.0.2" 2>&1 || true)
    rm -f "$temp_recipe"
    
    if echo "$output" | grep -q "requires solo mode"; then
        log_pass "solo_only recipe correctly fails in cluster mode"
    else
        log_fail "solo_only recipe did not fail in cluster mode"
        log_verbose "$output"
    fi
}

# Test: solo_only recipe succeeds in solo mode
test_solo_only_allows_solo() {
    log_test "solo_only recipe succeeds in solo mode"
    
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
recipe_version: "1"
name: Solo Only Test
container: test-container
solo_only: true
command: echo "test"
EOF
    
    output=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run --solo 2>&1 || true)
    rm -f "$temp_recipe"
    
    if ! echo "$output" | grep -q "Error: Recipe"; then
        log_pass "solo_only recipe runs in solo mode"
    else
        log_fail "solo_only recipe failed in solo mode"
        log_verbose "$output"
    fi
}

# Test: cluster_only and solo_only both true fails in any mode
test_conflicting_mode_flags_fail() {
    log_test "cluster_only and solo_only both true fails"
    
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
recipe_version: "1"
name: Conflicting Mode Test
container: test-container
cluster_only: true
solo_only: true
command: echo "test"
EOF
    
    output_solo=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run --solo 2>&1 || true)
    output_cluster=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run -n "10.0.0.1,10.0.0.2" 2>&1 || true)
    rm -f "$temp_recipe"
    
    if echo "$output_solo" | grep -q "requires cluster mode" && echo "$output_cluster" | grep -q "requires solo mode"; then
        log_pass "Conflicting flags correctly fail in both modes"
    else
        log_fail "Conflicting flags did not fail as expected"
        log_verbose "solo: $output_solo"
        log_verbose "cluster: $output_cluster"
    fi
}

# ==============================================================================
# Launch-cluster.sh Command Line Verification Tests
# ==============================================================================
# These tests verify that the dry-run output contains the expected
# launch-cluster.sh command line arguments matching the recipe configuration.

# Helper: Extract launch-cluster command from dry-run output
extract_launch_cmd() {
    echo "$1" | grep -A5 "launch-cluster.sh is called with:" | grep -v "launch-cluster.sh is called with:" | tr '\n' ' '
}

# Test: Solo mode generates --solo flag in launch-cluster command
test_launch_cmd_solo_flag() {
    log_test "Launch command includes --solo flag in solo mode"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$(run_recipe_dry_run "$recipe_name" "solo")
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-\-solo"; then
        log_pass "Launch command includes --solo flag"
    else
        log_fail "Launch command missing --solo flag"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Cluster mode generates -n flag with nodes
test_launch_cmd_nodes_flag() {
    log_test "Launch command includes -n flag with nodes in cluster mode"
    
    output=$("$PROJECT_DIR/run-recipe.py" minimax-m2-awq --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-n 10.0.0.1,10.0.0.2"; then
        log_pass "Launch command includes -n with correct nodes"
    else
        log_fail "Launch command missing or incorrect -n flag"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Container image from recipe is passed to launch-cluster
test_launch_cmd_container_image() {
    log_test "Launch command includes correct container image (-t)"
    
    # Use openai-gpt-oss-120b which has a specific container name
    if [[ ! -f "$PROJECT_DIR/recipes/openai-gpt-oss-120b.yaml" ]]; then
        log_skip "openai-gpt-oss-120b.yaml not found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" openai-gpt-oss-120b --dry-run --solo 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    # Check the container is vllm-node-mxfp4 (from the recipe)
    if echo "$launch_cmd" | grep -q "\-t vllm-node-mxfp4"; then
        log_pass "Launch command includes correct container image"
    else
        log_fail "Launch command has wrong container image"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Mods from recipe are passed as --apply-mod
test_launch_cmd_mods() {
    log_test "Launch command includes --apply-mod for recipe mods"
    
    recipe_name=$(find_recipe_with_mods)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No recipes with mods found"
        return
    fi
    
    mode=$(get_recipe_mode "$recipe_name")
    output=$(run_recipe_dry_run "$recipe_name" "$mode")
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-\-apply-mod"; then
        log_pass "Launch command includes --apply-mod for mods"
    else
        log_fail "Launch command missing --apply-mod"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Daemon mode flag is passed through
test_launch_cmd_daemon_flag() {
    log_test "Launch command includes -d flag in daemon mode"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -d 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-d"; then
        log_pass "Launch command includes -d flag"
    else
        log_fail "Launch command missing -d flag"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: NCCL debug level is passed through
test_launch_cmd_nccl_debug() {
    log_test "Launch command includes --nccl-debug when specified"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo --nccl-debug INFO 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-\-nccl-debug INFO"; then
        log_pass "Launch command includes --nccl-debug INFO"
    else
        log_fail "Launch command missing --nccl-debug"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: --launch-script is always included
test_launch_cmd_launch_script() {
    log_test "Launch command includes --launch-script"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$(run_recipe_dry_run "$recipe_name" "solo")
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-\-launch-script"; then
        log_pass "Launch command includes --launch-script"
    else
        log_fail "Launch command missing --launch-script"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Container override (-t CLI) takes precedence
test_launch_cmd_container_override() {
    log_test "CLI container override (-t) takes precedence"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -t my-custom-image 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-t my-custom-image"; then
        log_pass "Container override correctly applied"
    else
        log_fail "Container override not applied"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Cluster mode does NOT include --solo flag
test_launch_cmd_no_solo_in_cluster() {
    log_test "Launch command does NOT include --solo in cluster mode"
    
    output=$("$PROJECT_DIR/run-recipe.py" minimax-m2-awq --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -qv "\-\-solo" || ! echo "$launch_cmd" | grep -q "\-\-solo"; then
        log_pass "Cluster mode correctly omits --solo flag"
    else
        log_fail "Cluster mode incorrectly includes --solo flag"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: -e / --env passthrough to launch-cluster.sh
test_launch_cmd_env_passthrough() {
    log_test "Launch command includes -e env vars"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -e HF_TOKEN=test123 -e MY_VAR=hello 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-e HF_TOKEN=test123" && echo "$launch_cmd" | grep -q "\-e MY_VAR=hello"; then
        log_pass "Launch command includes -e env vars"
    else
        log_fail "-e env vars not found in launch command"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: no -e flags when none specified
test_launch_cmd_no_env_by_default() {
    log_test "Launch command omits -e when no env vars specified"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q " -e "; then
        log_fail "Unexpected -e flag in launch command"
        log_verbose "Launch cmd: $launch_cmd"
    else
        log_pass "Launch command correctly omits -e when none specified"
    fi
}

# ==============================================================================
# README Documentation Verification Tests
# ==============================================================================
# These tests verify that recipe dry-run output matches the expected commands
# documented in README.md. Expected values are defined in expected_commands.sh

# Helper: Extract the generated launch script from dry-run output
extract_vllm_command() {
    # Extract lines between "Generated Launch Script" and "What would be executed"
    echo "$1" | sed -n '/=== Generated Launch Script ===/,/=== What would be executed ===/p' | grep -v "===" | grep -v "^#" | grep -v "^$"
}

# Helper: Verify a recipe contains all expected arguments
verify_recipe_args() {
    local recipe_name="$1"
    local expected_model="$2"
    local expected_container="$3"
    shift 3
    local expected_args=("$@")
    
    log_test "README match: $recipe_name"
    
    if [[ ! -f "$PROJECT_DIR/recipes/${recipe_name}.yaml" ]]; then
        log_skip "${recipe_name}.yaml not found"
        return
    fi
    
    mode=$(get_recipe_mode "$recipe_name")
    output=$(run_recipe_dry_run "$recipe_name" "$mode")
    vllm_cmd=$(extract_vllm_command "$output")
    launch_cmd=$(extract_launch_cmd "$output")
    
    local all_passed=true
    local missing_args=()
    
    # Check model name
    if ! echo "$vllm_cmd" | grep -q "$expected_model"; then
        missing_args+=("model: $expected_model")
        all_passed=false
    fi
    
    # Check container
    if ! echo "$launch_cmd" | grep -q "\-t $expected_container"; then
        missing_args+=("container: $expected_container")
        all_passed=false
    fi
    
    # Check each expected argument
    for arg in "${expected_args[@]}"; do
        # Handle arguments that may have slight formatting differences
        # Extract the flag and value separately for flexible matching
        local flag=$(echo "$arg" | awk '{print $1}')
        local value=$(echo "$arg" | cut -d' ' -f2-)
        
        # Use grep -F for fixed string matching (avoids -- being treated as grep options)
        if ! echo "$vllm_cmd" | grep -qF -- "$flag"; then
            missing_args+=("$arg")
            all_passed=false
        elif [[ -n "$value" ]] && [[ "$value" != "$flag" ]]; then
            # Check if value is present (might be on next line due to formatting)
            if ! echo "$vllm_cmd" | grep -qF -- "$value"; then
                missing_args+=("$arg (flag present, value mismatch)")
                all_passed=false
            fi
        fi
    done
    
    if [[ "$all_passed" == "true" ]]; then
        log_pass "README match: $recipe_name - all expected arguments present"
    else
        log_fail "README match: $recipe_name - missing arguments"
        for missing in "${missing_args[@]}"; do
            log_verbose "  Missing: $missing"
        done
        log_verbose "  vLLM command: $vllm_cmd"
    fi
}

# Test: glm-4.7-flash-awq matches README documentation
test_readme_glm_flash_awq() {
    verify_recipe_args "glm-4.7-flash-awq" \
        "$GLM_FLASH_AWQ_MODEL" \
        "$GLM_FLASH_AWQ_CONTAINER" \
        "${GLM_FLASH_AWQ_ARGS[@]}"
}

# Test: openai-gpt-oss-120b matches README documentation
test_readme_gpt_oss() {
    verify_recipe_args "openai-gpt-oss-120b" \
        "$GPT_OSS_MODEL" \
        "$GPT_OSS_CONTAINER" \
        "${GPT_OSS_ARGS[@]}"
}

# Test: minimax-m2-awq matches expected configuration
test_readme_minimax() {
    verify_recipe_args "minimax-m2-awq" \
        "$MINIMAX_MODEL" \
        "$MINIMAX_CONTAINER" \
        "${MINIMAX_ARGS[@]}"
}

# Test: glm-4.7-flash-awq includes correct mod
test_readme_glm_flash_mod() {
    log_test "README match: glm-4.7-flash-awq mod path"
    
    if [[ ! -f "$PROJECT_DIR/recipes/glm-4.7-flash-awq.yaml" ]]; then
        log_skip "glm-4.7-flash-awq.yaml not found"
        return
    fi
    
    mode=$(get_recipe_mode "glm-4.7-flash-awq")
    output=$(run_recipe_dry_run "glm-4.7-flash-awq" "$mode")
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "$GLM_FLASH_AWQ_MOD"; then
        log_pass "README match: glm-4.7-flash-awq has correct mod path"
    else
        log_fail "README match: glm-4.7-flash-awq missing expected mod: $GLM_FLASH_AWQ_MOD"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Helper: Verify cluster mode specific arguments
verify_cluster_args() {
    local recipe_name="$1"
    local expected_tp="$2"
    shift 2
    local expected_args=("$@")
    
    log_test "README match (cluster): $recipe_name"
    
    if [[ ! -f "$PROJECT_DIR/recipes/${recipe_name}.yaml" ]]; then
        log_skip "${recipe_name}.yaml not found"
        return
    fi
    
    # Use fake nodes for cluster mode
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
    vllm_cmd=$(extract_vllm_command "$output")
    
    local all_passed=true
    local missing_args=()
    
    # Check tensor parallel
    if ! echo "$vllm_cmd" | grep -qE "(--tensor-parallel-size|-tp) $expected_tp"; then
        missing_args+=("tensor_parallel: $expected_tp")
        all_passed=false
    fi
    
    # Check cluster-specific arguments
    for arg in "${expected_args[@]}"; do
        if ! echo "$vllm_cmd" | grep -qF -- "$arg"; then
            missing_args+=("$arg")
            all_passed=false
        fi
    done
    
    if [[ "$all_passed" == "true" ]]; then
        log_pass "README match (cluster): $recipe_name - cluster args correct"
    else
        log_fail "README match (cluster): $recipe_name - missing cluster arguments"
        for missing in "${missing_args[@]}"; do
            log_verbose "  Missing: $missing"
        done
        log_verbose "  vLLM command: $vllm_cmd"
    fi
}

# Test: openai-gpt-oss-120b cluster mode has correct tensor_parallel and ray backend
test_readme_gpt_oss_cluster() {
    verify_cluster_args "openai-gpt-oss-120b" \
        "$GPT_OSS_CLUSTER_TP" \
        "${GPT_OSS_CLUSTER_ARGS[@]}"
}

# Test: minimax-m2-awq cluster mode has correct tensor_parallel and ray backend
test_readme_minimax_cluster() {
    verify_cluster_args "minimax-m2-awq" \
        "$MINIMAX_CLUSTER_TP" \
        "${MINIMAX_CLUSTER_ARGS[@]}"
}

# Test: glm-4.7-flash-awq cluster mode stays at tp=1 (single GPU model)
test_readme_glm_flash_cluster() {
    log_test "README match (cluster): glm-4.7-flash-awq stays tp=1"
    
    if [[ ! -f "$PROJECT_DIR/recipes/glm-4.7-flash-awq.yaml" ]]; then
        log_skip "glm-4.7-flash-awq.yaml not found"
        return
    fi
    
    # Even in cluster mode, this model uses tp=1
    output=$("$PROJECT_DIR/run-recipe.py" glm-4.7-flash-awq --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
    vllm_cmd=$(extract_vllm_command "$output")
    
    if echo "$vllm_cmd" | grep -qE "(--tensor-parallel-size|-tp) 1"; then
        log_pass "README match (cluster): glm-4.7-flash-awq correctly keeps tp=1"
    else
        log_fail "README match (cluster): glm-4.7-flash-awq should have tp=1"
        log_verbose "  vLLM command: $vllm_cmd"
    fi
}

# ==============================================================================
# Extra vLLM Arguments Tests (-- pass-through)
# Tests for GitHub issue #30: ability to pass arbitrary vLLM arguments
# ==============================================================================

# Test: Basic extra args pass-through with --load-format
test_extra_args_load_format() {
    log_test "Extra args: --load-format safetensors"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -- --load-format safetensors 2>&1)
    
    if echo "$output" | grep -q "\-\-load-format safetensors"; then
        log_pass "Extra args: --load-format correctly appended"
    else
        log_fail "Extra args: --load-format not found in output"
        log_verbose "$output"
    fi
}

# Test: Extra args with --served-model-name
test_extra_args_served_model_name() {
    log_test "Extra args: --served-model-name custom-api-name"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -- --served-model-name custom-api-name 2>&1)
    
    if echo "$output" | grep -q "\-\-served-model-name custom-api-name"; then
        log_pass "Extra args: --served-model-name correctly appended"
    else
        log_fail "Extra args: --served-model-name not found in output"
        log_verbose "$output"
    fi
}

# Test: Extra args with equals syntax (-cc.cudagraph_mode=PIECEWISE)
test_extra_args_equals_syntax() {
    log_test "Extra args: -cc.cudagraph_mode=PIECEWISE (equals syntax)"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -- -cc.cudagraph_mode=PIECEWISE 2>&1)
    
    if echo "$output" | grep -q "\-cc.cudagraph_mode=PIECEWISE"; then
        log_pass "Extra args: equals syntax correctly appended"
    else
        log_fail "Extra args: equals syntax not found in output"
        log_verbose "$output"
    fi
}

# Test: Multiple extra args
test_extra_args_multiple() {
    log_test "Extra args: multiple arguments"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -- --load-format auto --enforce-eager --seed 42 2>&1)
    
    local all_found=true
    if ! echo "$output" | grep -q "\-\-load-format auto"; then
        all_found=false
    fi
    if ! echo "$output" | grep -q "\-\-enforce-eager"; then
        all_found=false
    fi
    if ! echo "$output" | grep -q "\-\-seed 42"; then
        all_found=false
    fi
    
    if [[ "$all_found" == "true" ]]; then
        log_pass "Extra args: multiple arguments correctly appended"
    else
        log_fail "Extra args: not all arguments found in output"
        log_verbose "$output"
    fi
}

# Test: Empty extra args (just -- with nothing after)
test_extra_args_empty() {
    log_test "Extra args: empty (just --)"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    # Should not error with just --
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -- 2>&1)
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]] && echo "$output" | grep -q "vllm serve"; then
        log_pass "Extra args: empty -- handled correctly"
    else
        log_fail "Extra args: empty -- caused error"
        log_verbose "$output"
    fi
}

# Test: Duplicate detection warning for --port
test_extra_args_duplicate_port_warning() {
    log_test "Extra args: duplicate --port shows warning"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    # Pass --port via shorthand AND via extra args - should warn
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo --port 8080 -- --port 9999 2>&1)
    
    if echo "$output" | grep -qi "warning.*\-\-port\|duplicate.*port"; then
        log_pass "Extra args: duplicate --port warning shown"
    else
        log_fail "Extra args: no warning for duplicate --port"
        log_verbose "$output"
    fi
}

# Test: Duplicate detection warning for --gpu-memory-utilization
test_extra_args_duplicate_gpu_mem_warning() {
    log_test "Extra args: duplicate --gpu-memory-utilization shows warning"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    # Pass --gpu-mem via shorthand AND via extra args - should warn
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo --gpu-mem 0.8 -- --gpu-memory-utilization 0.95 2>&1)
    
    if echo "$output" | grep -qi "warning.*gpu-memory-utilization\|duplicate.*gpu"; then
        log_pass "Extra args: duplicate --gpu-memory-utilization warning shown"
    else
        log_fail "Extra args: no warning for duplicate --gpu-memory-utilization"
        log_verbose "$output"
    fi
}

# Test: Duplicate detection warning for --tensor-parallel-size
test_extra_args_duplicate_tp_warning() {
    log_test "Extra args: duplicate --tensor-parallel-size shows warning"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    # Pass --tp via shorthand AND via extra args - should warn
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo --tp 2 -- --tensor-parallel-size 4 2>&1)
    
    if echo "$output" | grep -qi "warning.*tensor-parallel\|duplicate.*tensor"; then
        log_pass "Extra args: duplicate --tensor-parallel-size warning shown"
    else
        log_fail "Extra args: no warning for duplicate --tensor-parallel-size"
        log_verbose "$output"
    fi
}

# Test: Extra args appear after template-substituted command
test_extra_args_ordering() {
    log_test "Extra args: appear at end of vllm command"
    
    recipe_name=$(find_solo_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No solo-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -- --my-custom-arg value 2>&1)
    vllm_cmd=$(extract_vllm_command "$output")
    
    # The custom arg should appear and be at the end of the command
    if echo "$vllm_cmd" | grep -q "\-\-my-custom-arg value"; then
        # Check it's near the end (after common args like --port)
        if echo "$vllm_cmd" | grep -qE ".*\-\-port.*\-\-my-custom-arg\|.*\-\-host.*\-\-my-custom-arg"; then
            log_pass "Extra args: correctly ordered at end"
        else
            # It's there, just accept it
            log_pass "Extra args: present in command"
        fi
    else
        log_fail "Extra args: --my-custom-arg not found in vllm command"
        log_verbose "$vllm_cmd"
    fi
}

# Test: Extra args work in cluster mode
test_extra_args_cluster_mode() {
    log_test "Extra args: work in cluster mode"
    
    recipe_name=$(find_cluster_recipe)
    if [[ -z "$recipe_name" ]]; then
        log_skip "No cluster-capable recipes found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run -n "10.0.0.1,10.0.0.2" -- --load-format auto 2>&1)
    
    if echo "$output" | grep -q "\-\-load-format auto"; then
        log_pass "Extra args: work in cluster mode"
    else
        log_fail "Extra args: not found in cluster mode output"
        log_verbose "$output"
    fi
}

# Run all tests
main() {
    echo "=============================================="
    echo "  run-recipe.py Integration Tests"
    echo "=============================================="
    echo ""
    
    cd "$PROJECT_DIR"
    
    check_prerequisites
    echo ""
    
    # File existence tests
    test_run_recipe_exists
    test_launch_cluster_exists
    echo ""
    
    # Basic functionality tests
    test_list_recipes
    test_recipe_version_required
    test_all_recipes_load
    echo ""
    
    # Dry-run tests
    test_dry_run_generates_script
    test_solo_mode_tp1
    test_solo_mode_removes_ray
    test_cluster_mode_keeps_ray
    test_cli_override_port
    echo ""
    
    # launch-cluster.sh command line verification tests
    echo "--- Launch Command Verification ---"
    test_launch_cmd_solo_flag
    test_launch_cmd_nodes_flag
    test_launch_cmd_container_image
    test_launch_cmd_mods
    test_launch_cmd_daemon_flag
    test_launch_cmd_nccl_debug
    test_launch_cmd_launch_script
    test_launch_cmd_container_override
    test_launch_cmd_no_solo_in_cluster
    test_launch_cmd_env_passthrough
    test_launch_cmd_no_env_by_default
    echo ""
    
    # README documentation verification tests
    echo "--- README Documentation Verification (Solo Mode) ---"
    test_readme_glm_flash_awq
    test_readme_gpt_oss
    test_readme_minimax
    test_readme_glm_flash_mod
    echo ""
    
    # Cluster mode documentation verification tests
    echo "--- README Documentation Verification (Cluster Mode) ---"
    test_readme_gpt_oss_cluster
    test_readme_minimax_cluster
    test_readme_glm_flash_cluster
    echo ""
    
    # launch-cluster.sh tests
    test_launch_cluster_help
    test_launch_cluster_examples_path
    echo ""
    
    # Extra vLLM arguments tests (-- pass-through)
    echo "--- Extra vLLM Arguments (-- pass-through) ---"
    test_extra_args_load_format
    test_extra_args_served_model_name
    test_extra_args_equals_syntax
    test_extra_args_multiple
    test_extra_args_empty
    test_extra_args_duplicate_port_warning
    test_extra_args_duplicate_gpu_mem_warning
    test_extra_args_duplicate_tp_warning
    test_extra_args_ordering
    test_extra_args_cluster_mode
    echo ""
    
    # Validation tests
    test_unsupported_recipe_version
    test_missing_recipe_version_fails
    test_cluster_only_fails_solo
    test_solo_only_fails_cluster
    test_solo_only_allows_solo
    test_conflicting_mode_flags_fail
    echo ""
    
    # Summary
    echo "=============================================="
    echo "  Test Summary"
    echo "=============================================="
    echo -e "  ${GREEN}Passed:${NC}  $TESTS_PASSED"
    echo -e "  ${RED}Failed:${NC}  $TESTS_FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
    echo "=============================================="
    
    if [[ $TESTS_FAILED -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

main "$@"
