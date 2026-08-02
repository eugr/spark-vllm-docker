#!/bin/bash
#
# Focused behavior tests for build-and-copy.sh image preparation.
# Uses fake docker/ssh/curl commands, so it never pulls images, builds images,
# copies to real hosts, or touches the repository wheel cache.

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
    if [ -n "${OUTPUT_LOG:-}" ] && [ -f "$OUTPUT_LOG" ]; then
        echo "--- output ---" >&2
        sed -n '1,220p' "$OUTPUT_LOG" >&2
    fi
    if [ -n "${TEST_LOG:-}" ] && [ -f "$TEST_LOG" ]; then
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
    cp "$PROJECT_DIR/build-and-copy.sh" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/autodiscover.sh" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/Dockerfile" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/Dockerfile.mxfp4" "$FIXTURE_DIR/"
    mkdir -p "$FIXTURE_DIR/wheels"
    touch "$FIXTURE_DIR/wheels/flashinfer-test.whl"
    touch "$FIXTURE_DIR/wheels/vllm-test.whl"
    touch "$FIXTURE_DIR/test.env"
    : > "$TEST_LOG"
    : > "$OUTPUT_LOG"

    cat > "$FAKE_BIN_DIR/docker" <<'DOCKER'
#!/bin/bash
set -euo pipefail
echo "docker $*" >> "$TEST_LOG"
if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then
    echo "${LOCAL_IMAGE_ID:-sha256:local}"
    exit 0
fi
if [ "${1:-}" = "save" ]; then
    out=""
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "-o" ]; then
            out="$2"
            shift 2
            continue
        fi
        shift
    done
    if [ -n "$out" ]; then
        printf 'fake image\n' > "$out"
    fi
fi
DOCKER

    cat > "$FAKE_BIN_DIR/ssh" <<'SSH'
#!/bin/bash
set -euo pipefail
echo "ssh $*" >> "$TEST_LOG"
target="${1:-}"
host="${target#*@}"
cmd="${*:2}"
if [[ "$cmd" == *"docker image inspect"* ]]; then
    case "$host" in
        samehost)
            echo "${LOCAL_IMAGE_ID:-sha256:local}"
            exit 0
            ;;
        diffhost)
            echo "sha256:remote"
            exit 0
            ;;
        *)
            exit 1
            ;;
    esac
fi
while IFS= read -r _line; do
    :
done
SSH

    cat > "$FAKE_BIN_DIR/curl" <<'CURL'
#!/bin/bash
set -euo pipefail
echo "curl $*" >> "$TEST_LOG"
exit 22
CURL

    chmod +x "$FAKE_BIN_DIR/docker" "$FAKE_BIN_DIR/ssh" "$FAKE_BIN_DIR/curl"
}

run_build() {
    (
        cd "$FIXTURE_DIR"
        PATH="$FAKE_BIN_DIR:$PATH" TEST_LOG="$TEST_LOG" ./build-and-copy.sh --config "$FIXTURE_DIR/test.env" "$@"
    ) > "$OUTPUT_LOG" 2>&1
}

assert_log_contains() {
    local pattern="$1"
    if ! grep -Eq "$pattern" "$TEST_LOG"; then
        fail "Expected command log to match: $pattern"
    fi
}

assert_log_not_contains() {
    local pattern="$1"
    if grep -Eq "$pattern" "$TEST_LOG"; then
        fail "Expected command log not to match: $pattern"
    fi
}

assert_output_contains() {
    local pattern="$1"
    if ! grep -Eq "$pattern" "$OUTPUT_LOG"; then
        fail "Expected output to match: $pattern"
    fi
}

test_default_uses_prebuilt() {
    setup_fixture
    run_build || fail "default run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker tag eugr/spark-vllm:latest vllm-node$'
    assert_log_not_contains '^docker build'
    pass "default pulls and tags prebuilt image"
}

test_tf5_uses_prebuilt_tf5_tag() {
    setup_fixture
    run_build --tf5 || fail "--tf5 run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker tag eugr/spark-vllm:latest vllm-node-tf5$'
    assert_log_not_contains '^docker build'
    pass "--tf5 pulls prebuilt image under vllm-node-tf5"
}

test_custom_tag_uses_prebuilt_custom_tag() {
    setup_fixture
    run_build -t custom-vllm || fail "custom tag run failed"
    assert_log_contains '^docker tag eugr/spark-vllm:latest custom-vllm$'
    assert_log_not_contains '^docker build'
    pass "custom tag pulls prebuilt image under requested tag"
}

test_default_gpu_arch_stays_prebuilt() {
    setup_fixture
    run_build --gpu-arch 12.1a || fail "default gpu arch run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker build'
    pass "explicit default gpu arch still uses prebuilt image"
}

test_non_default_gpu_arch_uses_wheel_build() {
    setup_fixture
    run_build --gpu-arch 12.0f || fail "non-default gpu arch run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker build -t vllm-node '
    assert_log_contains 'NCCL_NVCC_GENCODE=-gencode=arch=compute_120,code=sm_120'
    pass "non-default gpu arch uses wheel build path"
}

test_use_wheels_uses_wheel_build() {
    setup_fixture
    run_build --use-wheels || fail "--use-wheels run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker build --target flashinfer-export '
    assert_log_not_contains '^docker build --target vllm-export '
    assert_log_contains '^docker build -t vllm-node '
    assert_log_contains 'NCCL_NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121'
    pass "--use-wheels builds only the runner from precompiled wheels"
}

test_use_wheels_never_falls_back_to_source() {
    setup_fixture
    rm -f "$FIXTURE_DIR/wheels/flashinfer-test.whl" "$FIXTURE_DIR/wheels/vllm-test.whl"
    if run_build --use-wheels; then
        fail "--use-wheels unexpectedly succeeded without precompiled wheels"
    fi
    assert_log_not_contains '^docker build --target flashinfer-export '
    assert_log_not_contains '^docker build --target vllm-export '
    assert_log_not_contains '^docker build -t vllm-node '
    assert_output_contains 'Error: No precompiled FlashInfer wheels are available and the download failed\.'
    assert_output_contains 'Re-run with --rebuild-flashinfer to explicitly build FlashInfer from source\.'
    pass "--use-wheels fails instead of implicitly compiling missing wheels"
}

test_use_wheels_never_builds_missing_vllm_implicitly() {
    setup_fixture
    rm -f "$FIXTURE_DIR/wheels/vllm-test.whl"
    if run_build --use-wheels; then
        fail "--use-wheels unexpectedly succeeded without a precompiled vLLM wheel"
    fi
    assert_log_not_contains '^docker build --target flashinfer-export '
    assert_log_not_contains '^docker build --target vllm-export '
    assert_log_not_contains '^docker build -t vllm-node '
    assert_output_contains 'Error: No precompiled vLLM wheels are available and the download failed\.'
    assert_output_contains 'Re-run with --rebuild-vllm to explicitly build vLLM from source\.'
    pass "--use-wheels never implicitly compiles a missing vLLM wheel"
}

test_use_wheels_builds_only_explicit_source_target() {
    setup_fixture
    run_build --use-wheels --rebuild-vllm || fail "--use-wheels --rebuild-vllm run failed"
    assert_log_not_contains '^docker build --target flashinfer-export '
    assert_log_contains '^docker build --target vllm-export '
    assert_log_contains '^docker build -t vllm-node '
    pass "--use-wheels compiles vLLM only when explicitly requested"
}

test_use_wheels_builds_only_explicit_flashinfer_target() {
    setup_fixture
    run_build --use-wheels --rebuild-flashinfer || fail "--use-wheels --rebuild-flashinfer run failed"
    assert_log_contains '^docker build --target flashinfer-export '
    assert_log_not_contains '^docker build --target vllm-export '
    assert_log_contains '^docker build -t vllm-node '
    pass "--use-wheels compiles FlashInfer only when explicitly requested"
}

test_cleanup_stays_prebuilt() {
    setup_fixture
    run_build --cleanup || fail "--cleanup run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker build'
    pass "--cleanup is orthogonal and still allows prebuilt path"
}

test_prebuilt_copy_parallel() {
    setup_fixture
    run_build -c host1,host2 --copy-parallel || fail "prebuilt copy run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker tag eugr/spark-vllm:latest vllm-node$'
    assert_log_contains '^docker save -o .* vllm-node$'
    assert_log_contains '^ssh .*@host1 docker load$'
    assert_log_contains '^ssh .*@host2 docker load$'
    pass "prebuilt path saves requested tag and supports parallel copy"
}

test_copy_skips_matching_remote_image() {
    setup_fixture
    run_build -c samehost || fail "matching remote copy run failed"
    assert_log_contains '^docker image inspect --format \{\{\.Id\}\} vllm-node$'
    assert_log_contains '^ssh .*@samehost docker image inspect --format '\''\{\{\.Id\}\}'\'' vllm-node$'
    assert_log_not_contains '^docker save '
    assert_log_not_contains '^ssh .*@samehost docker load$'
    assert_output_contains "Image 'vllm-node' is already up to date on .*@samehost; skipping\."
    assert_output_contains 'All remote images are up to date; skipping save/copy\.'
    pass "copy skips save/load when remote image ID matches local"
}

test_copy_only_updates_missing_or_different_hosts() {
    setup_fixture
    run_build -c samehost,host1 --copy-parallel || fail "mixed remote copy run failed"
    assert_log_contains '^docker save -o .* vllm-node$'
    assert_log_not_contains '^ssh .*@samehost docker load$'
    assert_log_contains '^ssh .*@host1 docker load$'
    pass "copy loads only hosts whose image ID is missing or different"
}

test_no_build_skips_prebuilt() {
    setup_fixture
    run_build --no-build -c host1 || fail "--no-build copy run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker tag eugr/spark-vllm:latest'
    assert_log_contains '^docker save -o .* vllm-node$'
    assert_log_contains '^ssh .*@host1 docker load$'
    pass "--no-build skips prebuilt pull and copies existing local tag"
}

test_build_only_flags_warn_on_prebuilt() {
    setup_fixture
    run_build --network host --full-log -j 4 || fail "build-only flags prebuilt run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker build'
    assert_output_contains 'Warning: --network is only used for Docker builds; ignoring it while pulling eugr/spark-vllm:latest\.'
    assert_output_contains 'Warning: --full-log is only used for Docker builds; ignoring it while pulling eugr/spark-vllm:latest\.'
    assert_output_contains 'Warning: --build-jobs is only used for Docker builds; ignoring it while pulling eugr/spark-vllm:latest\.'
    pass "build-only flags warn but do not force wheel path"
}

test_flashinfer_ref_forwards_selected_ref() {
    setup_fixture
    run_build --flashinfer-ref 0123456789abcdef || fail "--flashinfer-ref run failed"
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_REF=0123456789abcdef'
    assert_output_contains 'Rebuilding FlashInfer wheels \(--flashinfer-ref specified\)\.\.\.'
    pass "--flashinfer-ref forwards selected ref"
}

test_requested_flashinfer_prs_apply_to_selected_ref() {
    setup_fixture
    run_build --flashinfer-ref 0123456789abcdef --apply-flashinfer-pr 12345 || fail "--apply-flashinfer-pr with --flashinfer-ref run failed"
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_REF=0123456789abcdef .*--build-arg FLASHINFER_PRS=12345'
    assert_output_contains 'Rebuilding FlashInfer wheels \(--flashinfer-ref and --apply-flashinfer-pr specified\)\.\.\.'
    assert_output_contains 'Applying FlashInfer PRs: 12345'
    pass "--apply-flashinfer-pr applies requested PRs to selected ref"
}

test_vllm_ref_skips_preset_prs_by_default() {
    setup_fixture
    run_build --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 || fail "--vllm-ref run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=ab666069935c1f23e8ef56038b4659ac9e8f19f8 .*--build-arg VLLM_APPLY_PRESET_PRS=0'
    assert_log_not_contains 'VLLM_APPLY_PRESET_PRS=1'
    assert_output_contains 'Skipping preset vLLM PRs because --vllm-ref or --apply-vllm-pr was specified\.'
    pass "--vllm-ref forwards preset PR opt-out by default"
}

test_rebuild_vllm_applies_preset_prs_by_default() {
    setup_fixture
    run_build --rebuild-vllm || fail "--rebuild-vllm run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=main .*--build-arg VLLM_APPLY_PRESET_PRS=1'
    assert_output_contains 'Applying preset vLLM PRs from the Dockerfile by default\.'
    pass "ordinary main source rebuild applies preset PRs by default"
}

test_apply_vllm_pr_skips_preset_prs_by_default() {
    setup_fixture
    run_build --apply-vllm-pr 12345 || fail "--apply-vllm-pr run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=main .*--build-arg VLLM_APPLY_PRESET_PRS=0 .*--build-arg VLLM_PRS=12345'
    assert_output_contains 'Skipping preset vLLM PRs because --vllm-ref or --apply-vllm-pr was specified\.'
    pass "--apply-vllm-pr suppresses preset PRs by default"
}

test_apply_vllm_pr_can_apply_preset_prs_explicitly() {
    setup_fixture
    run_build --apply-vllm-pr 12345 --apply-preset-vllm-prs || fail "custom and preset PR run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=main .*--build-arg VLLM_APPLY_PRESET_PRS=1 .*--build-arg VLLM_PRS=12345'
    assert_output_contains 'Applying preset vLLM PRs from the Dockerfile \(explicitly requested\)\.'
    pass "--apply-preset-vllm-prs overrides custom PR preset suppression"
}

test_vllm_ref_can_apply_preset_prs_explicitly() {
    setup_fixture
    run_build --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 --apply-preset-vllm-prs || fail "--vllm-ref preset run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=ab666069935c1f23e8ef56038b4659ac9e8f19f8 .*--build-arg VLLM_APPLY_PRESET_PRS=1'
    assert_output_contains 'Applying preset vLLM PRs from the Dockerfile \(explicitly requested\)\.'
    pass "--apply-preset-vllm-prs applies presets to selected ref"
}

test_apply_preset_prs_forces_vllm_rebuild() {
    setup_fixture
    run_build --apply-preset-vllm-prs || fail "--apply-preset-vllm-prs run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=main .*--build-arg VLLM_APPLY_PRESET_PRS=1'
    assert_output_contains 'Rebuilding vLLM wheels \(\--apply-preset-vllm-prs specified\)\.\.\.'
    pass "--apply-preset-vllm-prs forces a vLLM rebuild"
}

test_requested_vllm_prs_apply_to_selected_vllm_ref() {
    setup_fixture
    run_build --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 --apply-vllm-pr 12345 || fail "--apply-vllm-pr with --vllm-ref run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=ab666069935c1f23e8ef56038b4659ac9e8f19f8 .*--build-arg VLLM_APPLY_PRESET_PRS=0 .*--build-arg VLLM_PRS=12345'
    assert_output_contains 'Rebuilding vLLM wheels \(applying vLLM PRs to --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8\)\.\.\.'
    assert_output_contains 'Applying vLLM PRs: 12345'
    pass "--apply-vllm-pr applies requested PRs to selected ref"
}

test_copied_vllm_git_index_is_refreshed_before_patch_apply() {
    local source_repo="$TMP_BASE/git-index-source"
    local copied_repo="$TMP_BASE/git-index-copy"
    local patch_file="$TMP_BASE/git-index.patch"
    local apply_error="$TMP_BASE/git-index-apply-error.log"
    local base_commit patched_commit

    mkdir -p "$source_repo"
    git -C "$source_repo" init -q
    git -C "$source_repo" config user.email "test@example.com"
    git -C "$source_repo" config user.name "Test Builder"

    printf 'base\n' > "$source_repo/tracked.txt"
    git -C "$source_repo" add tracked.txt
    git -C "$source_repo" commit -qm "base"
    base_commit="$(git -C "$source_repo" rev-parse HEAD)"

    printf 'patched\n' > "$source_repo/tracked.txt"
    git -C "$source_repo" commit -qam "patch"
    patched_commit="$(git -C "$source_repo" rev-parse HEAD)"
    git -C "$source_repo" diff --binary "$base_commit" "$patched_commit" > "$patch_file"
    git -C "$source_repo" checkout -q --detach "$base_commit"

    cp -a "$source_repo" "$copied_repo"
    if git -C "$copied_repo" apply --3way --index --binary "$patch_file" 2> "$apply_error"; then
        fail "copied repository unexpectedly accepted --index apply without refreshing"
    fi
    if ! grep -q 'does not match index' "$apply_error"; then
        fail "copied repository did not reproduce the stale-index failure"
    fi

    git -C "$copied_repo" update-index --refresh
    git -C "$copied_repo" apply --3way --index --binary "$patch_file"
    if [ "$(git -C "$copied_repo" show :tracked.txt)" != "patched" ]; then
        fail "patch did not apply after refreshing the copied repository index"
    fi

    if ! grep -q 'git reset --hard HEAD' "$PROJECT_DIR/Dockerfile"; then
        fail "Dockerfile does not clean the cached vLLM checkout"
    fi
    if ! grep -q 'git update-index --refresh' "$PROJECT_DIR/Dockerfile"; then
        fail "Dockerfile does not refresh the copied vLLM index"
    fi
    pass "copied vLLM Git index is refreshed before patch apply"
}

test_dockerfile_applies_flashinfer_prs_without_merging_branch_history() {
    local flashinfer_pr_block="$TMP_BASE/flashinfer-pr-block"

    sed -n '/ARG FLASHINFER_PRS=""/,/# TEMPORARY patch/p' "$PROJECT_DIR/Dockerfile" > "$flashinfer_pr_block"
    for expected in \
        'git update-index --refresh' \
        'git merge-base origin/main pr-${pr}' \
        'git diff --binary "$pr_base" "pr-${pr}"' \
        'git apply --3way --index --binary "$patch_file"' \
        'git merge-base --is-ancestor "$FLASHINFER_REQUESTED_HEAD" HEAD'; do
        if ! grep -Fq "$expected" "$flashinfer_pr_block"; then
            fail "FlashInfer PR block is missing patch-only behavior: $expected"
        fi
    done
    if grep -Fq 'git merge pr-${pr}' "$flashinfer_pr_block"; then
        fail "FlashInfer PR block still merges complete PR branch history"
    fi
    if ! sed -n '/if \[ ! -d "flashinfer" \]/,/cp -a \/repo-cache\/flashinfer/p' "$PROJECT_DIR/Dockerfile" | grep -Fq 'git reset --hard HEAD'; then
        fail "Dockerfile does not clean the cached FlashInfer checkout"
    fi
    pass "FlashInfer PRs apply as patches without merging branch history"
}

test_dockerfiles_pin_tvm_ffi_regression_version() {
    if [ "$(grep -Fc 'apache-tvm-ffi==0.1.12' "$PROJECT_DIR/Dockerfile")" -ne 2 ] || \
       [ "$(grep -Fc 'apache-tvm-ffi==0.1.12' "$PROJECT_DIR/Dockerfile.mxfp4")" -ne 1 ]; then
        fail "Dockerfiles do not pin every TVM-FFI install to the known-good 0.1.12 release"
    fi
    pass "Dockerfiles pin TVM-FFI to the known-good 0.1.12 release"
}

test_default_uses_prebuilt
test_tf5_uses_prebuilt_tf5_tag
test_custom_tag_uses_prebuilt_custom_tag
test_default_gpu_arch_stays_prebuilt
test_non_default_gpu_arch_uses_wheel_build
test_use_wheels_uses_wheel_build
test_use_wheels_never_falls_back_to_source
test_use_wheels_never_builds_missing_vllm_implicitly
test_use_wheels_builds_only_explicit_source_target
test_use_wheels_builds_only_explicit_flashinfer_target
test_cleanup_stays_prebuilt
test_prebuilt_copy_parallel
test_copy_skips_matching_remote_image
test_copy_only_updates_missing_or_different_hosts
test_no_build_skips_prebuilt
test_build_only_flags_warn_on_prebuilt
test_flashinfer_ref_forwards_selected_ref
test_requested_flashinfer_prs_apply_to_selected_ref
test_rebuild_vllm_applies_preset_prs_by_default
test_vllm_ref_skips_preset_prs_by_default
test_apply_vllm_pr_skips_preset_prs_by_default
test_apply_vllm_pr_can_apply_preset_prs_explicitly
test_vllm_ref_can_apply_preset_prs_explicitly
test_apply_preset_prs_forces_vllm_rebuild
test_requested_vllm_prs_apply_to_selected_vllm_ref
test_copied_vllm_git_index_is_refreshed_before_patch_apply
test_dockerfile_applies_flashinfer_prs_without_merging_branch_history
test_dockerfiles_pin_tvm_ffi_regression_version

echo "Passed $TESTS_PASSED build-and-copy tests."
