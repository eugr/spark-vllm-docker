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
    cp -a "$PROJECT_DIR/docker" "$FIXTURE_DIR/"
    mkdir -p \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/regular" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/custom" \
        "$FIXTURE_DIR/.wheel-cache/vllm/regular" \
        "$FIXTURE_DIR/.wheel-cache/vllm/b12x" \
        "$FIXTURE_DIR/.wheel-cache/vllm/custom"
    touch \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_cubin-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_jit_cache-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_python-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/vllm/regular/vllm-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/vllm/b12x/vllm-test.whl"
    touch "$FIXTURE_DIR/test.env"
    : > "$TEST_LOG"
    : > "$OUTPUT_LOG"

    cat > "$FAKE_BIN_DIR/docker" <<'DOCKER'
#!/bin/bash
set -euo pipefail
echo "docker $*" >> "$TEST_LOG"
if [ "${1:-}" = "build" ]; then
    (
        target=""
        output=""
        build_arch="12.1a"
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --target) target="$2"; shift 2 ;;
                --output) output="$2"; shift 2 ;;
                --build-arg)
                    case "$2" in
                        TORCH_CUDA_ARCH_LIST=*) build_arch="${2#TORCH_CUDA_ARCH_LIST=}" ;;
                        FLASHINFER_CUDA_ARCH_LIST=*) build_arch="${2#FLASHINFER_CUDA_ARCH_LIST=}" ;;
                    esac
                    shift 2
                    ;;
                *) shift ;;
            esac
        done
        case "$output" in
            type=local,dest=*)
                dest="${output#type=local,dest=}"
                mkdir -p "$dest"
                case "$target" in
                    flashinfer-export)
                        printf 'fake wheel\n' > "$dest/flashinfer_cubin-built.whl"
                        printf 'fake wheel\n' > "$dest/flashinfer_jit_cache-built.whl"
                        printf 'fake wheel\n' > "$dest/flashinfer_python-built.whl"
                        printf 'flashinfer-commit\n' > "$dest/.flashinfer-commit"
                        printf '%s\n' "$build_arch" > "$dest/.flashinfer-arch"
                        ;;
                    vllm-export)
                        printf 'fake wheel\n' > "$dest/vllm-built.whl"
                        printf 'vllm-commit\n' > "$dest/.vllm-commit"
                        printf 'deepgemm-commit\n' > "$dest/.deepgemm-commit"
                        printf '%s\n' "$build_arch" > "$dest/.vllm-arch"
                        ;;
                esac
                ;;
        esac
    )
fi
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
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_CUDA_ARCH_LIST=12.0f '
    assert_log_contains '^docker build -t vllm-node '
    assert_log_contains 'NCCL_NVCC_GENCODE=-gencode=arch=compute_120,code=sm_120'
    assert_output_contains 'Rebuilding FlashInfer wheels for GPU architecture 12\.0f\.\.\.'
    pass "non-default gpu arch rebuilds FlashInfer for regular builds"
}

test_default_prebuilt_ignores_local_flashinfer_arch() {
    setup_fixture
    printf '12.0f\n' > "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/.flashinfer-arch"
    run_build || fail "default run with alternate local FlashInfer cache failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker build --target flashinfer-export '
    pass "default prebuilt image ignores the unused local FlashInfer cache"
}

test_regular_build_rebuilds_mismatched_cached_flashinfer_arch() {
    setup_fixture
    printf '12.0f\n' > "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/.flashinfer-arch"
    run_build --rebuild-vllm || fail "regular cached-arch build failed"
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_CUDA_ARCH_LIST=12.1a '
    assert_output_contains 'Rebuilding FlashInfer wheels for GPU architecture 12\.1a\.\.\.'
    pass "regular builds do not reuse FlashInfer wheels for another architecture"
}

test_regular_build_reuses_matching_cached_flashinfer_arch() {
    setup_fixture
    touch \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/custom/flashinfer_cubin-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/custom/flashinfer_jit_cache-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/custom/flashinfer_python-test.whl"
    printf '12.0f\n' > "$FIXTURE_DIR/.wheel-cache/flashinfer/custom/.flashinfer-arch"
    run_build --gpu-arch 12.0f || fail "regular matching cached-arch build failed"
    assert_log_not_contains '^docker build --target flashinfer-export '
    assert_log_contains '^docker build -t vllm-node '
    pass "regular builds can reuse FlashInfer wheels for the selected architecture"
}

test_use_wheels_rejects_mismatched_flashinfer_arch() {
    setup_fixture
    printf '12.0f\n' > "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/.flashinfer-arch"
    if run_build --use-wheels; then
        fail "--use-wheels unexpectedly accepted mismatched FlashInfer wheels"
    fi
    assert_log_not_contains '^docker build --target flashinfer-export '
    assert_output_contains 'Error: Cached FlashInfer wheels do not match GPU architecture 12\.1a\.'
    assert_output_contains 'Re-run with --rebuild-flashinfer or provide matching wheels with a \.flashinfer-arch marker\.'
    pass "--use-wheels rejects a mismatched FlashInfer wheel cache"
}

test_use_wheels_rejects_mismatched_vllm_arch() {
    setup_fixture
    printf '12.0f\n' > "$FIXTURE_DIR/.wheel-cache/vllm/regular/.vllm-arch"
    if run_build --use-wheels; then
        fail "--use-wheels unexpectedly accepted a mismatched vLLM wheel"
    fi
    assert_log_not_contains '^docker build --target vllm-export '
    assert_output_contains 'Error: Cached regular vLLM wheel does not match GPU architecture 12\.1a\.'
    pass "--use-wheels rejects a mismatched vLLM wheel cache"
}

test_use_wheels_non_default_empty_cache_skips_downloads() {
    setup_fixture
    rm -f "$FIXTURE_DIR/.wheel-cache/flashinfer/custom"/*.whl \
        "$FIXTURE_DIR/.wheel-cache/vllm/custom"/*.whl
    if run_build --use-wheels --gpu-arch 12.0f --force-download; then
        fail "--use-wheels unexpectedly accepted an empty non-default wheel cache"
    fi
    assert_log_not_contains '^curl '
    assert_log_not_contains '^docker build'
    assert_output_contains 'Error: Cached FlashInfer wheels do not match GPU architecture 12\.0f\.'
    pass "--use-wheels skips downloads for an empty non-default wheel cache"
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
    rm -f "$FIXTURE_DIR/.wheel-cache/flashinfer/regular"/*.whl \
        "$FIXTURE_DIR/.wheel-cache/vllm/regular"/*.whl
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
    rm -f "$FIXTURE_DIR/.wheel-cache/vllm/regular"/*.whl
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
    assert_log_contains '^docker build -t vllm-node .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/custom --build-context vllm_wheels=\./\.wheel-cache/vllm/regular '
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
    assert_output_contains 'Skipping preset vLLM PRs because --vllm-repo, --vllm-ref, or --apply-vllm-pr was specified\.'
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
    assert_output_contains 'Skipping preset vLLM PRs because --vllm-repo, --vllm-ref, or --apply-vllm-pr was specified\.'
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

test_custom_vllm_repo_forces_source_build() {
    setup_fixture
    run_build --vllm-repo https://github.com/example/vllm.git || fail "--vllm-repo run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=main --build-arg VLLM_REPO=https://github.com/example/vllm.git --build-arg VLLM_APPLY_PRESET_PRS=0'
    assert_log_contains '^docker build -t vllm-node .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/regular --build-context vllm_wheels=\./\.wheel-cache/vllm/custom '
    assert_log_not_contains 'B12X_REPO='
    assert_log_not_contains 'VLLM_PATCH_B12X_C128A_ALIGNMENT=1'
    assert_output_contains 'Rebuilding vLLM wheels \(--vllm-repo specified\)\.\.\.'
    assert_output_contains 'Skipping preset vLLM PRs because --vllm-repo, --vllm-ref, or --apply-vllm-pr was specified\.'
    pass "--vllm-repo forces a source build and suppresses upstream preset PRs"
}

test_exp_b12x_uses_prebuilt_image() {
    setup_fixture
    run_build --exp-b12x || fail "--exp-b12x run failed"
    assert_log_contains '^docker pull eugr/spark-vllm-b12x:latest$'
    assert_log_contains '^docker tag eugr/spark-vllm-b12x:latest vllm-node-b12x$'
    assert_log_not_contains '^docker build'
    pass "--exp-b12x pulls and tags the tested B12X image"
}

test_exp_b12x_rebuild_vllm_uses_preset_source_build() {
    setup_fixture
    run_build --exp-b12x --rebuild-vllm || fail "--exp-b12x --rebuild-vllm run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm-b12x:latest$'
    assert_log_contains '^docker build --target vllm-export .*--build-arg TORCH_CUDA_ARCH_LIST=12.1a --build-arg FLASHINFER_CUDA_ARCH_LIST=12.1a .*--build-arg TORCH_VERSION=2.13.0 --build-arg TORCHVISION_VERSION=0.28.0 --build-arg TORCHAUDIO_VERSION=2.11.0 --build-arg CUTLASS_DSL_VERSION=4.7.0 .*--build-arg VLLM_REF=dev/infernal-invocation --build-arg VLLM_REPO=https://github.com/local-inference-lab/vllm --build-arg VLLM_APPLY_PRESET_PRS=0 .*--build-arg VLLM_PRESERVE_SM12X_TARGET=1 --build-arg VLLM_PATCH_B12X_C128A_ALIGNMENT=1'
    assert_log_contains '^docker build -t vllm-node-b12x .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/regular --build-context vllm_wheels=\./\.wheel-cache/vllm/b12x .*--build-arg B12X_REPO=https://github.com/lukealonso/b12x.git --build-arg B12X_REF=master '
    assert_log_contains '.*--build-arg B12X_CACHEBUST=[0-9]+'
    assert_log_not_contains 'Dockerfile\.mxfp4'
    assert_output_contains 'Rebuilding vLLM wheels \(--exp-b12x preset\)\.\.\.'
    assert_output_contains 'Building B12X from https://github\.com/lukealonso/b12x\.git ref master for https://github\.com/local-inference-lab/vllm ref dev/infernal-invocation\.'
    pass "--exp-b12x --rebuild-vllm uses the B12X source-build profile"
}

test_exp_b12x_allows_vllm_prs() {
    setup_fixture
    run_build --exp-b12x --apply-vllm-pr 12345 || fail "--exp-b12x with vLLM PR run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=dev/infernal-invocation --build-arg VLLM_REPO=https://github.com/local-inference-lab/vllm --build-arg VLLM_APPLY_PRESET_PRS=0 --build-arg CACHEBUST_VLLM=[0-9]+ --build-arg VLLM_PRS=12345'
    assert_output_contains 'Rebuilding vLLM wheels \(--exp-b12x preset with requested vLLM PRs\)\.\.\.'
    assert_output_contains 'Applying vLLM PRs: 12345'
    pass "--exp-b12x accepts additional vLLM PR patches"
}

test_exp_b12x_respects_custom_tag() {
    setup_fixture
    run_build --exp-b12x -t custom-b12x || fail "--exp-b12x custom-tag run failed"
    assert_log_contains '^docker pull eugr/spark-vllm-b12x:latest$'
    assert_log_contains '^docker tag eugr/spark-vllm-b12x:latest custom-b12x$'
    assert_log_not_contains '^docker build'
    pass "an explicit tag overrides the --exp-b12x default tag"
}

test_exp_b12x_rejects_use_wheels() {
    setup_fixture
    if run_build --exp-b12x --use-wheels; then
        fail "--exp-b12x unexpectedly accepted --use-wheels"
    fi
    assert_log_not_contains '^docker pull '
    assert_log_not_contains '^docker build'
    assert_output_contains 'Error: --exp-b12x is incompatible with --use-wheels because B12X vLLM wheels are not published'
    pass "--exp-b12x rejects the unsupported wheel-only path"
}

test_exp_b12x_rejects_preset_overrides() {
    setup_fixture
    if run_build --exp-b12x --vllm-ref main; then
        fail "--exp-b12x unexpectedly accepted --vllm-ref"
    fi
    assert_log_not_contains '^docker build'
    assert_output_contains 'Error: --exp-b12x is incompatible with --vllm-ref'

    setup_fixture
    if run_build --exp-b12x --exp-mxfp4; then
        fail "--exp-b12x unexpectedly accepted --exp-mxfp4"
    fi
    assert_log_not_contains '^docker build'
    assert_output_contains 'Error: --exp-b12x is incompatible with --exp-mxfp4'
    pass "--exp-b12x rejects conflicting build presets and overrides"
}

test_exp_b12x_variable_names_are_generic() {
    if grep -q 'FATHOMLESS_' "$PROJECT_DIR/build-and-copy.sh"; then
        fail "build-and-copy.sh still contains FATHOMLESS-prefixed variables"
    fi
    for expected in \
        'EXP_B12X_VLLM_REPO=' \
        'EXP_B12X_VLLM_REF=' \
        'EXP_B12X_PACKAGE_REPO=' \
        'EXP_B12X_PACKAGE_REF='; do
        if ! grep -Fq "$expected" "$PROJECT_DIR/build-and-copy.sh"; then
            fail "build-and-copy.sh is missing generic B12X variable: $expected"
        fi
    done
    pass "B12X preset variables use generic EXP_B12X names"
}

test_exp_b12x_preserves_blackwell_arches() {
    local arch
    local nccl_arch
    for arch in 10.3a 12.0a 12.0f; do
        nccl_arch="${arch%[a-z]}"
        nccl_arch="${nccl_arch//./}"
        setup_fixture
        run_build --exp-b12x --gpu-arch "$arch" || \
            fail "--exp-b12x --gpu-arch $arch run failed"
        assert_log_contains "^docker build --target flashinfer-export .*--build-arg FLASHINFER_CUDA_ARCH_LIST=${arch} .*--build-arg FLASHINFER_REF=main"
        assert_log_contains "^docker build --target vllm-export .*--build-arg TORCH_CUDA_ARCH_LIST=${arch} --build-arg FLASHINFER_CUDA_ARCH_LIST=${arch} .*--build-arg NCCL_NVCC_GENCODE=-gencode=arch=compute_${nccl_arch},code=sm_${nccl_arch} .*--build-arg VLLM_PRESERVE_SM12X_TARGET=1"
        assert_log_contains "^docker build -t vllm-node-b12x .*--build-arg TORCH_CUDA_ARCH_LIST=${arch} --build-arg FLASHINFER_CUDA_ARCH_LIST=${arch} "
        assert_output_contains "Rebuilding FlashInfer wheels for GPU architecture ${arch}\.\.\."
    done
    pass "--exp-b12x preserves explicit Blackwell architecture selections"
}

test_exp_b12x_rebuilds_mismatched_cached_flashinfer_arch() {
    setup_fixture
    printf '12.0f\n' > "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/.flashinfer-arch"
    run_build --exp-b12x --rebuild-vllm || fail "--exp-b12x cached-arch run failed"
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_CUDA_ARCH_LIST=12.1a '
    assert_output_contains 'Rebuilding FlashInfer wheels for GPU architecture 12\.1a\.\.\.'
    pass "--exp-b12x does not reuse a FlashInfer wheel for another architecture"
}

test_exp_b12x_rebuilds_mismatched_cached_vllm_arch() {
    setup_fixture
    printf '12.0f\n' > "$FIXTURE_DIR/.wheel-cache/vllm/b12x/.vllm-arch"
    run_build --exp-b12x --rebuild-flashinfer || fail "--exp-b12x mismatched vLLM arch run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg TORCH_CUDA_ARCH_LIST=12.1a '
    assert_output_contains 'Rebuilding vLLM wheels \(--exp-b12x preset\)\.\.\.'
    pass "--exp-b12x does not reuse a vLLM wheel for another architecture"
}

test_b12x_c128a_alignment_patch_is_guarded_and_idempotent() {
    local patch_script="$PROJECT_DIR/docker/patch_vllm_b12x_c128a_topk_alignment.py"
    local patch_fixture="$TMP_BASE/b12x-c128a-patch"
    local target_dir="$patch_fixture/vllm/models/deepseek_v4"
    local helper_dir="$patch_fixture/vllm/v1/attention/backends/mla"
    local target="$target_dir/sparse_mla.py"
    local helper="$helper_dir/compressor_utils.py"
    local output="$patch_fixture/output.log"
    local unknown_fixture="$TMP_BASE/b12x-c128a-unknown"

    mkdir -p "$target_dir" "$helper_dir"
    cat > "$target" <<'PY'
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_c128a_topk_width,
    get_compressed_slot_mapping,
)


def active_width(value: int) -> int:
    return max(value, _C128A_TOPK_ALIGNMENT)
PY
    cat > "$helper" <<'PY'
_C128A_TOPK_ALIGNMENT = 128


def get_c128a_topk_width():
    pass


def get_compressed_slot_mapping():
    pass
PY
    cp -a "$patch_fixture" "$unknown_fixture"
    sed -i 's/_C128A_TOPK_ALIGNMENT = 128/_C128A_TOPK_ALIGNMENT = 64/' \
        "$unknown_fixture/vllm/v1/attention/backends/mla/compressor_utils.py"

    VLLM_PATCH_B12X_C128A_ALIGNMENT=0 python3 "$patch_script" \
        "$patch_fixture" > "$output"
    if grep -Fq '    _C128A_TOPK_ALIGNMENT,' "$target"; then
        fail "B12X C128A patch changed source while its build guard was disabled"
    fi

    VLLM_PATCH_B12X_C128A_ALIGNMENT=1 python3 "$patch_script" \
        "$patch_fixture" >> "$output"
    if [ "$(grep -Fc '    _C128A_TOPK_ALIGNMENT,' "$target")" -ne 1 ]; then
        fail "B12X C128A patch did not add exactly one alignment import"
    fi

    VLLM_PATCH_B12X_C128A_ALIGNMENT=1 python3 "$patch_script" \
        "$patch_fixture" >> "$output"
    if ! grep -Fq \
        'DeepSeek V4 C128A alignment is already defined or imported; skipping' \
        "$output"; then
        fail "B12X C128A patch is not idempotent"
    fi

    if VLLM_PATCH_B12X_C128A_ALIGNMENT=invalid python3 "$patch_script" \
        "$patch_fixture" >> "$output" 2>&1; then
        fail "B12X C128A patch accepted an invalid build guard"
    fi

    if VLLM_PATCH_B12X_C128A_ALIGNMENT=1 python3 "$patch_script" \
        "$unknown_fixture" >> "$output" 2>&1; then
        fail "B12X C128A patch accepted an unexpected helper constant"
    fi
    if grep -Fq '    _C128A_TOPK_ALIGNMENT,' \
        "$unknown_fixture/vllm/models/deepseek_v4/sparse_mla.py"; then
        fail "B12X C128A patch mutated an unknown source shape"
    fi
    pass "B12X C128A alignment workaround is guarded and idempotent"
}

test_dockerfile_preserves_selected_blackwell_target() {
    local blackwell_block="$TMP_BASE/blackwell-block"
    local patch_fixture="$TMP_BASE/blackwell-patch"

    sed -n '/# CUDA 13 vLLM builds normally collapse/,/# TEMPORARY PATCH: vLLM PR #47914/p' \
        "$PROJECT_DIR/Dockerfile" > "$blackwell_block"
    for expected in \
        'VLLM_PRESERVE_SM12X_TARGET="${VLLM_PRESERVE_SM12X_TARGET}"' \
        '/tmp/vllm-patches/patch_vllm_preserve_sm12x_target.py'; do
        if ! grep -Fq "$expected" "$blackwell_block"; then
            fail "Dockerfile Blackwell build guard is missing: $expected"
        fi
    done
    for expected in \
        '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1"' \
        'Enabled selected SM103 and SM12x targets for CUDA 13 vLLM build'; do
        if ! grep -Fq "$expected" \
            "$PROJECT_DIR/docker/patch_vllm_preserve_sm12x_target.py"; then
            fail "External Blackwell patch is missing: $expected"
        fi
    done

    mkdir -p "$patch_fixture"
    printf '%s\n' \
        'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")' \
        > "$patch_fixture/CMakeLists.txt"
    VLLM_PRESERVE_SM12X_TARGET=1 python3 \
        "$PROJECT_DIR/docker/patch_vllm_preserve_sm12x_target.py" \
        "$patch_fixture" > "$patch_fixture/output.log"
    if ! grep -Fq \
        'set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1")' \
        "$patch_fixture/CMakeLists.txt"; then
        fail "Blackwell patch did not add both CUDA 13 subarchitectures"
    fi
    VLLM_PRESERVE_SM12X_TARGET=1 python3 \
        "$PROJECT_DIR/docker/patch_vllm_preserve_sm12x_target.py" \
        "$patch_fixture" >> "$patch_fixture/output.log"
    if ! grep -Fq \
        'CUDA 13 Blackwell subarchitecture allow-list already present; skipping' \
        "$patch_fixture/output.log"; then
        fail "Blackwell patch is not idempotent"
    fi
    pass "B12X preserves selected SM103 and SM12x targets under CUDA 13"
}

test_custom_torch_versions_are_forwarded() {
    setup_fixture
    run_build \
        --vllm-repo https://github.com/local-inference-lab/vllm.git \
        --vllm-ref dev/fathomless-firmament \
        --torch-version 2.12.0 \
        --torchvision-version 0.27.0 \
        --torchaudio-version none || fail "custom Torch version run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg TORCH_VERSION=2.12.0 --build-arg TORCHVISION_VERSION=0.27.0 --build-arg TORCHAUDIO_VERSION=none .*--build-arg VLLM_REF=dev/fathomless-firmament --build-arg VLLM_REPO=https://github.com/local-inference-lab/vllm.git'
    assert_log_contains '^docker build -t vllm-node .*--build-arg TORCH_VERSION=2.12.0 --build-arg TORCHVISION_VERSION=0.27.0 --build-arg TORCHAUDIO_VERSION=none .*--build-arg B12X_REPO=https://github.com/lukealonso/b12x.git --build-arg B12X_REF=master '
    assert_log_contains '.*--build-arg B12X_CACHEBUST=[0-9]+'
    assert_output_contains 'Building B12X from https://github\.com/lukealonso/b12x\.git ref master for https://github\.com/local-inference-lab/vllm ref dev/fathomless-firmament\.'
    pass "Torch versions and the B12X source checkout are forwarded to the fork build"
}

test_local_inference_lab_b12x_applies_to_any_ref() {
    setup_fixture
    run_build \
        --vllm-repo https://github.com/local-inference-lab/vllm \
        --vllm-ref dev/spark-fixes-7-14 \
        --torch-version 2.12.0 || fail "local-inference-lab alternate ref run failed"
    assert_log_contains '^docker build -t vllm-node .*--build-arg TORCH_VERSION=2.12.0 .*--build-arg B12X_REPO=https://github.com/lukealonso/b12x.git --build-arg B12X_REF=master '
    assert_output_contains 'Building B12X from https://github\.com/lukealonso/b12x\.git ref master for https://github\.com/local-inference-lab/vllm ref dev/spark-fixes-7-14\.'
    pass "all local-inference-lab/vllm refs include the B12X source build"
}

test_local_inference_lab_b12x_requires_torch_212() {
    setup_fixture
    if run_build \
        --vllm-repo https://github.com/local-inference-lab/vllm.git \
        --vllm-ref dev/spark-fixes-7-14 \
        --torch-version 2.11.0; then
        fail "local-inference-lab B12X build unexpectedly accepted Torch 2.11"
    fi
    assert_log_not_contains '^docker build'
    assert_output_contains 'Error: https://github\.com/local-inference-lab/vllm requires --torch-version 2\.12\.0 or newer for B12X \(got 2\.11\.0\)\.'
    pass "local-inference-lab B12X builds reject Torch versions older than 2.12"
}

test_dockerfile_custom_repo_bypasses_shared_cache() {
    for expected in \
        'Custom vLLM repository selected; bypassing shared checkout cache.' \
        'git clone --recursive "$VLLM_REPO" /tmp/vllm-custom' \
        'cp -a /tmp/vllm-custom "$VLLM_BASE_DIR/vllm"'; do
        if ! grep -Fq "$expected" "$PROJECT_DIR/Dockerfile"; then
            fail "Dockerfile custom repository block is missing: $expected"
        fi
    done
    pass "custom vLLM repositories bypass the shared upstream checkout cache"
}

test_dockerfile_uses_configurable_torch_versions() {
    if [ "$(grep -Fc 'set -- "torch==$TORCH_VERSION" "$TORCHVISION_SPEC"' "$PROJECT_DIR/Dockerfile")" -ne 2 ] || \
       [ "$(grep -Fc 'uv pip install "$@" triton' "$PROJECT_DIR/Dockerfile")" -ne 2 ]; then
        fail "Dockerfile does not use configurable Torch package specs in both build and runner stages"
    fi
    if [ "$(grep -Fc 'echo "torchvision==${PINNED_TORCHVISION}"' "$PROJECT_DIR/Dockerfile")" -ne 2 ] || \
       [ "$(grep -Fc 'echo "torchaudio==${PINNED_TORCHAUDIO}"' "$PROJECT_DIR/Dockerfile")" -ne 2 ]; then
        fail "Dockerfile does not preserve the selected Torch-family versions during later installs"
    fi
    for expected in \
        'ARG TORCH_VERSION=2.13.0' \
        'ARG TORCHVISION_VERSION=0.28.0' \
        'ARG TORCHAUDIO_VERSION=2.11.0'; do
        if ! grep -Fq "$expected" "$PROJECT_DIR/Dockerfile"; then
            fail "Dockerfile is missing Torch-family default: $expected"
        fi
    done
    pass "Dockerfile uses configurable Torch package versions in build and runner stages"
}

test_dockerfile_pins_cutlass_dsl_47_everywhere() {
    for expected in \
        'ARG CUTLASS_DSL_VERSION=4.7.0' \
        '"nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VERSION"' \
        'echo "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" >> /tmp/wheel-override.txt' \
        'echo "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" >> /tmp/torch-override.txt' \
        '"$CUTLASS_DSL_VERSION" --expected-count 1 requirements/cuda.txt' \
        '--expected-count 5 /tmp/b12x-source/pyproject.toml'; do
        if ! grep -Fq -- "$expected" "$PROJECT_DIR/Dockerfile"; then
            fail "Dockerfile is missing CUTLASS DSL 4.7 enforcement: $expected"
        fi
    done
    pass "Dockerfile pins CUTLASS DSL 4.7 in regular and B12X builds"
}

test_dockerfile_uses_profiled_named_wheel_contexts() {
    for expected in \
        'from=flashinfer_wheels,target=/workspace/flashinfer-wheels' \
        'from=vllm_wheels,target=/workspace/vllm-wheels' \
        '/workspace/flashinfer-wheels/*.whl /workspace/vllm-wheels/*.whl' \
        'printf '\''%s\n'\'' "$TORCH_CUDA_ARCH_LIST" > /workspace/wheels/.vllm-arch'; do
        if ! grep -Fq "$expected" "$PROJECT_DIR/Dockerfile"; then
            fail "Dockerfile named wheel context support is missing: $expected"
        fi
    done
    if grep -Fq 'source=wheels,target=/workspace/wheels' "$PROJECT_DIR/Dockerfile"; then
        fail "Dockerfile still mounts the legacy combined wheel context"
    fi
    pass "Dockerfile mounts independent profiled wheel contexts"
}

test_dockerfile_builds_and_verifies_b12x_source() {
    for expected in \
        'git clone --depth 1 --branch "$B12X_REF" "$B12X_REPO" /tmp/b12x-source' \
        'Refreshing B12X source (cache key: $B12X_CACHEBUST)' \
        'uv pip install --reinstall --no-deps /tmp/b12x-source' \
        "import b12x; print('Verified B12X'" \
        "m.version('b12x')" \
        '/workspace/b12x-source-commit' \
        "m.version('nvidia-cutlass-dsl')"; do
        if ! grep -Fq "$expected" "$PROJECT_DIR/Dockerfile"; then
            fail "Dockerfile B12X source build is missing: $expected"
        fi
    done
    if grep -Fq 'b12x==' "$PROJECT_DIR/Dockerfile"; then
        fail "Dockerfile installs B12X from a package index instead of source"
    fi
    if grep -Eq "import sparkinfer|m\.version\('sparkinfer'\)" "$PROJECT_DIR/Dockerfile"; then
        fail "Dockerfile still verifies the retired sparkinfer package name"
    fi
    pass "Dockerfile builds B12X from source without replacing vLLM dependencies"
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

test_dockerfile_uses_prepared_python_for_flashinfer_builds() {
    local flashinfer_builder_block="$TMP_BASE/flashinfer-builder-block"

    sed -n '/FROM base AS flashinfer-builder/,/FROM scratch AS flashinfer-export/p' \
        "$PROJECT_DIR/Dockerfile" > "$flashinfer_builder_block"

    for expected in \
        'ARG FLASHINFER_BUILD_PYTHON=/usr/bin/python3' \
        'ENV UV_PYTHON_DOWNLOADS=never' \
        'uv pip install --python "$FLASHINFER_BUILD_PYTHON" packaging filelock' \
        '"$FLASHINFER_BUILD_PYTHON" -c '\''import filelock, packaging, requests, torch, tqdm'\'''; do
        if ! grep -Fq "$expected" "$flashinfer_builder_block"; then
            fail "FlashInfer builder does not prepare its selected Python consistently: $expected"
        fi
    done

    if [ "$(grep -Fc 'uv build --python "$FLASHINFER_BUILD_PYTHON" --no-build-isolation' "$flashinfer_builder_block")" -ne 3 ]; then
        fail "FlashInfer builder does not use the prepared Python for all three wheel builds"
    fi
    if grep -Fq 'uv build --no-build-isolation' "$flashinfer_builder_block"; then
        fail "FlashInfer builder still contains a build that can honor upstream .python-version"
    fi
    pass "FlashInfer wheel builds use the prepared system Python"
}

test_dockerfiles_pin_tvm_ffi_regression_version() {
    if [ "$(grep -Fc 'apache-tvm-ffi==0.1.12' "$PROJECT_DIR/Dockerfile")" -ne 2 ] || \
       [ "$(grep -Fc 'apache-tvm-ffi==0.1.12' "$PROJECT_DIR/Dockerfile.mxfp4")" -ne 1 ]; then
        fail "Dockerfiles do not pin every TVM-FFI install to the known-good 0.1.12 release"
    fi
    pass "Dockerfiles pin TVM-FFI to the known-good 0.1.12 release"
}

test_dockerfile_fetches_vllm_prs_from_upstream() {
    local vllm_pr_block="$TMP_BASE/vllm-pr-block"

    sed -n '/ARG VLLM_PRS=""/,/# TEMPORARY PATCH: vLLM PR/p' "$PROJECT_DIR/Dockerfile" > "$vllm_pr_block"
    for expected in \
        'git remote add vllm-upstream "$VLLM_UPSTREAM_REPO"' \
        'git fetch vllm-upstream +pull/${pr}/head:pr-${pr}' \
        'git merge-base vllm-upstream/main pr-${pr}'; do
        if ! grep -Fq "$expected" "$vllm_pr_block"; then
            fail "vLLM PR block does not use the dedicated upstream remote: $expected"
        fi
    done
    pass "vLLM PR patches are fetched from upstream when building a fork"
}

test_dockerfile_externalizes_vllm_source_patches() {
    local patch_section="$TMP_BASE/vllm-source-patches"
    local patch_count=0

    sed -n '/COPY docker\/patch_vllm_\*\.py/,/# Prepare build requirements/p' \
        "$PROJECT_DIR/Dockerfile" > "$patch_section"
    if grep -Eq 'python3[[:space:]]+-[[:space:]]+<<' "$patch_section"; then
        fail "Dockerfile still embeds a Python vLLM source patch"
    fi

    for patch_path in "$PROJECT_DIR"/docker/patch_vllm_*.py; do
        patch_name="$(basename "$patch_path")"
        patch_count=$((patch_count + 1))
        if ! grep -Fq "/tmp/vllm-patches/$patch_name" "$patch_section"; then
            fail "Dockerfile does not execute external patch: $patch_name"
        fi
    done
    if [ "$patch_count" -ne 11 ]; then
        fail "Expected 11 external vLLM patch scripts, found $patch_count"
    fi
    if ! python3 -c '
from pathlib import Path
import sys
files = sorted(Path(sys.argv[1]).glob("patch_vllm_*.py"))
for path in files:
    compile(path.read_text(), str(path), "exec")
' "$PROJECT_DIR/docker"; then
        fail "An external vLLM patch script has invalid Python syntax"
    fi
    pass "Dockerfile externalizes every active vLLM source patch"
}

test_default_uses_prebuilt
test_tf5_uses_prebuilt_tf5_tag
test_custom_tag_uses_prebuilt_custom_tag
test_default_gpu_arch_stays_prebuilt
test_non_default_gpu_arch_uses_wheel_build
test_default_prebuilt_ignores_local_flashinfer_arch
test_regular_build_rebuilds_mismatched_cached_flashinfer_arch
test_regular_build_reuses_matching_cached_flashinfer_arch
test_use_wheels_rejects_mismatched_flashinfer_arch
test_use_wheels_rejects_mismatched_vllm_arch
test_use_wheels_non_default_empty_cache_skips_downloads
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
test_custom_vllm_repo_forces_source_build
test_exp_b12x_uses_prebuilt_image
test_exp_b12x_rebuild_vllm_uses_preset_source_build
test_exp_b12x_allows_vllm_prs
test_exp_b12x_respects_custom_tag
test_exp_b12x_rejects_use_wheels
test_exp_b12x_rejects_preset_overrides
test_exp_b12x_variable_names_are_generic
test_exp_b12x_preserves_blackwell_arches
test_exp_b12x_rebuilds_mismatched_cached_flashinfer_arch
test_exp_b12x_rebuilds_mismatched_cached_vllm_arch
test_b12x_c128a_alignment_patch_is_guarded_and_idempotent
test_dockerfile_preserves_selected_blackwell_target
test_custom_torch_versions_are_forwarded
test_local_inference_lab_b12x_applies_to_any_ref
test_local_inference_lab_b12x_requires_torch_212
test_dockerfile_custom_repo_bypasses_shared_cache
test_dockerfile_uses_configurable_torch_versions
test_dockerfile_pins_cutlass_dsl_47_everywhere
test_dockerfile_uses_profiled_named_wheel_contexts
test_dockerfile_builds_and_verifies_b12x_source
test_copied_vllm_git_index_is_refreshed_before_patch_apply
test_dockerfile_applies_flashinfer_prs_without_merging_branch_history
test_dockerfile_uses_prepared_python_for_flashinfer_builds
test_dockerfiles_pin_tvm_ffi_regression_version
test_dockerfile_fetches_vllm_prs_from_upstream
test_dockerfile_externalizes_vllm_source_patches

echo "Passed $TESTS_PASSED build-and-copy tests."
