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

write_wheel_metadata() {
    python3 - "$@" <<'PY'
import sys
from zipfile import ZipFile

path, name, version, *requirements = sys.argv[1:]
metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
metadata += "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements)
with ZipFile(path, "w") as archive:
    archive.writestr(f"{name.replace('-', '_')}-{version}.dist-info/METADATA", metadata + "\n")
PY
}
# The fake Docker exporter uses the same metadata fixtures as the local cache.
export -f write_wheel_metadata

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
    LOCAL_INSPECT_FILE="$PROJECT_DIR/tests/fixtures/image-identity/classic.json"
    REMOTE_INSPECT_FILE="$PROJECT_DIR/tests/fixtures/image-identity/containerd.json"
    export LOCAL_INSPECT_FILE REMOTE_INSPECT_FILE
    mkdir -p \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/regular" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/custom" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/b12x" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/b12x-custom" \
        "$FIXTURE_DIR/.wheel-cache/vllm/regular" \
        "$FIXTURE_DIR/.wheel-cache/vllm/b12x" \
        "$FIXTURE_DIR/.wheel-cache/vllm/custom"
    touch \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_cubin-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_python-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/vllm/regular/vllm-test.whl" \
        "$FIXTURE_DIR/.wheel-cache/vllm/b12x/vllm-test.whl"
    write_wheel_metadata "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_jit_cache-test.whl" \
        flashinfer-jit-cache 0.7.0
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
        build_cubin=1
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --target) target="$2"; shift 2 ;;
                --output) output="$2"; shift 2 ;;
                --build-arg)
                    case "$2" in
                        TORCH_CUDA_ARCH_LIST=*) build_arch="${2#TORCH_CUDA_ARCH_LIST=}" ;;
                        FLASHINFER_CUDA_ARCH_LIST=*) build_arch="${2#FLASHINFER_CUDA_ARCH_LIST=}" ;;
                        FLASHINFER_BUILD_CUBIN=*) build_cubin="${2#FLASHINFER_BUILD_CUBIN=}" ;;
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
                        if [ "$build_cubin" = 1 ]; then
                            printf 'fake wheel\n' > "$dest/flashinfer_cubin-built.whl"
                        fi
                        requirements=()
                        if [ "${MOCK_FLASHINFER_SPLIT_EXPORT:-}" != "" ]; then
                            requirements=('flashinfer-jit-cache-sm121a==0.7.0')
                            if [ "$MOCK_FLASHINFER_SPLIT_EXPORT" = complete ]; then
                                write_wheel_metadata "$dest/flashinfer_jit_cache_sm121a-0.7.0-py3-none-any.whl" \
                                    flashinfer-jit-cache-sm121a 0.7.0
                            fi
                        fi
                        write_wheel_metadata "$dest/flashinfer_jit_cache-built.whl" \
                            flashinfer-jit-cache 0.7.0 "${requirements[@]}"
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
    if [ "${LOCAL_IMAGE_MISSING:-false}" = true ]; then
        exit 1
    fi
    if [ "${3:-}" = "--format" ]; then
        echo "${LOCAL_IMAGE_ID:-sha256:local}"
        exit 0
    fi
    cat "$LOCAL_INSPECT_FILE"
    exit "${LOCAL_INSPECT_STATUS:-0}"
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
    if [[ "$cmd" != *"--format"* ]]; then
        cat "$REMOTE_INSPECT_FILE"
        exit "${REMOTE_INSPECT_STATUS:-0}"
    fi
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
[ -n "${MOCK_WHEEL_RELEASE_DIR:-}" ] || exit 22
url=""
output=""
headers=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) output="$2"; shift 2 ;;
        -*I*) headers=true; shift ;;
        https://*) url="$1"; shift ;;
        *) shift ;;
    esac
done
case "$url" in
    */releases/expanded_assets/prebuilt-flashinfer-current)
        for wheel in "$MOCK_WHEEL_RELEASE_DIR"/flashinfer*.whl; do
            printf '<a href="/eugr/spark-vllm-docker/releases/download/prebuilt-flashinfer-current/%s">wheel</a>\n' "${wheel##*/}"
        done
        ;;
    */releases/tag/prebuilt-flashinfer-current)
        echo 'Prebuilt FlashInfer Wheels (0.7.0-deadbeef-d20260928)'
        ;;
    */releases/download/prebuilt-flashinfer-current/*)
        if [ "$headers" = true ]; then
            echo 'Last-Modified: Sat, 01 Jan 2000 00:00:00 GMT'
        elif [ "${url##*/}" = "${MOCK_WHEEL_FAIL_ASSET:-}" ]; then
            exit 22
        else
            cp "$MOCK_WHEEL_RELEASE_DIR/${url##*/}" "$output"
        fi
        ;;
    *) exit 22 ;;
esac
CURL

    chmod +x "$FAKE_BIN_DIR/docker" "$FAKE_BIN_DIR/ssh" "$FAKE_BIN_DIR/curl"
}

run_build() {
    (
        cd "$FIXTURE_DIR"
        PATH="$FAKE_BIN_DIR:$PATH" TEST_LOG="$TEST_LOG" ./build-and-copy.sh --config "$FIXTURE_DIR/test.env" "$@"
    ) > "$OUTPUT_LOG" 2>&1
}

create_local_vllm_source() {
    LOCAL_VLLM_SOURCE_DIR="$CASE_DIR/local-vllm"
    mkdir -p "$LOCAL_VLLM_SOURCE_DIR"
    git -C "$LOCAL_VLLM_SOURCE_DIR" init -q
    git -C "$LOCAL_VLLM_SOURCE_DIR" config user.name "Build Test"
    git -C "$LOCAL_VLLM_SOURCE_DIR" config user.email "build-test@example.invalid"
    git -C "$LOCAL_VLLM_SOURCE_DIR" config commit.gpgsign false
    printf '[build-system]\nrequires = []\n' > "$LOCAL_VLLM_SOURCE_DIR/pyproject.toml"
    git -C "$LOCAL_VLLM_SOURCE_DIR" add pyproject.toml
    git -C "$LOCAL_VLLM_SOURCE_DIR" commit -q -m "local vLLM fixture"
    git -C "$LOCAL_VLLM_SOURCE_DIR" branch -M qwen38next
    LOCAL_VLLM_SOURCE_COMMIT=$(git -C "$LOCAL_VLLM_SOURCE_DIR" rev-parse HEAD)
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
        "$FIXTURE_DIR/.wheel-cache/flashinfer/custom/flashinfer_python-test.whl"
    write_wheel_metadata "$FIXTURE_DIR/.wheel-cache/flashinfer/custom/flashinfer_jit_cache-test.whl" \
        flashinfer-jit-cache 0.7.0
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

test_missing_flashinfer_provider_stops_before_vllm_build() {
    setup_fixture
    write_wheel_metadata "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_jit_cache-test.whl" \
        flashinfer-jit-cache 0.7.0 'flashinfer-jit-cache-sm121a==0.7.0'
    if run_build --rebuild-vllm; then
        fail "missing FlashInfer provider unexpectedly passed validation"
    fi
    assert_log_not_contains '^docker build'
    assert_output_contains 'provider wheel for arch sm121a .* is missing'
    assert_output_contains 'Re-run with --rebuild-flashinfer.*GPU arch 12\.1a'
    pass "missing cached FlashInfer provider stops before vLLM or runner builds"
}

test_complete_flashinfer_provider_cache_builds_runner() {
    setup_fixture
    write_wheel_metadata "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_jit_cache-test.whl" \
        flashinfer-jit-cache 0.7.0 'flashinfer-jit-cache-sm121a==0.7.0'
    write_wheel_metadata "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_jit_cache_sm121a-0.7.0-py3-none-any.whl" \
        flashinfer-jit-cache-sm121a 0.7.0
    run_build --use-wheels || fail "complete provider cache was rejected"
    assert_log_not_contains '^docker build --target '
    assert_log_contains '^docker build -t vllm-node '
    pass "complete FlashInfer provider cache builds the runner without recompiling"
}

test_incomplete_flashinfer_export_preserves_cache() {
    setup_fixture
    local cached_jit="$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_jit_cache-test.whl"
    cp "$cached_jit" "$CASE_DIR/previous-jit.whl"
    if MOCK_FLASHINFER_SPLIT_EXPORT=missing run_build --rebuild-flashinfer; then
        fail "missing exported FlashInfer provider unexpectedly passed validation"
    fi
    assert_log_contains '^docker build --target flashinfer-export '
    assert_log_not_contains '^docker build --target vllm-export '
    assert_log_not_contains '^docker build -t '
    assert_output_contains 'provider wheel for arch sm121a .* is missing'
    assert_output_contains 'keeping the previous wheel profile unchanged'
    cmp -s "$cached_jit" "$CASE_DIR/previous-jit.whl" || fail "old FlashInfer cache changed"
    if [ -f "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_jit_cache-built.whl" ]; then
        fail "incomplete FlashInfer export replaced cached wheels"
    fi
    pass "incomplete FlashInfer export preserves the previous wheel cache"
}

test_complete_flashinfer_export_builds_runner() {
    setup_fixture
    MOCK_FLASHINFER_SPLIT_EXPORT=complete run_build --rebuild-flashinfer || fail "complete FlashInfer export failed"
    assert_log_contains '^docker build --target flashinfer-export '
    assert_log_contains '^docker build -t vllm-node '
    [ -f "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/flashinfer_jit_cache_sm121a-0.7.0-py3-none-any.whl" ] || \
        fail "complete FlashInfer provider was not promoted"
    pass "complete FlashInfer export promotes providers and builds the runner"
}

test_flashinfer_provider_validation() {
    if ! python3 "$PROJECT_DIR/tests/test_flashinfer_wheel_validation.py"; then
        fail "FlashInfer wheel validation regression tests failed"
    fi
    pass "FlashInfer validators check provider metadata and retain monolithic support"
}

create_flashinfer_release() {
    RELEASE_DIR="$CASE_DIR/release"
    mkdir -p "$RELEASE_DIR"
    write_wheel_metadata "$RELEASE_DIR/flashinfer_cubin-0.7.0-py3-none-any.whl" flashinfer-cubin 0.7.0
    write_wheel_metadata "$RELEASE_DIR/flashinfer_python-0.7.0-py3-none-any.whl" flashinfer-python 0.7.0
    local requirements=()
    if [ "$1" != monolithic ]; then
        requirements=('flashinfer-jit-cache-sm121a==0.7.0')
    fi
    write_wheel_metadata "$RELEASE_DIR/flashinfer_jit_cache-0.7.0-cp39-abi3-manylinux_2_28_aarch64.whl" \
        flashinfer-jit-cache 0.7.0 "${requirements[@]}"
    if [ "$1" = complete ]; then
        write_wheel_metadata "$RELEASE_DIR/flashinfer_jit_cache_sm121a-0.7.0-cp39-abi3-manylinux_2_28_aarch64.whl" \
            flashinfer-jit-cache-sm121a 0.7.0
    fi
}

test_flashinfer_release_download_includes_device_provider() {
    setup_fixture
    create_flashinfer_release complete
    rm "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/"*.whl
    MOCK_WHEEL_RELEASE_DIR="$RELEASE_DIR" run_build --rebuild-vllm || fail "regular build with published providers failed"
    assert_log_contains '^curl -fL .*flashinfer_jit_cache_sm121a-0.7.0-'
    assert_log_contains '^docker build --target vllm-export '
    assert_log_contains '^docker build -t vllm-node '
    assert_log_not_contains '^docker build --target flashinfer-export '
    pass "regular source build downloads the sm121a provider alongside the shim"
}

test_monolithic_flashinfer_release_download_still_works() {
    setup_fixture
    create_flashinfer_release monolithic
    MOCK_WHEEL_RELEASE_DIR="$RELEASE_DIR" run_build --use-wheels --force-flashinfer-download || fail "monolithic release failed"
    assert_log_contains '^docker build -t vllm-node '
    assert_log_not_contains '^curl -fL .*flashinfer_jit_cache_sm'
    pass "legacy monolithic FlashInfer release downloads need no device provider"
}

test_incomplete_flashinfer_download_preserves_cache() {
    setup_fixture
    create_flashinfer_release missing
    local cache="$FIXTURE_DIR/.wheel-cache/flashinfer/regular"
    printf 'previous-commit\n' > "$cache/.flashinfer-commit"
    printf '12.1a\n' > "$cache/.flashinfer-arch"
    cp -a "$cache" "$CASE_DIR/previous-cache"
    MOCK_WHEEL_RELEASE_DIR="$RELEASE_DIR" run_build --use-wheels --force-flashinfer-download || fail "valid old cache was not reused"
    assert_output_contains "FlashInfer release 'prebuilt-flashinfer-current' contains an incomplete or invalid wheel set"
    assert_output_contains 'Restoring previous flashinfer wheels'
    diff -r "$cache" "$CASE_DIR/previous-cache" || fail "incomplete release changed cached wheels or provenance"
    assert_log_contains '^docker build -t vllm-node '
    pass "an incomplete published set restores the previous FlashInfer cache and markers"
}

test_incomplete_flashinfer_download_without_cache_fails() {
    setup_fixture
    create_flashinfer_release missing
    rm "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/"*.whl
    if MOCK_WHEEL_RELEASE_DIR="$RELEASE_DIR" run_build --rebuild-vllm; then
        fail "incomplete published wheels unexpectedly allowed vLLM compilation"
    fi
    assert_log_not_contains '^docker build'
    assert_output_contains "FlashInfer release 'prebuilt-flashinfer-current' contains an incomplete or invalid wheel set"
    if compgen -G "$FIXTURE_DIR/.wheel-cache/flashinfer/regular/*.whl" >/dev/null; then
        fail "incomplete downloaded wheels remained in the cache"
    fi
    pass "incomplete releases fail before compilation when no valid cache exists"
}

test_repaired_release_bypasses_incomplete_cache_shortcuts() {
    local shortcut
    for shortcut in newer matching-commit; do
        setup_fixture
        create_flashinfer_release complete
        local cache="$FIXTURE_DIR/.wheel-cache/flashinfer/regular"
        rm "$cache/"*.whl
        cp "$RELEASE_DIR/"*.whl "$cache/"
        printf 'deadbeef\n' > "$cache/.flashinfer-commit"
        if [ "$shortcut" = newer ]; then
            rm "$cache/"flashinfer_jit_cache_sm*.whl
        else
            printf 'broken wheel\n' > "$cache/flashinfer_jit_cache_sm121a-0.7.0-cp39-abi3-manylinux_2_28_aarch64.whl"
        fi
        MOCK_WHEEL_RELEASE_DIR="$RELEASE_DIR" run_build --use-wheels || fail "$shortcut invalid cache skipped repaired release"
        assert_log_contains '^curl -fL .*flashinfer_jit_cache_sm121a-0.7.0-'
        assert_log_contains '^docker build -t vllm-node '
    done
    pass "repaired releases replace incomplete caches despite matching commits or newer timestamps"
}

test_valid_newer_flashinfer_cache_skips_download() {
    setup_fixture
    create_flashinfer_release complete
    MOCK_WHEEL_RELEASE_DIR="$RELEASE_DIR" run_build --use-wheels || fail "valid newer cache failed"
    assert_output_contains 'Local flashinfer wheels are newer than release'
    assert_log_not_contains '^curl -fL '
    pass "valid newer locally built FlashInfer wheels still skip downloads"
}

test_failed_flashinfer_asset_download_restores_cache() {
    setup_fixture
    create_flashinfer_release complete
    local cache="$FIXTURE_DIR/.wheel-cache/flashinfer/regular"
    printf 'previous-commit\n' > "$cache/.flashinfer-commit"
    cp -a "$cache" "$CASE_DIR/previous-cache"
    MOCK_WHEEL_RELEASE_DIR="$RELEASE_DIR" MOCK_WHEEL_FAIL_ASSET=flashinfer_python-0.7.0-py3-none-any.whl \
        run_build --use-wheels --force-flashinfer-download || fail "download failure lost the previous cache"
    diff -r "$cache" "$CASE_DIR/previous-cache" || fail "download failure changed cached wheels or provenance"
    assert_output_contains 'Failed to download flashinfer_python'
    assert_output_contains 'Restoring previous flashinfer wheels'
    pass "failed wheel transfers restore the complete previous cache"
}

test_regular_build_omits_b12x_package() {
    setup_fixture
    run_build --use-wheels || fail "regular B12X package run failed"
    assert_log_contains '^docker build -t vllm-node .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/regular --build-context vllm_wheels=\./\.wheel-cache/vllm/regular '
    assert_log_not_contains 'B12X_(FROM_PYPI|REPO|REF|CACHEBUST)='
    pass "regular builds use upstream FlashInfer without a separate B12X package"
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
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_REPO=https://github.com/flashinfer-ai/flashinfer.git --build-arg FLASHINFER_BUILD_CUBIN=1 '
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
    REMOTE_INSPECT_FILE="$CASE_DIR/different.json"
    sed 's/2222222222/3333333333/g' "$LOCAL_INSPECT_FILE" > "$REMOTE_INSPECT_FILE"
    run_build -c samehost,diffhost,host1 --copy-parallel || fail "mixed remote copy run failed"
    assert_log_contains '^docker save -o .* vllm-node$'
    assert_log_not_contains '^ssh .*@samehost docker load$'
    assert_log_contains '^ssh .*@host1 docker load$'
    assert_log_contains '^ssh .*@diffhost docker load$'
    pass "copy loads only hosts whose image content is missing or different"
}

test_copy_skips_matching_content_across_stores() {
    for parallel in false true; do
        setup_fixture
        if [ "$parallel" = true ]; then
            run_build --no-build -c diffhost --copy-parallel || fail "parallel mixed-store copy check failed"
        else
            run_build --no-build -c diffhost || fail "mixed-store copy check failed"
        fi
        assert_log_contains '^docker image inspect sha256:local$'
        assert_log_contains '^ssh .*@diffhost docker image inspect sha256:remote$'
        assert_log_not_contains '^docker save '
        assert_log_not_contains '^ssh .* docker load$'
        assert_output_contains 'skipping \(matching image content\)'
    done
    pass "matching content skips save/load across stores in serial and parallel modes"
}

test_copy_mixed_stores_only_loads_missing_hosts() {
    setup_fixture
    run_build --no-build -c samehost,diffhost,host1 --copy-parallel || fail "mixed-store partial copy failed"
    assert_log_contains '^docker save '
    assert_log_not_contains '^ssh .*@samehost docker load$'
    assert_log_not_contains '^ssh .*@diffhost docker load$'
    assert_log_contains '^ssh .*@host1 docker load$'
    pass "mixed-store content matches skip individual hosts during a parallel copy"
}

test_copy_detects_changed_runtime_config() {
    setup_fixture
    REMOTE_INSPECT_FILE="$CASE_DIR/different-config.json"
    sed 's/"bash"/"python3"/' "$LOCAL_INSPECT_FILE" > "$REMOTE_INSPECT_FILE"
    run_build --no-build -c diffhost || fail "config mismatch copy run failed"
    assert_log_contains '^docker save '
    assert_log_contains '^ssh .*@diffhost docker load$'
    pass "matching layers with different runtime config still trigger copy"
}

test_copy_does_not_skip_unreadable_content() {
    for side in local remote; do
        for failure in malformed command; do
            setup_fixture
            if [ "$failure" = malformed ]; then
                if [ "$side" = local ]; then
                    LOCAL_INSPECT_FILE=/dev/null
                else
                    REMOTE_INSPECT_FILE=/dev/null
                fi
                run_build --no-build -c diffhost || fail "unreadable metadata copy failed"
            elif [ "$side" = local ]; then
                LOCAL_INSPECT_STATUS=1 run_build --no-build -c diffhost || fail "local inspect failure copy failed"
            else
                REMOTE_INSPECT_STATUS=1 run_build --no-build -c diffhost || fail "remote inspect failure copy failed"
            fi
            assert_log_contains '^docker save '
            assert_log_contains '^ssh .*@diffhost docker load$'
        done
    done
    pass "unreadable metadata or failed inspection cannot skip copying different IDs"
}

test_copy_missing_local_image_fails() {
    setup_fixture
    if LOCAL_IMAGE_MISSING=true run_build --no-build -c diffhost; then
        fail "copy succeeded without a local image"
    fi
    assert_output_contains "Local image 'vllm-node' not found"
    assert_log_not_contains '^docker save '
    assert_log_not_contains '^ssh .* docker load$'
    pass "missing local image stops copying"
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
    pass "ordinary main source rebuild enables configured preset vLLM PRs by default"
}

test_apply_vllm_pr_skips_preset_prs_by_default() {
    setup_fixture
    run_build --apply-vllm-pr 12345 || fail "--apply-vllm-pr run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=main .*--build-arg VLLM_APPLY_PRESET_PRS=0 .*--build-arg VLLM_PRS=12345'
    assert_output_contains 'Skipping preset vLLM PRs because --vllm-repo, --vllm-ref, or --apply-vllm-pr was specified\.'
    pass "--apply-vllm-pr suppresses preset PRs by default"
}

test_apply_vllm_pr_url_is_forwarded_to_source_build() {
    setup_fixture
    local pr_url="https://github.com/local-inference-lab/vllm/pull/669"

    run_build --apply-vllm-pr "${pr_url}/" \
        || fail "full --apply-vllm-pr URL run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=main .*--build-arg VLLM_APPLY_PRESET_PRS=0 .*--build-arg VLLM_PRS=https://github.com/local-inference-lab/vllm/pull/669'
    assert_output_contains 'Applying vLLM PRs: https://github\.com/local-inference-lab/vllm/pull/669'
    pass "full --apply-vllm-pr URL is normalized and forwarded to the source build"
}

test_apply_vllm_pr_rejects_invalid_reference() {
    setup_fixture
    if run_build --apply-vllm-pr 'https://example.com/example/vllm/pull/1'; then
        fail "invalid --apply-vllm-pr URL unexpectedly succeeded"
    fi
    assert_output_contains 'requires a positive integer PR number or full https://github\.com/OWNER/REPO/pull/NUMBER URL'
    assert_log_not_contains '^docker build'
    pass "invalid --apply-vllm-pr reference is rejected before build"
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
    assert_log_not_contains 'B12X_FROM_PYPI=1'
    assert_log_not_contains 'VLLM_PATCH_B12X_C128A_ALIGNMENT=1'
    assert_output_contains 'Rebuilding vLLM wheels \(--vllm-repo specified\)\.\.\.'
    assert_output_contains 'Skipping preset vLLM PRs because --vllm-repo, --vllm-ref, or --apply-vllm-pr was specified\.'
    pass "--vllm-repo forces a source build and suppresses upstream preset PRs"
}

test_local_vllm_source_builds_selected_ref() {
    setup_fixture
    create_local_vllm_source
    run_build --vllm-source-dir "$LOCAL_VLLM_SOURCE_DIR" --vllm-ref qwen38next || \
        fail "--vllm-source-dir run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=qwen38next --build-arg VLLM_REPO=local-source .*--build-context vllm_source=.*/spark-vllm-source\.[^/]+/vllm --build-arg VLLM_SOURCE_MODE=local --build-arg VLLM_SOURCE_COMMIT='"$LOCAL_VLLM_SOURCE_COMMIT"
    assert_log_contains '^docker build -t vllm-node .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/regular --build-context vllm_wheels=\./\.wheel-cache/vllm/custom '
    assert_log_not_contains 'B12X_REPO='
    assert_output_contains 'Using clean local vLLM source at commit '"$LOCAL_VLLM_SOURCE_COMMIT"'\.'
    assert_output_contains 'Rebuilding vLLM wheels \(--vllm-source-dir specified\)\.\.\.'
    assert_output_contains 'Skipping preset vLLM PRs because --vllm-source-dir was specified\.'
    if grep -Fq "$LOCAL_VLLM_SOURCE_DIR" "$OUTPUT_LOG"; then
        fail "local source path leaked into build output"
    fi
    if [ "$(git -C "$LOCAL_VLLM_SOURCE_DIR" branch --show-current)" != "qwen38next" ] || \
       [ -n "$(git -C "$LOCAL_VLLM_SOURCE_DIR" status --porcelain)" ]; then
        fail "local source checkout was modified by the build wrapper"
    fi
    pass "--vllm-source-dir stages and builds the selected local ref"
}

test_local_vllm_source_defaults_to_head() {
    setup_fixture
    create_local_vllm_source
    run_build --vllm-source-dir "$LOCAL_VLLM_SOURCE_DIR" || \
        fail "--vllm-source-dir HEAD run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF='"$LOCAL_VLLM_SOURCE_COMMIT"' --build-arg VLLM_REPO=local-source '
    assert_log_contains 'VLLM_SOURCE_COMMIT='"$LOCAL_VLLM_SOURCE_COMMIT"
    pass "--vllm-source-dir defaults to the checkout HEAD"
}

test_local_vllm_source_rejects_conflicting_repo() {
    setup_fixture
    create_local_vllm_source
    if run_build \
        --vllm-source-dir "$LOCAL_VLLM_SOURCE_DIR" \
        --vllm-repo https://github.com/example/vllm.git; then
        fail "--vllm-source-dir unexpectedly accepted --vllm-repo"
    fi
    assert_log_not_contains '^docker build'
    assert_output_contains 'Error: --vllm-source-dir is incompatible with --vllm-repo\.'
    pass "--vllm-source-dir rejects an explicit repository"
}

test_local_vllm_source_rejects_dirty_checkout() {
    setup_fixture
    create_local_vllm_source
    printf '# dirty\n' >> "$LOCAL_VLLM_SOURCE_DIR/pyproject.toml"
    if run_build --vllm-source-dir "$LOCAL_VLLM_SOURCE_DIR"; then
        fail "--vllm-source-dir unexpectedly accepted a dirty checkout"
    fi
    assert_log_not_contains '^docker build'
    assert_output_contains 'Error: --vllm-source-dir must be clean; commit or stash local changes first\.'
    pass "--vllm-source-dir rejects a dirty checkout"
}

test_local_vllm_source_rejects_missing_ref() {
    setup_fixture
    create_local_vllm_source
    if run_build --vllm-source-dir "$LOCAL_VLLM_SOURCE_DIR" --vllm-ref missing-ref; then
        fail "--vllm-source-dir unexpectedly accepted a missing ref"
    fi
    assert_log_not_contains '^docker build'
    assert_output_contains "Error: --vllm-ref 'missing-ref' is not available in the local vLLM checkout\."
    pass "--vllm-source-dir resolves refs without fetching"
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
    assert_log_contains '^docker build --target vllm-export .*--build-arg TORCH_CUDA_ARCH_LIST=12.1a --build-arg FLASHINFER_CUDA_ARCH_LIST=12.1a .*--build-arg TORCH_VERSION=2.13.0 --build-arg TORCHVISION_VERSION=0.28.0 --build-arg TORCHAUDIO_VERSION=2.11.0 --build-arg CUTLASS_DSL_VERSION=4.7.0 .*--build-arg VLLM_REF=dev/karmic-kraken --build-arg VLLM_REPO=https://github.com/local-inference-lab/vllm --build-arg VLLM_APPLY_PRESET_PRS=0 .*--build-arg VLLM_PRESERVE_SM12X_TARGET=1 --build-arg VLLM_PATCH_B12X_C128A_ALIGNMENT=1'
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_REF=main --build-arg FLASHINFER_REPO=https://github.com/local-inference-lab/flashinfer.git --build-arg FLASHINFER_BUILD_CUBIN=0 '
    assert_log_contains '^docker build -t vllm-node-b12x .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/b12x --build-context vllm_wheels=\./\.wheel-cache/vllm/b12x '
    assert_log_not_contains 'B12X_(FROM_PYPI|REPO|REF|CACHEBUST)='
    assert_log_not_contains '^curl .*prebuilt-flashinfer'
    assert_log_not_contains 'Dockerfile\.mxfp4'
    assert_output_contains 'Rebuilding vLLM wheels \(--exp-b12x preset\)\.\.\.'
    pass "--exp-b12x --rebuild-vllm uses the B12X source-build profile"
}

test_exp_b12x_allows_vllm_prs() {
    setup_fixture
    run_build --exp-b12x --apply-vllm-pr 12345 || fail "--exp-b12x with vLLM PR run failed"
    assert_log_contains '^docker build --target vllm-export .*--build-arg VLLM_REF=dev/karmic-kraken --build-arg VLLM_REPO=https://github.com/local-inference-lab/vllm --build-arg VLLM_APPLY_PRESET_PRS=0 --build-arg CACHEBUST_VLLM=[0-9]+ --build-arg VLLM_PRS=12345'
    assert_output_contains 'Rebuilding vLLM wheels \(--exp-b12x preset with requested vLLM PRs\)\.\.\.'
    assert_output_contains 'Applying vLLM PRs: 12345'
    pass "--exp-b12x accepts additional vLLM PR patches"
}

test_b12x_flashinfer_cache_is_independent() {
    setup_fixture
    local regular="$FIXTURE_DIR/.wheel-cache/flashinfer/regular"
    local b12x="$FIXTURE_DIR/.wheel-cache/flashinfer/b12x"
    cp -a "$regular" "$CASE_DIR/regular-before"
    run_build --exp-b12x --rebuild-vllm || fail "B12X fork build failed"
    [ -f "$b12x/flashinfer_python-built.whl" ] || fail "B12X Python wheel missing"
    [ -f "$b12x/flashinfer_jit_cache-built.whl" ] || fail "B12X JIT wheel missing"
    if compgen -G "$b12x/flashinfer_cubin-*.whl" >/dev/null; then
        fail "B12X cache includes a cubin wheel"
    fi
    diff -r "$regular" "$CASE_DIR/regular-before" || fail "B12X build changed upstream wheels"
    : > "$TEST_LOG"
    run_build --exp-b12x --rebuild-vllm || fail "cached B12X build failed"
    assert_log_not_contains '^docker build --target flashinfer-export '
    assert_log_not_contains '^curl .*prebuilt-flashinfer'
    assert_output_contains 'Using cached B12X FlashInfer wheels\.'
    : > "$TEST_LOG"
    run_build --use-wheels || fail "regular build after B12X failed"
    assert_log_contains '^docker build -t vllm-node .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/regular '
    pass "B12X caches cubin-free fork wheels without changing regular wheels"
}

test_b12x_flashinfer_custom_ref_is_isolated() {
    setup_fixture
    run_build --exp-b12x --flashinfer-ref fork-ref --apply-flashinfer-pr 123 || fail "custom fork ref build failed"
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_REF=fork-ref --build-arg FLASHINFER_REPO=https://github.com/local-inference-lab/flashinfer.git --build-arg FLASHINFER_BUILD_CUBIN=0 .*--build-arg FLASHINFER_PRS=123 '
    assert_log_contains '^docker build -t vllm-node-b12x .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/b12x-custom '
    assert_log_not_contains '^curl .*prebuilt-flashinfer'
    pass "custom B12X refs and PRs use the fork and an isolated wheel cache"
}

test_b12x_flashinfer_rejects_upstream_download() {
    setup_fixture
    if run_build --exp-b12x --force-flashinfer-download; then
        fail "B12X accepted upstream FlashInfer downloads"
    fi
    assert_log_not_contains '^docker '
    assert_log_not_contains '^curl '
    assert_output_contains 'Error: B12X FlashInfer wheels are not published'
    pass "B12X rejects unsupported FlashInfer downloads"
}

test_fork_use_wheels_never_implicitly_builds_flashinfer() {
    setup_fixture
    if run_build --use-wheels --vllm-repo https://github.com/local-inference-lab/vllm.git; then
        fail "fork --use-wheels unexpectedly compiled missing FlashInfer wheels"
    fi
    assert_log_not_contains '^docker build'
    assert_log_not_contains '^curl .*prebuilt-flashinfer'
    assert_output_contains 'Error: No complete cached B12X FlashInfer wheels are available\.'
    pass "fork --use-wheels requires cached FlashInfer or an explicit rebuild"
}

test_b12x_flashinfer_export_validates_providers() {
    setup_fixture
    MOCK_FLASHINFER_SPLIT_EXPORT=complete run_build --exp-b12x --rebuild-flashinfer || fail "split B12X export failed"
    local cache="$FIXTURE_DIR/.wheel-cache/flashinfer/b12x"
    cp -a "$cache" "$CASE_DIR/b12x-before"
    : > "$TEST_LOG"
    if MOCK_FLASHINFER_SPLIT_EXPORT=incomplete run_build --exp-b12x --rebuild-flashinfer; then
        fail "B12X export accepted missing JIT providers"
    fi
    diff -r "$cache" "$CASE_DIR/b12x-before" || fail "failed fork export replaced its cache"
    assert_log_not_contains '^docker build -t '
    pass "B12X validates JIT providers and preserves the cache after a failed export"
}

test_cleanup_includes_b12x_flashinfer_profiles() {
    setup_fixture
    touch "$FIXTURE_DIR/.wheel-cache/flashinfer/b12x/flashinfer_python-old.whl" \
        "$FIXTURE_DIR/.wheel-cache/flashinfer/b12x-custom/flashinfer_python-old.whl"
    run_build --cleanup || fail "B12X cache cleanup failed"
    if compgen -G "$FIXTURE_DIR/.wheel-cache/flashinfer/b12x*/*.whl" >/dev/null; then
        fail "cleanup left B12X wheels behind"
    fi
    pass "explicit cleanup covers both B12X FlashInfer profiles"
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
    printf '12.0f\n' > "$FIXTURE_DIR/.wheel-cache/flashinfer/b12x/.flashinfer-arch"
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

test_spark_kv_cache_cleanup_patch_supports_b12x_final_snapshot() {
    local patch_script="$PROJECT_DIR/docker/patch_vllm_spark_kv_cache_cleanup.py"
    local legacy_fixture="$TMP_BASE/kv-cleanup-legacy"
    local b12x_fixture="$TMP_BASE/kv-cleanup-b12x"
    local unknown_fixture="$TMP_BASE/kv-cleanup-unknown"
    local target_rel="vllm/v1/worker/gpu_worker.py"
    local output="$TMP_BASE/kv-cleanup-output.log"

    mkdir -p \
        "$legacy_fixture/vllm/v1/worker" \
        "$b12x_fixture/vllm/v1/worker"
    cat > "$legacy_fixture/$target_rel" <<'PY'
import torch


class Worker:
    def determine_available_memory(self):
        free_gpu_memory = profile_result.after_profile.free_memory
        return free_gpu_memory

    def initialize_from_config(self, kv_cache_config):
        """Allocate the KV cache."""
        self.model_runner.initialize_kv_cache(kv_cache_config)
PY
    cat > "$b12x_fixture/$target_rel" <<'PY'
import torch


class Worker:
    def determine_available_memory(self):
        final_profile_snapshot = MemorySnapshot(device=self.device)
        late_persistent_memory = max(
            profile_result.after_profile.free_memory
            - final_profile_snapshot.free_memory,
            0,
        )
        free_gpu_memory = final_profile_snapshot.free_memory
        return free_gpu_memory - late_persistent_memory

    def initialize_from_config(self, kv_cache_config):
        """Allocate the KV cache."""
        self.model_runner.initialize_kv_cache(kv_cache_config)
PY

    python3 "$patch_script" "$legacy_fixture" > "$output"
    python3 "$patch_script" "$b12x_fixture" >> "$output"
    python3 -m py_compile \
        "$legacy_fixture/$target_rel" \
        "$b12x_fixture/$target_rel"
    python3 - "$legacy_fixture/$target_rel" "$b12x_fixture/$target_rel" <<'PY'
from pathlib import Path
import sys

marker = "# spark-vllm-docker: post-profile cleanup before KV sizing"
prealloc_marker = "# spark-vllm-docker: pre-KV cache allocator cleanup"
legacy = Path(sys.argv[1]).read_text()
b12x = Path(sys.argv[2]).read_text()

assert legacy.count(marker) == 1
assert legacy.count(prealloc_marker) == 1
assert legacy.index(marker) < legacy.index(
    "free_gpu_memory = profile_result.after_profile.free_memory"
)

assert b12x.count(marker) == 1
assert b12x.count(prealloc_marker) == 1
assert b12x.index(marker) < b12x.index("final_profile_snapshot = MemorySnapshot")
assert b12x.index("final_profile_snapshot = MemorySnapshot") < b12x.index(
    "free_gpu_memory = final_profile_snapshot.free_memory"
)

for patched in (legacy, b12x):
    assert patched.count(
        'if hasattr(profile_result, "transient_peak_headroom"):'
    ) == 1
    assert "profile_result.before_create.free_memory" in patched
    assert "- profile_result.after_profile.free_memory" in patched
    assert "profile_result.total_consumed" in patched
    assert "+ profile_result.transient_peak_headroom" in patched
    assert "# Compatibility with older profiling results." in patched
PY

    cp "$b12x_fixture/$target_rel" "$b12x_fixture/gpu_worker.once.py"
    python3 "$patch_script" "$b12x_fixture" >> "$output"
    if ! cmp -s \
        "$b12x_fixture/gpu_worker.once.py" \
        "$b12x_fixture/$target_rel"; then
        fail "Spark KV cache cleanup patch is not idempotent for the B12X source shape"
    fi

    cp -a "$b12x_fixture" "$unknown_fixture"
    sed -i \
        's/final_profile_snapshot = MemorySnapshot(device=self.device)/final_profile_snapshot = capture_memory()/' \
        "$unknown_fixture/$target_rel"
    sed -i '/spark-vllm-docker:/d' "$unknown_fixture/$target_rel"
    sed -i '/profile_result.after_profile.measure()/d' "$unknown_fixture/$target_rel"
    sed -i '/diff_from_create.non_torch_memory/d' "$unknown_fixture/$target_rel"
    if python3 "$patch_script" "$unknown_fixture" >> "$output" 2>&1; then
        fail "Spark KV cache cleanup patch accepted an unknown profiling snapshot"
    fi

    pass "Spark KV cache cleanup supports upstream and B12X final-snapshot source shapes"
}

test_mrv2_speculator_cudagraph_pool_patch_is_guarded_and_idempotent() {
    local patch_script="$PROJECT_DIR/docker/patch_vllm_mrv2_speculator_cudagraph_pool.py"
    local patch_fixture="$TMP_BASE/mrv2-speculator-cudagraph-pool"
    local target_dir="$patch_fixture/vllm/v1/worker/gpu"
    local target="$target_dir/cudagraph_utils.py"
    local output="$patch_fixture/output.log"
    local unknown_fixture="$TMP_BASE/mrv2-speculator-cudagraph-pool-unknown"
    local equivalent_fixture="$TMP_BASE/mrv2-speculator-cudagraph-pool-equivalent"

    mkdir -p "$target_dir"
    cat > "$target" <<'PY'
from typing import Any


class CudaGraphManager:
    pass


def profile_cudagraph_memory(runner):
    """PIECEWISE, encoder and speculator graphs are measured in full."""
    manager = runner.cudagraph_manager
    all_wrappers: list[Any] = []
    original_pools: dict[int, Any] = {}
    try:
        manager.pool = current_platform.graph_pool_handle()
        if manager.use_breakable_cg:
            pass
    finally:
        CUDAGraphWrapper.clear_all_graphs()
        BreakableCUDAGraphWrapper.clear_all_graphs()
        for wrapper in all_wrappers:
            pass
PY
    cp -a "$patch_fixture" "$unknown_fixture"
    sed -i 's/if manager.use_breakable_cg:/if bool(manager.use_breakable_cg):/' \
        "$unknown_fixture/vllm/v1/worker/gpu/cudagraph_utils.py"

    python3 "$patch_script" "$patch_fixture" > "$output"
    for expected in \
        'speculator_manager.pool = manager.pool' \
        'speculator_manager.graphs.clear()' \
        'setattr(runner.speculator, name, None)'; do
        if ! grep -Fq "$expected" "$target"; then
            fail "MRV2 speculator pool patch is missing: $expected"
        fi
    done
    python3 -m py_compile "$target"

    local before after
    before=$(sha256sum "$target")
    python3 "$patch_script" "$patch_fixture" >> "$output"
    after=$(sha256sum "$target")
    if [ "$before" != "$after" ]; then
        fail "MRV2 speculator pool patch is not idempotent"
    fi
    if ! grep -Fq \
        'Equivalent MRV2 speculator CUDA-graph pool fix is present; skipping' \
        "$output"; then
        fail "MRV2 speculator pool patch did not report its idempotent skip"
    fi

    local equivalent_target="$equivalent_fixture/vllm/v1/worker/gpu/cudagraph_utils.py"
    mkdir -p "$(dirname "$equivalent_target")"
    cat > "$equivalent_target" <<'PY'
from typing import Any


class CudaGraphManager:
    pass


def _profiling_cudagraph_managers(runner) -> list[CudaGraphManager]:
    managers = [runner.cudagraph_manager]
    speculator = runner.speculator
    if speculator is not None:
        for name in ("prefill_cudagraph_manager", "decode_cudagraph_manager"):
            candidate = getattr(speculator, name, None)
            if isinstance(candidate, CudaGraphManager):
                managers.append(candidate)
    return managers


def profile_cudagraph_memory(runner):
    """PIECEWISE, encoder and speculator graphs are measured in full."""
    manager = runner.cudagraph_manager
    all_wrappers: list[Any] = []
    original_pools: dict[int, Any] = {}
    graph_managers = _profiling_cudagraph_managers(runner)
    original_manager_pools = {
        id(graph_manager): graph_manager.pool for graph_manager in graph_managers
    }
    try:
        manager.pool = current_platform.graph_pool_handle()
        for graph_manager in graph_managers:
            graph_manager.pool = manager.pool
        if manager.use_breakable_cg:
            pass
    finally:
        CUDAGraphWrapper.clear_all_graphs()
        BreakableCUDAGraphWrapper.clear_all_graphs()
        for graph_manager in graph_managers:
            graph_manager.graphs.clear()
            graph_manager.pool = original_manager_pools[id(graph_manager)]
        for wrapper in all_wrappers:
            pass
PY
    local equivalent_before equivalent_after
    equivalent_before=$(sha256sum "$equivalent_target")
    python3 "$patch_script" "$equivalent_fixture" >> "$output"
    equivalent_after=$(sha256sum "$equivalent_target")
    if [ "$equivalent_before" != "$equivalent_after" ]; then
        fail "MRV2 speculator pool patch modified an equivalent manager-collection fix"
    fi
    if ! tail -n 1 "$output" | grep -Fq \
        'Equivalent MRV2 speculator CUDA-graph pool fix is present; skipping'; then
        fail "MRV2 speculator pool patch did not recognize the manager-collection fix"
    fi

    local unknown_target="$unknown_fixture/vllm/v1/worker/gpu/cudagraph_utils.py"
    local unknown_before unknown_after
    unknown_before=$(sha256sum "$unknown_target")
    if python3 "$patch_script" "$unknown_fixture" >> "$output" 2>&1; then
        fail "MRV2 speculator pool patch accepted an unknown vulnerable layout"
    fi
    unknown_after=$(sha256sum "$unknown_target")
    if [ "$unknown_before" != "$unknown_after" ]; then
        fail "MRV2 speculator pool patch partially modified an unknown layout"
    fi
    pass "MRV2 speculator CUDA-graph pool workaround is guarded and idempotent"
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
    assert_log_contains '^docker build -t vllm-node .*--build-arg TORCH_VERSION=2.12.0 --build-arg TORCHVISION_VERSION=0.27.0 --build-arg TORCHAUDIO_VERSION=none .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/b12x '
    assert_log_not_contains 'B12X_(FROM_PYPI|REPO|REF|CACHEBUST)='
    pass "Torch versions and fork FlashInfer wheels are forwarded to the fork build"
}

test_local_inference_lab_b12x_applies_to_any_ref() {
    setup_fixture
    run_build \
        --vllm-repo https://github.com/local-inference-lab/vllm \
        --vllm-ref dev/spark-fixes-7-14 \
        --torch-version 2.12.0 || fail "local-inference-lab alternate ref run failed"
    assert_log_contains '^docker build --target flashinfer-export .*--build-arg FLASHINFER_REPO=https://github.com/local-inference-lab/flashinfer.git --build-arg FLASHINFER_BUILD_CUBIN=0 '
    assert_log_contains '^docker build -t vllm-node .*--build-arg TORCH_VERSION=2.12.0 .*--build-context flashinfer_wheels=\./\.wheel-cache/flashinfer/b12x '
    pass "all local-inference-lab/vllm refs use the fork's FlashInfer wheels"
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

test_dockerfile_accepts_local_vllm_context() {
    for expected in \
        'FROM scratch AS vllm_source' \
        '--mount=type=bind,from=vllm_source,target=/tmp/vllm-local-source' \
        'if [ "$VLLM_SOURCE_MODE" = "local" ]' \
        'if [ "$(git rev-parse HEAD)" != "$VLLM_SOURCE_COMMIT" ]' \
        'git remote remove origin 2>/dev/null || true'; do
        if ! grep -Fq -- "$expected" "$PROJECT_DIR/Dockerfile"; then
            fail "Dockerfile local vLLM source block is missing: $expected"
        fi
    done
    pass "Dockerfile consumes a verified local vLLM named context"
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
        '"$CUTLASS_DSL_VERSION" --expected-count 1 requirements/cuda.txt'; do
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

test_dockerfile_omits_standalone_b12x() {
    if grep -Eq 'B12X_(REPO|REF|CACHEBUST|FROM_PYPI)|b12x-source|refresh-package b12x|patch_b12x_cache_integrity.py' "$PROJECT_DIR/Dockerfile"; then
        fail "Dockerfile still installs or patches standalone B12X"
    fi
    pass "Dockerfile relies on FlashInfer for B12X"
}

test_build_dependency_updates() {
    if ! python3 "$PROJECT_DIR/tests/test_build_dependency_updates.py"; then
        fail "FlashInfer dependency build regression tests failed"
    fi
    pass "FlashInfer wheels build with the selected settings"
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
    if ! sed -n '/if \[ ! -d "flashinfer" \]/,/WORKDIR \/workspace\/flashinfer/p' "$PROJECT_DIR/Dockerfile" | grep -Fq 'git reset --hard HEAD'; then
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
        'git fetch vllm-upstream "+pull/${pr}/head:${pr_head}"' \
        'git merge-base vllm-upstream/main "$pr_head"' \
        'curl -fsSL --retry 3 --retry-delay 1 "${pr_url}.diff" -o "$patch_file"'; do
        if ! grep -Fq "$expected" "$vllm_pr_block"; then
            fail "vLLM PR block is missing reference-specific fetch logic: $expected"
        fi
    done
    pass "vLLM PR patches use upstream for numbers and the named repository for URLs"
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
    if [ "$patch_count" -ne 17 ]; then
        fail "Expected 17 external vLLM patch scripts, found $patch_count"
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

test_api_key_auth_patch() {
    if ! python3 "$PROJECT_DIR/tests/test_vllm_api_key_auth_patch.py"; then
        fail "API-key authentication patch regression tests failed"
    fi
    pass "API-key authentication patch protects routes in regular and B12X source builds"
}

test_swa_block_size_patch() {
    if ! python3 "$PROJECT_DIR/tests/test_vllm_swa_block_size_patch.py"; then
        fail "SWA block fallback regression tests failed"
    fi
    local runner_block="$TMP_BASE/swa-runner-block"
    sed -n '/^FROM .* AS runner/,$p' "$PROJECT_DIR/Dockerfile" > "$runner_block"
    if grep -Fq 'patch_vllm_swa_block_size.py' "$runner_block"; then
        fail "SWA fix must be applied during the vLLM source build, not in the runner"
    fi
    pass "SWA block fallback preserves supported primary sizes and is applied only at source build"
}

test_torch_schema_enumeration_patch() {
    if ! python3 "$PROJECT_DIR/tests/test_torch_schema_enumeration_patch.py"; then
        fail "Torch schema enumeration regression tests failed"
    fi
    pass "Torch schema enumeration preserves defaults and patches the installed runner"
}

test_instanttensor_vllm_memory_patch() {
    if ! python3 "$PROJECT_DIR/tests/test_instanttensor_vllm_memory_patch.py"; then
        fail "InstantTensor memory accounting regression tests failed"
    fi
    pass "InstantTensor uses vLLM memory accounting and preserves budget checks"
}

test_b12x_moe_tuning_memory_patch() {
    if ! python3 "$PROJECT_DIR/tests/test_vllm_b12x_moe_tuning_memory_patch.py"; then
        fail "B12X MoE trial-buffer lifetime regression tests failed"
    fi
    pass "B12X MoE releases trial buffers before KV cache profiling"
}

test_startup_heap_trim_patch() {
    if ! python3 "$PROJECT_DIR/tests/test_vllm_startup_heap_trim_patch.py"; then
        fail "Startup CPU heap trim regression tests failed"
    fi
    pass "Startup CPU heap trim preserves GC and skips unsupported allocators"
}

test_b12x_cache_integrity_patch() {
    if ! python3 "$PROJECT_DIR/tests/test_b12x_cache_integrity_patch.py"; then
        fail "B12X cache integrity regression tests failed"
    fi
    pass "B12X validates cached objects and durably publishes replacement kernels"
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
test_missing_flashinfer_provider_stops_before_vllm_build
test_complete_flashinfer_provider_cache_builds_runner
test_incomplete_flashinfer_export_preserves_cache
test_complete_flashinfer_export_builds_runner
test_flashinfer_provider_validation
test_flashinfer_release_download_includes_device_provider
test_monolithic_flashinfer_release_download_still_works
test_incomplete_flashinfer_download_preserves_cache
test_incomplete_flashinfer_download_without_cache_fails
test_repaired_release_bypasses_incomplete_cache_shortcuts
test_valid_newer_flashinfer_cache_skips_download
test_failed_flashinfer_asset_download_restores_cache
test_regular_build_omits_b12x_package
test_use_wheels_never_falls_back_to_source
test_use_wheels_never_builds_missing_vllm_implicitly
test_use_wheels_builds_only_explicit_source_target
test_use_wheels_builds_only_explicit_flashinfer_target
test_cleanup_stays_prebuilt
test_prebuilt_copy_parallel
test_copy_skips_matching_remote_image
test_copy_only_updates_missing_or_different_hosts
test_copy_skips_matching_content_across_stores
test_copy_mixed_stores_only_loads_missing_hosts
test_copy_detects_changed_runtime_config
test_copy_does_not_skip_unreadable_content
test_copy_missing_local_image_fails
test_no_build_skips_prebuilt
test_build_only_flags_warn_on_prebuilt
test_flashinfer_ref_forwards_selected_ref
test_requested_flashinfer_prs_apply_to_selected_ref
test_rebuild_vllm_applies_preset_prs_by_default
test_vllm_ref_skips_preset_prs_by_default
test_apply_vllm_pr_skips_preset_prs_by_default
test_apply_vllm_pr_url_is_forwarded_to_source_build
test_apply_vllm_pr_rejects_invalid_reference
test_apply_vllm_pr_can_apply_preset_prs_explicitly
test_vllm_ref_can_apply_preset_prs_explicitly
test_apply_preset_prs_forces_vllm_rebuild
test_requested_vllm_prs_apply_to_selected_vllm_ref
test_custom_vllm_repo_forces_source_build
test_local_vllm_source_builds_selected_ref
test_local_vllm_source_defaults_to_head
test_local_vllm_source_rejects_conflicting_repo
test_local_vllm_source_rejects_dirty_checkout
test_local_vllm_source_rejects_missing_ref
test_exp_b12x_uses_prebuilt_image
test_exp_b12x_rebuild_vllm_uses_preset_source_build
test_exp_b12x_allows_vllm_prs
test_b12x_flashinfer_cache_is_independent
test_b12x_flashinfer_custom_ref_is_isolated
test_b12x_flashinfer_rejects_upstream_download
test_fork_use_wheels_never_implicitly_builds_flashinfer
test_b12x_flashinfer_export_validates_providers
test_cleanup_includes_b12x_flashinfer_profiles
test_exp_b12x_respects_custom_tag
test_exp_b12x_rejects_use_wheels
test_exp_b12x_rejects_preset_overrides
test_exp_b12x_preserves_blackwell_arches
test_exp_b12x_rebuilds_mismatched_cached_flashinfer_arch
test_exp_b12x_rebuilds_mismatched_cached_vllm_arch
test_b12x_c128a_alignment_patch_is_guarded_and_idempotent
test_spark_kv_cache_cleanup_patch_supports_b12x_final_snapshot
test_mrv2_speculator_cudagraph_pool_patch_is_guarded_and_idempotent
test_dockerfile_preserves_selected_blackwell_target
test_custom_torch_versions_are_forwarded
test_local_inference_lab_b12x_applies_to_any_ref
test_local_inference_lab_b12x_requires_torch_212
test_dockerfile_custom_repo_bypasses_shared_cache
test_dockerfile_accepts_local_vllm_context
test_dockerfile_uses_configurable_torch_versions
test_dockerfile_pins_cutlass_dsl_47_everywhere
test_dockerfile_uses_profiled_named_wheel_contexts
test_dockerfile_omits_standalone_b12x
test_build_dependency_updates
test_copied_vllm_git_index_is_refreshed_before_patch_apply
test_dockerfile_applies_flashinfer_prs_without_merging_branch_history
test_dockerfile_uses_prepared_python_for_flashinfer_builds
test_dockerfiles_pin_tvm_ffi_regression_version
test_dockerfile_fetches_vllm_prs_from_upstream
test_dockerfile_externalizes_vllm_source_patches
test_api_key_auth_patch
test_swa_block_size_patch
test_torch_schema_enumeration_patch
test_instanttensor_vllm_memory_patch
test_b12x_moe_tuning_memory_patch
test_startup_heap_trim_patch
test_b12x_cache_integrity_patch

echo "Passed $TESTS_PASSED build-and-copy tests."
