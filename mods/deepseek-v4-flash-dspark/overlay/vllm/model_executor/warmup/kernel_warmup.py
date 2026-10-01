# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.compilation.caching import aot_compile_hash_factors
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.deepseek_v4_mhc_warmup import (
    deepseek_v4_mhc_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_DEEPSEEK_V4_SPARSE_MLA_BACKENDS = frozenset(
    {
        "FLASHMLA_SPARSE",
        "DEEPSEEK_SPARSE_SWA",
    }
)

_DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS = 16
_DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS = 8192
_DEEPSEEK_V4_DSPARK_DECODE_AUTOTUNE_SEQ_LENS = (512, 2048)
_DEEPSEEK_V4_DSPARK_SHORT_PREFILL_WARMUP_TOKENS = (12, 16, 20, 32)
_DEEPSEEK_V4_DSPARK_ROUTE_PACK_PREFILL_TOKENS = (
    # The single-stream coding benchmark targets a 512-token prompt, but the
    # chat template makes the served prompt land near 415 tokens. B12X route-pack
    # kernels specialize on exact packed-route workspace sizes, not only the
    # next power-of-two capacity, so warm the observed prompt neighborhood too.
    411,
    415,
    416,
    512,
    513,
    1024,
)

# Fan of num_tokens specializations to pre-JIT for
# `_compute_slot_mapping_kernel`. On SM12x cold JIT can emit
# non-deterministic codegen that writes wrong slot_mapping → KV corruption
# → downstream sparse-MLA IMA.
_DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS = tuple(range(1, 17)) + (
    32,
    64,
    128,
    256,
    512,
)


def _attention_backend_name(backend: object) -> str | None:
    get_name = getattr(backend, "get_name", None)
    if get_name is None:
        return None
    try:
        return get_name()
    except NotImplementedError:
        return None


def _has_deepseek_v4_sparse_mla_backend(runner: "GPUModelRunner") -> bool:
    for groups in getattr(runner, "attn_groups", []) or ():
        for group in groups:
            name = _attention_backend_name(getattr(group, "backend", None))
            if name in _DEEPSEEK_V4_SPARSE_MLA_BACKENDS:
                return True
    return False


def _clamp_warmup_tokens(num_tokens: int, max_tokens: int) -> int:
    return max(0, min(num_tokens, max_tokens))


def _runner_max_num_tokens(runner: "GPUModelRunner") -> int:
    max_num_tokens = getattr(runner, "max_num_tokens", None)
    if max_num_tokens is not None:
        return int(max_num_tokens)

    scheduler_config = getattr(runner, "scheduler_config", None)
    max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", 1)
    return int(max_num_batched_tokens)


def _runner_vocab_size(runner: "GPUModelRunner") -> int:
    vocab_size = getattr(runner, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)

    model_config = getattr(runner, "model_config", None)
    get_vocab_size = getattr(model_config, "get_vocab_size", None)
    if get_vocab_size is not None:
        return int(get_vocab_size())

    input_batch = getattr(runner, "input_batch", None)
    vocab_size = getattr(input_batch, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)

    return 1


def _deepseek_v4_hf_config(worker: "Worker") -> object | None:
    model_config = getattr(worker.model_runner, "model_config", None)
    return getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )


def _dspark_spec_decode_query_len(worker: "Worker") -> int | None:
    spec_config = getattr(worker.vllm_config, "speculative_config", None)
    if spec_config is None:
        return None
    is_dspark = getattr(spec_config, "is_dspark", None)
    if is_dspark is not None:
        if not is_dspark():
            return None
    elif getattr(spec_config, "method", None) != "dspark":
        return None

    num_spec_tokens = getattr(spec_config, "num_speculative_tokens", None)
    if num_spec_tokens is None:
        return None
    query_len = int(num_spec_tokens) + 1
    if query_len <= 1:
        return None
    return query_len


def _dspark_uniform_decode_autotune_kwargs(
    worker: "Worker",
) -> list[dict[str, object]]:
    query_len = _dspark_spec_decode_query_len(worker)
    if query_len is None:
        return []
    max_tokens = _runner_max_num_tokens(worker.model_runner)
    if query_len > max_tokens:
        return []

    max_model_len = int(getattr(worker.model_runner, "max_model_len", 0) or 0)
    seq_lens = [
        seq_len
        for seq_len in _DEEPSEEK_V4_DSPARK_DECODE_AUTOTUNE_SEQ_LENS
        if max_model_len <= 0 or seq_len <= max_model_len
    ]
    if not seq_lens:
        seq_lens = [query_len]

    return [
        dict(
            num_tokens=query_len,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            uniform_decode=True,
            profile_seq_lens=seq_len,
        )
        for seq_len in seq_lens
    ]


def _dspark_route_pack_token_counts(worker: "Worker") -> tuple[int, ...]:
    query_len = _dspark_spec_decode_query_len(worker)
    if query_len is None:
        return ()

    max_tokens = _runner_max_num_tokens(worker.model_runner)
    token_counts = [query_len]
    token_counts.extend(_DEEPSEEK_V4_DSPARK_SHORT_PREFILL_WARMUP_TOKENS)
    token_counts.extend(_DEEPSEEK_V4_DSPARK_ROUTE_PACK_PREFILL_TOKENS)
    return tuple(
        sorted(
            {token_count for token_count in token_counts if token_count <= max_tokens}
        )
    )


def _dspark_warmup_request_counts(worker: "Worker") -> tuple[int, ...]:
    max_num_seqs = max(1, int(worker.scheduler_config.max_num_seqs))
    return tuple(sorted({1, min(max_num_seqs, 4)}))


@torch.inference_mode()
def _deepseek_v4_b12x_route_pack_warmup(worker: "Worker") -> None:
    """Pre-JIT B12X/FlashInfer W4A16 MoE route-packing kernels.

    DSpark's first live request is usually a short prompt plus speculative
    decode width 6. Those shapes are too small to be covered reliably by the
    broader DeepGEMM warmup, but they still hit Triton route-pack prefix kernels.
    """
    token_counts = _dspark_route_pack_token_counts(worker)
    if not token_counts:
        return

    hf_config = _deepseek_v4_hf_config(worker)
    num_experts = int(getattr(hf_config, "n_routed_experts", 0) or 0)
    top_k = int(getattr(hf_config, "num_experts_per_tok", 0) or 0)
    if num_experts <= 0 or top_k <= 0:
        return

    try:
        from b12x.moe.fused.w4a16.host import (
            max_packed_route_slots,
            select_route_block_size_m,
        )
        from b12x.moe.fused.w4a16.kernel import pack_topk_routes_by_expert
    except ImportError:
        logger.debug("Skipping B12X route-pack warmup: package is unavailable.")
        return

    device = worker.model_runner.device
    for token_count in token_counts:
        block_size_m = select_route_block_size_m(token_count, top_k, num_experts)
        topk_ids = torch.zeros(
            (token_count, top_k), dtype=torch.int32, device=device
        )
        route_id_shapes = (topk_ids, topk_ids.view(-1))
        for route_ids in route_id_shapes:
            pack_topk_routes_by_expert(route_ids, block_size_m, num_experts)

            routed_rows = int(route_ids.numel())
            route_slots = max(
                1,
                max_packed_route_slots(routed_rows, block_size_m, num_experts),
            )
            route_blocks = max(1, (route_slots + block_size_m - 1) // block_size_m)
            pack_topk_routes_by_expert(
                route_ids,
                block_size_m,
                num_experts,
                packed_route_indices=torch.empty(
                    (route_slots,), dtype=torch.int32, device=device
                ),
                block_expert_ids=torch.empty(
                    (route_blocks,), dtype=torch.int32, device=device
                ),
                packed_route_count=torch.empty(1, dtype=torch.int32, device=device),
                expert_offsets=torch.empty(
                    (num_experts + 1,), dtype=torch.int32, device=device
                ),
            )


@torch.inference_mode()
def _deepseek_v4_spec_decode_padded_kernel_warmup(worker: "Worker") -> None:
    """Pre-JIT padded speculative decode input-prep kernels.

    DSpark uses the padded speculative path with query length
    `1 + num_speculative_tokens`; the first real request otherwise pays Triton
    JIT for these small kernels.
    """
    query_len = _dspark_spec_decode_query_len(worker)
    if query_len is None:
        return

    runner = worker.model_runner
    device = runner.device
    vocab_size = _runner_vocab_size(runner)
    next_token_kernel, inputs_kernel = _spec_decode_padded_warmup_kernels()

    for num_reqs in _dspark_warmup_request_counts(worker):
        discard_buffer = getattr(
            getattr(runner, "discard_request_mask", None), "gpu", None
        )
        if discard_buffer is not None and discard_buffer.numel() >= num_reqs:
            discard_request_mask = discard_buffer[:num_reqs]
            discard_request_mask.zero_()
        else:
            discard_request_mask = torch.zeros(
                num_reqs, dtype=torch.bool, device=device
            )

        drafter = getattr(runner, "drafter", None)
        backup_buffer = getattr(
            getattr(drafter, "backup_next_token_ids", None), "gpu", None
        )
        if backup_buffer is not None and backup_buffer.numel() >= num_reqs:
            backup_tokens = backup_buffer[:num_reqs]
            backup_tokens.zero_()
        else:
            backup_tokens = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        valid_sampled_tokens_count = None
        for sample_width in range(1, query_len + 1):
            sampled_token_ids = torch.zeros(
                (num_reqs, sample_width), dtype=torch.int32, device=device
            )
            if sample_width > 1:
                sampled_token_ids[:, -1] = -1

            next_token_ids = torch.empty(num_reqs, dtype=torch.int32, device=device)
            valid_sampled_tokens_count = torch.empty(
                num_reqs, dtype=torch.int32, device=device
            )
            block_size_tokens = 1 << (sample_width - 1).bit_length()

            next_token_kernel[(num_reqs,)](
                sampled_token_ids,
                discard_request_mask,
                backup_tokens,
                next_token_ids,
                valid_sampled_tokens_count,
                vocab_size,
                sample_width,
                num_reqs,
                sampled_token_ids.stride(0),
                BLOCK_SIZE_TOKENS=block_size_tokens,
            )

        assert valid_sampled_tokens_count is not None

        cu_num_draft_tokens = torch.arange(
            query_len - 1,
            (query_len - 1) * num_reqs + 1,
            query_len - 1,
            dtype=torch.int32,
            device=device,
        )
        query_start_loc = torch.arange(
            0,
            query_len * num_reqs + 1,
            query_len,
            dtype=torch.int32,
            device=device,
        )
        token_indices_to_sample = torch.empty(
            num_reqs, dtype=torch.int32, device=device
        )
        num_rejected_tokens = torch.empty(num_reqs, dtype=torch.int32, device=device)

        inputs_kernel[(num_reqs,)](
            cu_num_draft_tokens,
            valid_sampled_tokens_count,
            query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens,
            num_reqs,
        )


@torch.inference_mode()
def _deepseek_v4_rejection_sampler_warmup(worker: "Worker") -> None:
    """Pre-JIT greedy rejection sampling for the DSpark draft width."""
    query_len = _dspark_spec_decode_query_len(worker)
    if query_len is None:
        return

    num_draft_tokens = query_len - 1
    if num_draft_tokens <= 0:
        return

    from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel

    device = worker.model_runner.device
    for batch_size in _dspark_warmup_request_counts(worker):
        total_draft_tokens = batch_size * num_draft_tokens
        output_token_ids = torch.empty(
            (batch_size, query_len), dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = torch.arange(
            num_draft_tokens,
            total_draft_tokens + 1,
            num_draft_tokens,
            dtype=torch.int32,
            device=device,
        )
        draft_token_ids = torch.zeros(
            total_draft_tokens, dtype=torch.int32, device=device
        )
        target_argmax = torch.zeros(
            total_draft_tokens, dtype=torch.int64, device=device
        )
        bonus_token_ids = torch.zeros((batch_size, 1), dtype=torch.int32, device=device)

        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            None,
            num_draft_tokens,
            None,
            None,
            SYNTHETIC_MODE=False,
        )


def _spec_decode_padded_warmup_kernels():
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
    )

    return eagle_prepare_next_token_padded_kernel, eagle_prepare_inputs_padded_kernel


def _deepseek_v4_slot_mapping_warmup(runner: "GPUModelRunner") -> None:
    """Pre-JIT `_compute_slot_mapping_kernel` across decode-shaped sizes."""
    max_tokens = _runner_max_num_tokens(runner)
    input_batch = getattr(runner, "input_batch", None)
    legacy_block_table = getattr(input_batch, "block_table", None)
    v2_block_tables = getattr(runner, "block_tables", None)
    if legacy_block_table is None and v2_block_tables is None:
        logger.debug("Skipping DeepSeek V4 slot-mapping warmup: no block tables.")
        return

    # Snapshot the runner buffers we mutate so warmup doesn't leak state.
    saved_query_start_loc_np = None
    saved_query_start_loc_gpu = None
    if hasattr(runner, "query_start_loc"):
        saved_query_start_loc_np = runner.query_start_loc.np[:2].copy()
        saved_query_start_loc_gpu = runner.query_start_loc.gpu[:2].clone()

    try:
        for requested_tokens in _DEEPSEEK_V4_SLOT_MAPPING_WARMUP_TOKENS:
            num_tokens = _clamp_warmup_tokens(requested_tokens, max_tokens)
            if num_tokens <= 0:
                continue

            positions_source = torch.arange(
                num_tokens, dtype=torch.int64, device=runner.device
            )
            if hasattr(runner, "query_start_loc"):
                runner.query_start_loc.np[0] = 0
                runner.query_start_loc.np[1] = num_tokens
                runner.query_start_loc.copy_to_gpu(2)
                query_start_loc = runner.query_start_loc.gpu[:2]
            else:
                query_start_loc = torch.tensor(
                    [0, num_tokens], dtype=torch.int32, device=runner.device
                )

            if hasattr(runner, "positions"):
                saved_positions = runner.positions[:num_tokens].clone()
                runner.positions[:num_tokens].copy_(positions_source)
                positions = runner.positions[:num_tokens]
            else:
                saved_positions = None
                positions = positions_source

            try:
                if legacy_block_table is not None:
                    legacy_block_table.commit_block_table(1)
                    legacy_block_table.compute_slot_mapping(
                        1, query_start_loc, positions
                    )
                else:
                    idx_mapping = torch.zeros(
                        1, dtype=torch.int32, device=runner.device
                    )
                    assert v2_block_tables is not None
                    v2_block_tables.compute_slot_mappings(
                        idx_mapping,
                        query_start_loc,
                        positions,
                        num_tokens_padded=num_tokens,
                    )
            finally:
                if saved_positions is not None:
                    runner.positions[:num_tokens].copy_(saved_positions)
    finally:
        if saved_query_start_loc_np is not None:
            runner.query_start_loc.np[:2] = saved_query_start_loc_np
            assert saved_query_start_loc_gpu is not None
            runner.query_start_loc.gpu[:2].copy_(saved_query_start_loc_gpu)


@torch.inference_mode()
def _deepseek_v4_request_prep_warmup(worker: "Worker") -> None:
    """Pre-JIT the slot-mapping kernel before the first real request."""
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return
    if not current_platform.is_cuda_alike():
        return

    logger.info("Warming up DeepSeek V4 request preparation kernels.")
    _deepseek_v4_slot_mapping_warmup(runner)
    _deepseek_v4_b12x_route_pack_warmup(worker)
    _deepseek_v4_spec_decode_padded_kernel_warmup(worker)
    _deepseek_v4_rejection_sampler_warmup(worker)
    torch.accelerator.synchronize()


def _deepseek_v4_sparse_mla_decode_autotune(
    worker: "Worker",
    num_tokens: int,
) -> bool:
    """Autotune FlashInfer's DSv4 SM120 sparse-MLA decode path.

    Returns True when this function consumed the mixed attention warmup shape.
    """
    if worker.vllm_config.kernel_config.enable_flashinfer_autotune is not True:
        return False
    if not has_flashinfer() or not current_platform.is_device_capability_family(120):
        return False

    try:
        from flashinfer import sparse_mla_sm120_decode_dsv4_autotune
        from flashinfer.autotuner import AutoTuner
    except ImportError:
        logger.warning(
            "Skipping DeepSeek V4 sparse MLA decode autotune because this "
            "FlashInfer build does not expose sparse_mla_sm120_decode_dsv4_autotune."
        )
        return False

    from vllm.distributed.parallel_state import get_world_group

    runner = worker.model_runner
    world = get_world_group()
    is_leader = world.rank_in_group == 0
    cache_path = _resolve_flashinfer_autotune_file(runner)

    dummy_run_kwargs: list[dict[str, object]] = [
        dict(
            num_tokens=num_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
    ]
    dummy_run_kwargs.extend(_dspark_uniform_decode_autotune_kwargs(worker))

    if is_leader and len(dummy_run_kwargs) > 1:
        logger.info(
            "Including %d DSpark uniform-decode sparse MLA autotune shapes.",
            len(dummy_run_kwargs) - 1,
        )

    def run_autotune_shapes() -> None:
        for kwargs in dummy_run_kwargs:
            runner._dummy_run(**kwargs)

    if is_leader:
        logger.info(
            "Autotuning DeepSeek V4 SM120 sparse MLA decode with FlashInfer "
            "cache file: %s",
            cache_path,
        )

    with torch.inference_mode():
        if is_leader:
            with sparse_mla_sm120_decode_dsv4_autotune(cache_path=str(cache_path)):
                run_autotune_shapes()
        else:
            run_autotune_shapes()

    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)
    if tune_results is None:
        logger.warning(
            "No DeepSeek V4 sparse MLA decode autotune cache entries found. "
            "Falling back to FlashInfer's default tactic heuristic."
        )
        world.barrier()
        return True

    if not is_leader and world.local_rank == 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(tune_results)
    world.barrier()

    AutoTuner.get().load_configs(str(cache_path))
    logger.info(
        "DeepSeek V4 sparse MLA decode autotune cache loaded on rank %d from %s.",
        world.rank_in_group,
        cache_path,
    )
    return True


def _deepseek_v4_sparse_mla_attention_warmup(worker: "Worker") -> None:
    """Warm sparse-MLA attention shapes via `_dummy_run`.

    Three shapes: mixed prefill+decode, single max-chunk prefill, and a
    second-chunk prefill (prior context) — the last covers
    `_build_prefill_chunk_metadata_kernel`'s alt-shape specialization.
    """
    if not envs.VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not _has_deepseek_v4_sparse_mla_backend(runner):
        return

    max_tokens = worker.scheduler_config.max_num_batched_tokens
    mixed_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens
    )
    prefill_tokens = _clamp_warmup_tokens(
        _DEEPSEEK_V4_SPARSE_MLA_PREFILL_WARMUP_TOKENS, max_tokens
    )
    if mixed_tokens <= 0 and prefill_tokens <= 0:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention "
        "for mixed tokens=%s and prefill tokens=%s.",
        mixed_tokens,
        prefill_tokens,
    )
    if mixed_tokens > 0:
        mixed_warmup_done = _deepseek_v4_sparse_mla_decode_autotune(
            worker, mixed_tokens
        )
        if not mixed_warmup_done:
            runner._dummy_run(
                num_tokens=mixed_tokens,
                skip_eplb=True,
                is_profile=True,
                force_attention=True,
                create_mixed_batch=True,
            )
    if prefill_tokens > 0:
        for short_prefill_tokens in _DEEPSEEK_V4_DSPARK_SHORT_PREFILL_WARMUP_TOKENS:
            short_prefill_tokens = _clamp_warmup_tokens(
                short_prefill_tokens, max_tokens
            )
            if short_prefill_tokens <= 0:
                continue
            runner._dummy_run(
                num_tokens=short_prefill_tokens,
                skip_eplb=True,
                is_profile=True,
                force_attention=True,
                create_single_prefill=True,
            )
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
        )
        # Second-chunk shape: indexer sees prior context, hits the alt
        # specialization of `_build_prefill_chunk_metadata_kernel`.
        runner._dummy_run(
            num_tokens=prefill_tokens,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_single_prefill=True,
            profile_seq_lens=prefill_tokens * 2,
        )


def _flashinfer_autotune_cache_hash(runner: "GPUModelRunner") -> str:
    factors = aot_compile_hash_factors(runner.vllm_config)
    return hashlib.sha256(str(factors).encode()).hexdigest()


def _resolve_flashinfer_autotune_file(runner: "GPUModelRunner") -> Path:
    override_dir = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        from flashinfer.jit import env as flashinfer_jit_env

        flashinfer_workspace = flashinfer_jit_env.FLASHINFER_WORKSPACE_DIR
        root = (
            Path(envs.VLLM_CACHE_ROOT)
            / "flashinfer_autotune_cache"
            / flashinfer_workspace.parent.name
            / flashinfer_workspace.name
        )

    output_dir = root / _flashinfer_autotune_cache_hash(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "autotune_configs.json"


def kernel_warmup(worker: "Worker"):
    # DSv4 mHC TileLang kernels run every decoder layer per token; warm them
    # across token sizes first so the first real request doesn't pay JIT cost.
    # No-op for non-DSv4 models and for the b12x mHC path (gated inside).
    deepseek_v4_mhc_warmup(
        worker.get_model(),
        max_tokens=worker.scheduler_config.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
        ),
    )

    # Run next so input-prep kernels JIT against pristine runner state.
    _deepseek_v4_sparse_mla_attention_warmup(worker)
    _deepseek_v4_request_prep_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        worker.model_runner._dummy_run(
            num_tokens=16,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )


# TODO: remove once FlashInfer upstream fixes the persistent file cache
# to resolve collisions like `use_8x4_sf_layout=True/False`, which causes
# invalid tactics to be chosen
_FLASHINFER_USE_PERSISTENT_CACHE = False


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Tuning is performed only on rank 0. The resulting cache is broadcast
    to every rank so all ranks dispatch the same kernel tactic.
    """
    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import get_world_group

    if not _FLASHINFER_USE_PERSISTENT_CACHE:
        with torch.inference_mode(), fi_utils.autotune():
            runner._dummy_run(
                num_tokens=runner.scheduler_config.max_num_batched_tokens,
                skip_eplb=True,
                is_profile=True,
            )
        get_world_group().barrier()
        return

    world = get_world_group()
    is_leader = world.rank_in_group == 0

    cache_path = _resolve_flashinfer_autotune_file(runner)
    if is_leader:
        logger.info("Using FlashInfer autotune cache file: %s", cache_path)

    # We skip EPLB here since we don't want to record dummy metrics.
    # When autotuning with number of tokens m, flashinfer will autotune
    # operations for all number of tokens up to m, so we only need to
    # run with the max number of tokens.
    dummy_run_kwargs = dict(
        num_tokens=runner.scheduler_config.max_num_batched_tokens,
        skip_eplb=True,
        is_profile=True,
    )

    with torch.inference_mode():
        if is_leader:
            with fi_utils.autotune(tune_mode=True, cache=str(cache_path)):
                runner._dummy_run(**dummy_run_kwargs)
        else:
            runner._dummy_run(**dummy_run_kwargs)

    # Broadcast autotune cache from rank 0 to all other ranks so every
    # rank loads the same set of chosen tactics.
    tune_results: bytes | None = None
    if is_leader and cache_path.exists():
        with open(cache_path, "rb") as f:
            tune_results = f.read()

    tune_results = world.broadcast_object(tune_results, src=0)

    if tune_results is None:
        logger.warning(
            "No FlashInfer autotune cache entries found."
            "Falling back to default tactics."
        )
    else:
        if not is_leader and world.local_rank == 0:
            with open(cache_path, "wb") as f:
                f.write(tune_results)
        world.barrier()
        from flashinfer.autotuner import AutoTuner

        AutoTuner.get().load_configs(str(cache_path))
        logger.info(
            "FlashInfer autotune cache loaded on rank %d from %s.",
            world.rank_in_group,
            cache_path,
        )
