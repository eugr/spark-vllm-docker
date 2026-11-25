# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Upstream excerpt from 5c8ae7c74aaa99d1f7fbeea942fd7282e3413810:
# vllm/model_executor/warmup/kernel_warmup.py

def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    # Tune the widest bucket set first so bounded passes reuse its configs.
    token_counts = [max_tokens]
    token_counts.extend(_flashinfer_deferred_moe_token_counts(runner))
    return tuple(dict.fromkeys(token_counts))


def _run_flashinfer_autotune_dummy_runs(
    runner: "GPUModelRunner", *, skip_attn: bool = False
) -> None:
    import vllm.utils.flashinfer as fi_utils
    from vllm.models.deepseek_v41.attention import autotune_indexer_wq_b

    dummy_run_kwargs = {"skip_attn": True} if skip_attn else {}
    for num_tokens in _flashinfer_autotune_token_counts(runner):
        # The drafter may use more tokens than this pass in the same dummy run.
        # Include its max M when building the autotune buckets.
        max_tuning_tokens = num_tokens
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            num_query_per_req = speculator.num_query_per_req
            num_draft_reqs = min(
                num_tokens,
                runner.max_num_reqs,
                runner.max_num_tokens // num_query_per_req,
            )
            max_tuning_tokens = max(
                max_tuning_tokens, num_draft_reqs * num_query_per_req
            )

        tuning_buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(
            max_tuning_tokens
        )
        logger.info(
            "Running FlashInfer autotune with %d tokens and token buckets %s.",
            num_tokens,
            tuning_buckets,
        )
        # Round M up to match serving-time bucket selection for non-bucket M.
        # See https://github.com/flashinfer-ai/flashinfer/issues/5450
        with fi_utils.autotune(tuning_buckets=tuning_buckets, round_up=True):
            runner._dummy_run(
                num_tokens=num_tokens,
                skip_eplb=True,
                is_profile=True,
                randomize_inputs=True,
                **dummy_run_kwargs,
            )
            autotune_indexer_wq_b(runner.get_model(), max_tuning_tokens)


