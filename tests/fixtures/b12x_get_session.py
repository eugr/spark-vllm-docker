# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# get_b12x_session from local-inference-lab/vllm@8244d1c44348ae8a5bbfc9f1b607021652283ed4.

def get_b12x_session(worker: Worker):
    """Return the worker's preparation session, creating it on first use."""
    session = getattr(worker, "_b12x_session", None)
    if session is not None and session.state != "CLOSED":
        return session
    from b12x.preparation import PreparationSession

    config = worker.vllm_config
    spec = config.speculative_config
    namespace = {
        "model": str(config.model_config.model),
        "dtype": str(config.model_config.dtype),
        "kv_cache_dtype": str(config.cache_config.cache_dtype),
        "tensor_parallel": int(config.parallel_config.tensor_parallel_size),
        "pipeline_parallel": int(config.parallel_config.pipeline_parallel_size),
        "decode_context_parallel": int(
            getattr(config.parallel_config, "decode_context_parallel_size", 1) or 1
        ),
        "draft_model": None if spec is None else str(getattr(spec, "model", None)),
        "speculative_method": None
        if spec is None
        else str(getattr(spec, "method", None)),
    }
    session = PreparationSession(
        device=worker.device,
        autotune=bool(config.kernel_config.enable_b12x_autotune),
        namespace=namespace,
    )
    if session.autotune and os.environ.get("B12X_AUTOTUNE", "1") != "0":
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        ranks = tuple(sorted(int(rank) for rank in tp_group.ranks))
        session.configure_tuning_shard(int(worker.rank), ranks)
    worker._b12x_session = session
    return session
