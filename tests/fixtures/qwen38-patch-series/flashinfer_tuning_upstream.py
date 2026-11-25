# Excerpts from flashinfer-ai/flashinfer@431c9687f3bb; no GPU imports.
from __future__ import annotations
from dataclasses import dataclass, replace

@dataclass(kw_only=True, unsafe_hash=True)
class TuningConfig:
    """Configuration for autotuning.

    This class specifies all the tuning configurations for a single tuning process.
    Args:
        dynamic_tensor_specs (Tuple[DynamicTensorSpec]): Specifications for how different tensor dimensions
            should be tuned to optimize performance. Each spec defines:
            - Which input tensor dimension is dynamic
            - How to generate tuning values
            - How to map dimensions to valid values during inference

            Example:
                >>> config = TuningConfig(
                ...     dynamic_tensor_specs=(
                ...         DynamicTensorSpec(
                ...             input_idx=(0,),
                ...             dim_idx=(1,),
                ...             gen_tuning_buckets=(32, 64, 128),
                ...             map_to_tuning_buckets=lambda x: ((x + 31) // 32) * 32
                ...         ),
                ...     )
                ... )
        constraint_specs (Tuple[ConstraintSpec]): Specifications for constraints on tensor dimensions.
            Each spec defines:
            - Which input tensor dimension is constrained
            - How to infer the shape of the dimension based on other dimensions

            Example:
                >>> config = TuningConfig(
                ...     constraint_specs=(
                ...         ConstraintSpec(
                ...             input_idx=1,
                ...             dim_idx=2,
                ...             infer_shape=lambda shapes: shapes[0][0] * 2
                ...         ),
                ...     )
                ... )
        use_cold_l2_cache (bool): Whether to measure every profiled sample
            with a cold L2 cache. A buffer twice the device L2 size is written
            immediately before each sample. The flush is ordered before the
            sample's start event, so its cost is excluded from the reported
            kernel latency.
        use_cuda_graph (bool): Whether to use CUDA graph for the tuning process.
        cuda_graph_profile_replays (int): Number of CUDA graph samples per
            profiling repeat. With ``use_cold_l2_cache=False`` these are
            back-to-back replays and measure sustained execution. With
            ``use_cold_l2_cache=True`` every replay is independently preceded
            by a full L2 flush.
        profiling_repeat (int | None): Per-operation profiling repeat override.
            ``None`` uses the autotuner default. This affects measurement
            precision, not tactic compatibility or persisted cache identity.
        use_cold_l2_graph_replay (bool): Replay the captured graph once untimed
            before hot-L2 timing (cold-L2 timing always does). Requires
            ``use_cuda_graph``.
        cold_chain_launches (int): Launches per cold-L2 graph sample, each on
            its own profile-arena lane; the sample is reported per launch, so
            launch latency no longer dominates microsecond kernels. Only the
            first launch follows the flush. Part of the persisted policy.
        tensor_initializers (Tuple[Tuple[int, TensorInitializer]]): Per-input-index
            initializer closures used to synthesize profiling tensors. Each entry
            pairs an input tensor index with the closure that fills that input.
            Indices not listed fall back to a random-scaled default initializer.

            Example:
                >>> config = TuningConfig(
                ...     tensor_initializers=(
                ...         (0, autotuner_initializer_randn),
                ...         (1, autotuner_initializer_ones),
                ...     )
                ... )
        value_aware_input_indices: Input tensor positions whose contents may vary
            between profiling samples while their shapes and dtypes remain stable.
        profile_arena_input_indices: Input tensor positions assigned graph-stable
            lane views in the reusable A/B profiling arenas. Their contents are
            staged once per profile even when they are not value-aware. Shared
            read-only tensors may be omitted and retain one pointer.
        use_cuda_graph_individual_timing: When using CUDA graphs, measure each
            invocation with its own in-graph event interval and return the
            median latency. The default measures amortized sustained execution.
            Cold-L2 profiling uses a separate measurement path. This
            choice is recorded as tuning provenance because it can change the
            winner; ordinary lookup can still consume that winner.
    """

    dynamic_tensor_specs: tuple[DynamicTensorSpec, ...] = ()
    constraint_specs: tuple[ConstraintSpec, ...] = ()
    tensor_initializers: tuple[tuple[int, TensorInitializer], ...] = ()
    use_cold_l2_cache: bool = False
    use_cuda_graph: bool = False
    cuda_graph_profile_replays: int = 1
    profiling_repeat: int | None = None
    use_cold_l2_graph_replay: bool = False
    cold_chain_launches: int = 1
    value_aware_input_indices: tuple[int, ...] = ()
    profile_arena_input_indices: tuple[int, ...] = ()
    # Optional callback invoked once per profile bucket, after dynamic
    # tensors are synthesized but before the per-tactic profile loop.
    # Receives the full list of tensors and returns a (possibly modified)
    # list. Use this to inject a deterministic, realistic distribution
    # for inputs whose default tensor_initializer would be random
    # (e.g. token_selected_experts in MoE workloads).
    inputs_pre_hook: Callable | None = None
    use_cuda_graph_individual_timing: bool = False

    @property
    def uses_cold_chain(self) -> bool:
        """Whether cold-L2 samples time a multi-launch arena chain."""
        return (
            self.cold_chain_launches > 1
            and self.use_cuda_graph
            and self.use_cold_l2_cache
            and bool(self.profile_arena_input_indices or self.value_aware_input_indices)
        )

    @property
    def l2_cache_policy(self) -> str:
        """L2 measurement policy recorded in persisted profiling provenance."""
        if not self.use_cold_l2_cache:
            return "hot"
        if self.uses_cold_chain:
            return f"cold-chain:{self.cold_chain_launches}"
        return "cold"


class AutoTuner:
    def _apply_tuning_overrides(self, tuning_config: TuningConfig) -> TuningConfig:
        """Return a ``TuningConfig`` with active context overrides applied.

        The result is cached so the same logical override produces the same
        object, keeping the nearest-profile LRU cache effective.
        """
        buckets = self._override_tuning_buckets
        round_up_flag = self._override_round_up
        profile_replays = self._override_cuda_graph_profile_replays

        per_config = self._override_config_cache.get(tuning_config)
        cache_key = (buckets, round_up_flag, profile_replays)
        if per_config is not None and cache_key in per_config:
            return per_config[cache_key]

        new_specs: list[DynamicTensorSpec] = []
        for spec in tuning_config.dynamic_tensor_specs:
            new_gen: tuple[int, ...] | Callable[[int], Iterable[int]]
            new_map: Callable[[int], int]
            if buckets is not None:
                new_gen = tuple(sorted(set(buckets)))
                new_map = make_bucket_mapper(new_gen, round_map=round_up_flag)
            elif round_up_flag:
                if isinstance(spec.gen_tuning_buckets, (list, tuple)):
                    sorted_gen = tuple(sorted(set(spec.gen_tuning_buckets)))
                    new_gen = sorted_gen
                    new_map = make_bucket_mapper(sorted_gen, round_map=True)
                else:
                    # gen_tuning_buckets is a callable — keep it, but build a
                    # mapper that rounds up to power-of-2 and clamps to the
                    # generated bucket set so we never exceed the last bucket.
                    gen_fn = spec.gen_tuning_buckets
                    new_gen = gen_fn

                    def _clamped_po2_mapper(x: int, gen_fn=gen_fn):
                        buckets = tuple(sorted(set(gen_fn(x))))
                        return make_bucket_mapper(buckets, round_map=True)(
                            next_positive_power_of_2(x)
                        )

                    new_map = _clamped_po2_mapper
            else:
                new_specs.append(spec)
                continue

            new_specs.append(
                DynamicTensorSpec(
                    input_idx=spec.input_idx,
                    dim_idx=spec.dim_idx,
                    gen_tuning_buckets=new_gen,
                    map_to_tuning_buckets=new_map,
                )
            )

        new_config = replace(
            tuning_config,
            dynamic_tensor_specs=tuple(new_specs),
            cuda_graph_profile_replays=(
                profile_replays
                if profile_replays is not None
                else tuning_config.cuda_graph_profile_replays
            ),
        )
        self._override_config_cache.setdefault(tuning_config, {})[cache_key] = (
            new_config
        )
        return new_config
