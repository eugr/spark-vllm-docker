# Excerpts from flashinfer-ai/flashinfer@82e93625e87b; no GPU imports.
from __future__ import annotations


class AutoTuner:
    def _profile_single_kernel(
        self,
        runner: TunableRunner,
        inputs: list[torch.Tensor],
        tactic: Any,
        tuning_config: TuningConfig,
        input_tensor_batches: list[list[Any]] | None = None,
        **kwargs,
    ) -> float:
        """Profile a single kernel implementation for performance measurement.

        Args:
            runner (TunableRunner): The runner implementation to profile
            inputs (List[torch.Tensor]): Input tensors for the kernel
            tactic (int): Tactic ID to use for this profiling run
            tuning_config (TuningConfig): Tuning configuration

        Returns:
            Execution time in milliseconds. Cold-L2 profiling returns the
            median isolated-invocation latency (per launch for a cold chain).
            Hot graph profiling can opt
            into individual-call intervals; otherwise hot profiling averages
            back-to-back invocations.

        Note:
            The method performs warmup runs, then measures multiple iterations.
            Stream synchronization and delays are used to ensure accurate
            timing.

            All runner invocations inside this method (warmup + measurement)
            execute under ``_profile_measurement_scope`` so that runners can
            consult ``is_in_profile_measurement()`` to apply consistent
            allocation behavior across tactics.  This is what unbiases the
            autotuner's tactic comparison for runners whose normal
            (non-profiling) path uses preallocated buffers sized for a
            specific tile_size: during the measurement window every tactic
            sees the same per-call allocation overhead, and the autotuner
            picks based on intrinsic kernel time.
        """
        if self._v2_local.replay_report is not None:
            raise RuntimeError("replay diagnostics forbid profiling")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "autotune measurement cannot run inside a CUDA graph "
                "capture: it synchronizes and may capture its own graph "
                "(nested capture is forbidden).  Tune before capture, not "
                "inside it -- close the autotune_v2/autotune context before "
                "the framework captures the model."
            )

        # MeasurementPolicy(timer="cupti") routes to per-iteration GPU-span
        # measurement; "auto"/"events" (and cupti-python unavailable) use
        # the historical CUDA-event window below.  Resolved once here so
        # both the routing decision and pure_profile's delay-kernel skip
        # see the same policy.
        measure_policy = self._effective_measure_policy

        profiling_repeat = self._get_profiling_repeat(tuning_config)

        stream = torch.cuda.current_stream()
        avg_time = float("inf")

        def cold_l2_profile(stream: torch.cuda.Stream, repeat: int) -> float:
            """Measure cold-L2 samples without charging for the L2 flush.

            A cold-chain sample is reported per launch, and chained samples
            are capped at ``_COLD_CHAIN_SAMPLE_BUDGET_MS`` of chain time.
            """
            profile_replays = (
                tuning_config.cuda_graph_profile_replays
                if tuning_config.use_cuda_graph
                else 1
            )
            if profile_replays < 1:
                raise ValueError("cuda_graph_profile_replays must be at least one")

            device = next(
                (
                    tensor.device
                    for tensor in inputs
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda
                ),
                None,
            )
            if device is None:
                raise ValueError("Cold-L2 profiling requires at least one CUDA tensor")

            # Twice L2 matches the standalone benchmark policy and prevents
            # static operands passed through kwargs (for example MoE weights)
            # from remaining resident between samples.
            flush_buffer = torch.empty(
                2 * self._get_l2_cache_size_in_bytes(device.index),
                dtype=torch.int8,
                device=device,
            )
            num_samples = repeat * profile_replays
            graph = torch.cuda.CUDAGraph() if tuning_config.use_cuda_graph else None
            chain = graph is not None and tuning_config.uses_cold_chain

            def _run_once(profile_inputs):
                runner(profile_inputs, tactic=tactic, **kwargs)

            captured_batches = (
                input_tensor_batches if chain else input_tensor_batches[-1:]
            )
            with torch.cuda.stream(stream):
                if graph is not None:
                    with torch.cuda.graph(graph):
                        for profile_inputs in captured_batches:
                            _run_once(profile_inputs)
                    # Exercise the captured graph before collecting samples;
                    # every measured replay below is cold independently.
                    warm_start = torch.cuda.Event(enable_timing=True)
                    warm_end = torch.cuda.Event(enable_timing=True)
                    warm_start.record(stream)
                    graph.replay()
                    warm_end.record(stream)

                stream.synchronize()
                if chain:
                    chain_ms = max(warm_start.elapsed_time(warm_end), 1e-3)
                    num_samples = max(
                        1,
                        min(
                            num_samples,
                            int(self._COLD_CHAIN_SAMPLE_BUDGET_MS // chain_ms),
                        ),
                    )
                starts = [
                    torch.cuda.Event(enable_timing=True) for _ in range(num_samples)
                ]
                ends = [
                    torch.cuda.Event(enable_timing=True) for _ in range(num_samples)
                ]
                delay_kernel_time_usec = (
                    self._CUDA_GRAPH_DELAY_MICRO_SECS
                    if graph is not None
                    else self.stream_delay_micro_secs
                )
                if delay_kernel_time_usec > 0:
                    delay_kernel(delay_kernel_time_usec)

                for sample_idx in range(num_samples):
                    flush_buffer.zero_()
                    starts[sample_idx].record(stream)
                    if graph is not None:
                        graph.replay()
                    else:
                        _run_once(
                            input_tensor_batches[sample_idx % len(input_tensor_batches)]
                        )
                    ends[sample_idx].record(stream)

                # One synchronization after all samples keeps host overhead
                # out of the per-invocation measurements.
                stream.synchronize()

            samples = [
                start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
            ]
            return statistics.median(samples) / len(captured_batches)

        def pure_profile(stream: torch.cuda.Stream, repeat: int) -> float:
            graph = torch.cuda.CUDAGraph()

            if self._use_global_timer:
                start_ts = torch.empty(1, dtype=torch.int64, device="cuda")
                end_ts = torch.empty(1, dtype=torch.int64, device="cuda")

                def record_start():
                    self._record_global_timer(start_ts)

                def record_end():
                    self._record_global_timer(end_ts)

                def elapsed_time():
                    # GPU %globaltimer counts in ns; convert to ms to match the
                    # units of Torch.cuda.Event.elapsed_time()
                    return (end_ts.item() - start_ts.item()) / 1e6
            else:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)

                def record_start():
                    start_evt.record()

                def record_end():
                    end_evt.record()

                def elapsed_time():
                    return start_evt.elapsed_time(end_evt)

            def _run_kernels():
                for r in range(repeat):
                    runner(
                        input_tensor_batches[r % len(input_tensor_batches)],
                        tactic=tactic,
                        **kwargs,
                    )

            with torch.cuda.stream(stream):
                if tuning_config.use_cuda_graph:
                    with torch.cuda.graph(graph):
                        _run_kernels()

                stream.synchronize()
                if (
                    tuning_config.use_cuda_graph
                    and tuning_config.use_cold_l2_graph_replay
                ):
                    # Exclude first-replay initialization from the timed sample.
                    graph.replay()
                    stream.synchronize()

                # Delay the profiled kernel launch to eliminate affects of host time overhead in profiling.
                # The events_no_delay timer skips it: the eager sustained
                # rate, host cost included, is exactly what it measures.
                if not (
                    measure_policy is not None
                    and measure_policy.timer == "events_no_delay"
                ):
                    delay_kernel_time_usec = (
                        self._CUDA_GRAPH_DELAY_MICRO_SECS
                        if tuning_config.use_cuda_graph
                        else self.stream_delay_micro_secs
                    )
                    if delay_kernel_time_usec > 0:
                        delay_kernel(delay_kernel_time_usec)

                if tuning_config.use_cuda_graph:
                    profile_replays = tuning_config.cuda_graph_profile_replays
                    if profile_replays < 1:
                        raise ValueError(
                            "cuda_graph_profile_replays must be at least one"
                        )
                    if profile_replays > 1:
                        for _ in range(profile_replays):
                            graph.replay()
                else:
                    profile_replays = 1

                record_start()

                if tuning_config.use_cuda_graph:
                    for _ in range(profile_replays):
                        graph.replay()
                else:
                    _run_kernels()

                record_end()
                stream.synchronize()

                return elapsed_time() / (repeat * profile_replays)

        # Run the timing under ``_profile_measurement_scope`` (so runners
        # can consult ``is_in_profile_measurement()``), then — if a
        # cross-rank group is set — all-reduce the measured time so every
        # rank's argmin picks the same tactic. Local GPU timing noise
        # otherwise causes per-rank tactic divergence that can deadlock
        # collective allocation paths (e.g. NCCL_WIN_COLL_SYMMETRIC). See
        # ``set_autotune_process_group``.
        #
        # Collective cardinality invariant: every rank MUST reach the
        # all-reduce below exactly once per ``_profile_single_kernel``
        # call. If one rank raises inside the measurement window and exits
        # without reducing while peers are still waiting, the next
        # tactic's reduce deadlocks. We therefore catch failures here,
        # mark this rank's ``avg_time`` as ``inf`` (so the tactic is
        # disqualified everywhere after the SUM), run the reduce
        # unconditionally, and then re-raise so the outer error-handling
        # path in ``choose_one`` still runs (logging, stats, OOM
        # fallback).
        profile_exc: Optional[BaseException] = None
        try:
            cupti_requested = (
                measure_policy is not None and measure_policy.timer == "cupti"
            )
            use_cupti = False
            cupti_error = None
            if cupti_requested:
                available = not self._cupti_disabled and _load_cupti() is not None
                use_cupti = _sync_cupti_status(int(not available)) == 0
            if use_cupti:
                try:
                    avg_time = self._profile_single_kernel_cupti(
                        runner, inputs, tactic, tuning_config, **kwargs
                    )
                except _CuptiInfraError as e:
                    cupti_error = e
                except BaseException as e:  # noqa: BLE001
                    profile_exc = e
                # Missing activity records can fail after execution. Agree on
                # fallback before any rank starts another round of launches.
                status = _sync_cupti_status(
                    2 if profile_exc is not None else int(cupti_error is not None)
                )
                if status == 2:
                    avg_time = float("inf")
                elif status == 1:
                    use_cupti = False
            if cupti_requested and not use_cupti:
                if not self._cupti_disabled:
                    reason = cupti_error or "CUPTI unavailable on a tuning rank"
                    logger.warning(
                        f"[Autotuner]: CUPTI activity tracing failed ({reason}); "
                        f"falling back to CUDA-event timing for the rest of "
                        f"this process.  NOTE: entries tuned in this session "
                        f"are event-timed even though the store identity says "
                        f"timer='cupti' -- re-tune without the conflicting "
                        f"profiler for faithful measurements."
                    )
                self._cupti_disabled = True
            if not use_cupti:
                delayed_events = tuning_config.use_cold_l2_cache or not (
                    (
                        tuning_config.use_cuda_graph
                        and tuning_config.use_cuda_graph_individual_timing
                    )
                    or (
                        measure_policy is not None
                        and measure_policy.timer == "events_no_delay"
                    )
                )
                delay_micro_secs = (
                    self._CUDA_GRAPH_DELAY_MICRO_SECS
                    if tuning_config.use_cuda_graph
                    else self.stream_delay_micro_secs
                )
                if delayed_events and delay_micro_secs > 0:
                    # Loading the delay helper after candidate warmup can skew
                    # the first candidate. Initialize it before warming inputs.
                    get_trtllm_utils_module()
                # Rotating input batches (up to ~3x L2 of clones under
                # cold_l2) are only used by the event path; allocating them
                # before the routing decision would double the transient
                # footprint alongside the CUPTI path's own flush buffer.
                if input_tensor_batches is None:
                    uses_profile_arena = bool(
                        tuning_config.profile_arena_input_indices
                        or tuning_config.value_aware_input_indices
                    )
                    input_tensor_batches = (
                        self._prepare_input_tensors_with_batches(inputs, tuning_config)
                        if uses_profile_arena or not tuning_config.use_cold_l2_cache
                        else [inputs]
                    )
                with _profile_measurement_scope():
                    # warm up, no timing
                    for _ in range(self.warmup):
                        runner(input_tensor_batches[-1], tactic=tactic, **kwargs)

                    if tuning_config.use_cold_l2_cache:
                        avg_time = cold_l2_profile(stream, profiling_repeat)
                    elif (
                        tuning_config.use_cuda_graph
                        and tuning_config.use_cuda_graph_individual_timing
                    ):
                        from ._graph_timing import profile_individual_graph_calls

                        avg_time = profile_individual_graph_calls(
                            lambda index: runner(
                                input_tensor_batches[index % len(input_tensor_batches)],
                                tactic=tactic,
                                **kwargs,
                            ),
                            repeat=profiling_repeat,
                            replays=tuning_config.cuda_graph_profile_replays,
                            record_global_timer=(
                                self._record_global_timer
                                if self._use_global_timer
                                else None
                            ),
                        )
                    else:
                        avg_time = pure_profile(stream, profiling_repeat)
        except BaseException as e:  # noqa: BLE001
            # Catch everything (incl. KeyboardInterrupt / SystemExit): this
            # rank must still reach the all-reduce below or peers already
            # waiting on it deadlock. The original error is re-raised after.
            avg_time = float("inf")
            profile_exc = e

        try:
            if _tune_process_group is not None:
                import torch.distributed as dist

                # NCCL requires a CUDA tensor; a gloo (CPU) subgroup — the
                # recommended choice — uses a CPU tensor.
                backend = str(dist.get_backend(_tune_process_group)).lower()
                device = "cuda" if backend == "nccl" else "cpu"
                time_tensor = torch.tensor(
                    [avg_time], dtype=torch.float64, device=device
                )
                dist.all_reduce(
                    time_tensor, op=dist.ReduceOp.SUM, group=_tune_process_group
                )
                avg_time = time_tensor.item() / dist.get_world_size(_tune_process_group)
        finally:
            # Re-raise even if the collective itself failed, so the original
            # profiling error is never masked by a secondary reduce error.
            if profile_exc is not None:
                raise profile_exc

        shapes = self._get_input_sizes(inputs)
        logger.debug(
            f"[Autotuner]: profiling {runner} {tactic}, shapes={shapes}, avg_time {avg_time}"
        )

        return avg_time

    def _profile_single_kernel_cupti(
        self,
        runner: TunableRunner,
        inputs: list[torch.Tensor],
        tactic: Any,
        tuning_config: TuningConfig,
        **kwargs,
    ) -> float:
        """CUPTI variant of ``_profile_single_kernel``
        (``MeasurementPolicy(timer="cupti")``).

        Measures each iteration's GPU kernel span from CUPTI activity
        records instead of one CUDA-event window around the whole repeat
        loop.  Needs no delay kernel (GPU-side timestamps exclude host
        launch latency structurally) at the cost of a device synchronize
        per iteration.  Semantics: per-call *latency* span — internal gaps
        of a multi-kernel tactic are honestly included, queue-saturation
        effects are not (see ``MeasurementPolicy``).

        Cold-L2 is achieved by zeroing an L2-sized flush buffer once per
        iteration.  ``zero_()`` dispatches a fill KERNEL (not a memset), so
        it is handed to ``_cupti_measure_spans`` as a prologue that runs and
        is synced BEFORE the correlation window opens; running it inside the
        window would fold ~100 us of flush into the tactic's measured span
        and bias selection.  Input-buffer rotation is not usable here
        because the CUDA-graph mode replays a single captured iteration,
        which binds the captured tensors.

        Raises:
            _CuptiInfraError: propagated from ``_cupti_measure_spans`` when
                tracing cannot initialize or returns no kernel records.
                ``_profile_single_kernel`` coordinates CUDA-event fallback
                across the tuning group.
        """

        def check_ready(error):
            if _sync_cupti_status(int(error is not None)):
                raise _CuptiInfraError(
                    error or "CUPTI initialization failed on another tuning rank"
                )

        with _profile_measurement_scope():
            graph = None
            flush_buffer = None

            def prepare():
                nonlocal graph, flush_buffer
                for _ in range(self.warmup):
                    runner(inputs, tactic=tactic, **kwargs)

                if tuning_config.use_cold_l2_cache:
                    device = next(
                        (t.device for t in inputs if isinstance(t, torch.Tensor)),
                        torch.device("cuda"),
                    )
                    flush_buffer = torch.empty(
                        self._get_l2_cache_size_in_bytes() * 3,
                        device=device,
                        dtype=torch.int8,
                    )

                if tuning_config.use_cuda_graph:
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        runner(inputs, tactic=tactic, **kwargs)

            def run_iteration():
                if graph is None:
                    runner(inputs, tactic=tactic, **kwargs)
                else:
                    graph.replay()

            def prologue():
                # zero_ launches a fill kernel; exclude it from the span.
                if flush_buffer is not None:
                    flush_buffer.zero_()

            spans_ms = _cupti_measure_spans(
                run_iteration,
                self.repeat,
                prologue=prologue,
                check_ready=check_ready,
                prepare=prepare,
            )

        # Median, not mean: robust to first-iteration instrumentation
        # attach overhead and post-synchronize clock ramp.
        median_time = statistics.median(spans_ms)
        shapes = self._get_input_sizes(inputs)
        logger.debug(
            f"[Autotuner]: cupti-profiling {runner} {tactic}, shapes={shapes}, "
            f"median_time {median_time}"
        )
        return median_time

