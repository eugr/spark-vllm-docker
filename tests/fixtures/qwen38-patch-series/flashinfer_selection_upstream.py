# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Upstream excerpts from b1cc4f35a5e30bdf378ccb5f6c398ce154375774:
# flashinfer/autotuner/autotuner.py

class AutoTuner:
    def select(self, measured, min_time, tuning_config, runners, tensors, p,
               custom_op, inputs, prepared_input_batches=None, kwargs=None):
        runner_id = 0
        # Retain the original nesting depth for the vendored source hunk.
        if True:
            if True:
                if True:
                    if True:
                        if runner_id is not None:
                            # At least one valid (runner, tactic) pair is found
                            winner = self._settle_near_ties(
                                measured,
                                functools.partial(
                                    self._retime_candidate,
                                    runners,
                                    tensors,
                                    tuning_config,
                                    prepared_input_batches,
                                    kwargs,
                                ),
                                self._tie_tolerance,
                            )
                            _, runner_id, tactic = measured[winner]
                            cache_key = self._get_cache_key(
                                custom_op,
                                runners[runner_id],
                                p.get_opt_shapes(),
                                tuning_config,
                                runners[runner_id].get_cache_key_extras(tensors),
                            )
                            if _tune_process_group is None:
                                self._discard_profile_cache(
                                    custom_op,
                                    runners,
                                    p.get_opt_shapes(),
                                    tuning_config,
                                    inputs,
                                    profile_inputs=tensors,
                                    republish_key=cache_key.file_key,
                                )
                            self._cache_profile_result(
                                cache_key,
                                p,
                                tuning_config,
                                tactic,
                                self._selection_record(
                                    [t for t, _, _ in measured], winner
                                ),
                            )
                            self.stats.tuned_op_successful_configs[custom_op] = (
                                self.stats.tuned_op_successful_configs.get(custom_op, 0)
                                + 1
                            )
                            logger.debug(
                                f"[Autotuner]: profiling chosen runner: {runners[runner_id]} {tactic} for {cache_key}"
                            )


    @staticmethod
    def _selection_record(times: list[float], winner: int) -> dict[str, Any]:
        """How the tactic at ``times[winner]`` was chosen, saved with its entry.

        ``runner_up_ms`` is the fastest other candidate's time and ``margin``
        is ``runner_up_ms / time_ms - 1``, both from the first measurement.
        When ``tie_tolerance`` kept an earlier-enumerated tactic over a faster
        one, ``runner_up_ms`` is below ``time_ms`` and ``margin`` is negative.
        """
        others = [t for i, t in enumerate(times) if i != winner and t < float("inf")]
        time_ms = times[winner]
        runner_up_ms = min(others, default=None)
        return {
            "time_ms": round(time_ms, 6),
            "runner_up_ms": None if runner_up_ms is None else round(runner_up_ms, 6),
            "margin": (
                round(runner_up_ms / time_ms - 1.0, 6)
                if runner_up_ms is not None and time_ms > 0
                else None
            ),
            "candidates": len(others) + 1,
        }

    def _settle_near_ties(
        self,
        measured: list[tuple[float, int, Any]],
        retime: Callable[[int, Any], float],
        tolerance: float,
    ) -> int:
        """Return the index in ``measured`` of the tactic to keep (#7884).

        ``measured`` holds ``(time_ms, runner_id, tactic)`` per candidate in
        enumeration order. The candidates within ``tolerance`` of the fastest
        and enumerated before it (at most ``_TIE_MAX_CONTENDERS``) are re-timed
        against it in that order, alternating over ``_TIE_CONFIRM_ROUNDS``
        rounds; the first whose mean stays within ``tolerance`` is kept, else
        the fastest. Unlike measured order, enumeration order is the same in
        every process, and all ranks see the same group-reduced times.
        """
        inf = float("inf")
        finite = [i for i, (t, _, _) in enumerate(measured) if t < inf]
        # min() keeps the first of equal times, as the strict argmin does.
        fastest = min(finite, key=lambda i: measured[i][0])
        limit = measured[fastest][0] * (1.0 + tolerance)
        contenders = [i for i in finite if i < fastest and measured[i][0] <= limit]
        for i in contenders[: self._TIE_MAX_CONTENDERS]:
            own: list[float] = []
            ref: list[float] = []
            for round_idx in range(self._TIE_CONFIRM_ROUNDS):
                pair = ((i, own), (fastest, ref))
                # Alternate the order so drift favours neither of the two.
                for j, out in pair if round_idx % 2 == 0 else pair[::-1]:
                    _, runner_id, tactic = measured[j]
                    out.append(retime(runner_id, tactic))
            # A failed re-timing is inf, so that contender is not kept.
            if max(own + ref) < inf and statistics.fmean(own) <= statistics.fmean(
                ref
            ) * (1.0 + tolerance):
                return i
        return fastest
