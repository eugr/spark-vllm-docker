# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

StepCurve = Callable[[int], float]

_STACKED_PARAM_NAME_MAPPING = (
    ("attn.fused_wqa_wkv", ".attn.wq_a", 0),
    ("attn.fused_wqa_wkv", ".attn.wkv", 1),
)


def map_dspark_stacked_param_name(name: str) -> tuple[str, int] | None:
    """Map checkpoint names that load into stacked vLLM parameters.

    Keep this segment-aware: the DSpark checkpoint also has names such as
    ``markov_w1`` that must not be treated as FFN ``w1`` shards.
    """

    if ".experts." in name:
        return None
    for param_name, weight_name, shard_id in _STACKED_PARAM_NAME_MAPPING:
        if weight_name in name:
            return name.replace(weight_name, f".{param_name}"), shard_id
    return None


def make_dspark_warmup_draft_token_ids(
    *,
    batch_size: int,
    num_speculative_tokens: int,
    noise_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a valid synthetic draft block for DSpark's cache-warm step.

    The first DSpark proposal call has prompt target features available, but no
    generated-token target feature yet. vLLM's async speculative path still
    expects a fixed tensor of draft ids, so use the model's valid noise token as
    a conservative one-step proposal rather than Python empty lists or -1
    placeholders.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_speculative_tokens <= 0:
        raise ValueError(
            f"num_speculative_tokens must be positive, got {num_speculative_tokens}"
        )
    if noise_token_id < 0:
        raise ValueError(f"noise_token_id must be non-negative, got {noise_token_id}")
    return torch.full(
        (batch_size, num_speculative_tokens),
        int(noise_token_id),
        dtype=torch.int32,
        device=device,
    )


def unpack_mhc_pre_outputs(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return MHC pre outputs as ``(layer_input, post_mix, comb_mix)``.

    The direct mHC pre op returns ``(post_mix, comb_mix, layer_input)`` while
    DSpark's draft layer wants to mirror the decoder block local names.
    """

    post_mix, comb_mix, layer_input = outputs
    return layer_input, post_mix, comb_mix


@dataclass(frozen=True)
class DSparkModelSpec:
    """Normalized DSpark fields from a model config."""

    block_size: int
    noise_token_id: int
    target_layer_ids: tuple[int, ...]
    markov_rank: int
    markov_head_type: str
    confidence_head_with_markov: bool
    weight_prefix: str | None = None
    num_draft_layers: int | None = None

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> DSparkModelSpec:
        if not has_dspark_config(hf_config):
            raise ValueError("hf_config does not contain DSpark fields")

        block_size = int(_get_config_value(hf_config, "dspark_block_size"))
        if block_size <= 0:
            raise ValueError(f"dspark_block_size must be positive, got {block_size}")

        noise_token_id = int(_get_config_value(hf_config, "dspark_noise_token_id"))
        if noise_token_id < 0:
            raise ValueError(
                f"dspark_noise_token_id must be non-negative, got {noise_token_id}"
            )

        raw_layer_ids = _get_config_value(hf_config, "dspark_target_layer_ids")
        target_layer_ids = tuple(int(layer_id) for layer_id in raw_layer_ids)
        if not target_layer_ids:
            raise ValueError("dspark_target_layer_ids must not be empty")
        if any(
            layer_id <= previous
            for previous, layer_id in zip(target_layer_ids, target_layer_ids[1:])
        ):
            raise ValueError("dspark_target_layer_ids must be strictly increasing")

        num_hidden_layers = _get_config_value(hf_config, "num_hidden_layers", None)
        if num_hidden_layers is not None:
            upper = int(num_hidden_layers) - 1
            for layer_id in target_layer_ids:
                if layer_id != -1 and not 0 <= layer_id <= upper:
                    raise ValueError(
                        "dspark_target_layer_ids contains layer "
                        f"{layer_id}, outside {{-1}} U [0, {upper}]"
                    )

        markov_rank = int(_get_config_value(hf_config, "dspark_markov_rank", 0))
        if markov_rank < 0:
            raise ValueError(
                f"dspark_markov_rank must be non-negative, got {markov_rank}"
            )

        markov_head_type = _get_config_value(hf_config, "dspark_markov_head_type", None)
        if markov_head_type is None:
            markov_head_type = "vanilla"
        markov_head_type = str(markov_head_type).lower()

        confidence_head_with_markov = _get_config_value(
            hf_config, "dspark_confidence_head_with_markov", None
        )
        if confidence_head_with_markov is None:
            confidence_head_with_markov = markov_rank > 0

        return cls(
            block_size=block_size,
            noise_token_id=noise_token_id,
            target_layer_ids=target_layer_ids,
            markov_rank=markov_rank,
            markov_head_type=markov_head_type,
            confidence_head_with_markov=bool(confidence_head_with_markov),
            weight_prefix=infer_dspark_weight_prefix(
                _get_config_value(hf_config, "_weight_names", ())
            ),
            num_draft_layers=infer_dspark_num_draft_layers(
                _get_config_value(hf_config, "_weight_names", ())
            ),
        )

    def confidence_input_dim(self, hidden_size: int) -> int:
        hidden_size = int(hidden_size)
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if self.confidence_head_with_markov:
            return hidden_size + self.markov_rank
        return hidden_size


@dataclass(frozen=True)
class DSparkScheduleResult:
    """Selected DSpark verification lengths and their profiled throughput."""

    lengths: tuple[int, ...]
    expected_accepted_tokens: float
    batch_tokens: int
    expected_tokens_per_second: float


@dataclass(frozen=True)
class DSparkDiagnosticsSnapshot:
    """Aggregated DSpark scheduling signals for logs or metrics exporters."""

    num_steps: int
    num_requests: int
    num_possible_draft_tokens: int
    num_scheduled_draft_tokens: int
    scheduled_length_histogram: tuple[int, ...]
    avg_scheduled_length: float
    draft_token_prune_rate: float
    expected_acceptance_length: float
    avg_expected_tokens_per_second: float
    avg_confidence_per_pos: tuple[float, ...]
    avg_survival_per_pos: tuple[float, ...]
    scheduled_fraction_per_pos: tuple[float, ...]


@dataclass(frozen=True)
class DSparkPosition0DiagnosticsSnapshot:
    """Aggregated position-0 draft quality signals."""

    num_tokens: int
    num_matches: int
    match_rate: float
    avg_confidence: float | None
    avg_confidence_when_matched: float | None
    avg_confidence_when_missed: float | None
    num_confidence_logits_normalized: int


@dataclass
class DSparkDiagnostics:
    """Accumulate DSpark confidence-scheduler diagnostics.

    This intentionally sits below Prometheus/logging integration so the DSpark
    proposer can collect useful signals before we settle the runtime wiring.
    """

    max_spec_tokens: int
    num_steps: int = 0
    num_requests: int = 0
    num_possible_draft_tokens: int = 0
    num_scheduled_draft_tokens: int = 0
    total_expected_accepted_tokens: float = 0.0
    total_expected_tokens_per_second: float = 0.0
    scheduled_length_histogram: list[int] = field(default_factory=list)
    confidence_sums: list[float] = field(default_factory=list)
    confidence_counts: list[int] = field(default_factory=list)
    survival_sums: list[float] = field(default_factory=list)
    survival_counts: list[int] = field(default_factory=list)
    scheduled_counts: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.max_spec_tokens = int(self.max_spec_tokens)
        if self.max_spec_tokens <= 0:
            raise ValueError(
                f"max_spec_tokens must be positive, got {self.max_spec_tokens}"
            )
        self.scheduled_length_histogram = [0] * (self.max_spec_tokens + 1)
        self.confidence_sums = [0.0] * self.max_spec_tokens
        self.confidence_counts = [0] * self.max_spec_tokens
        self.survival_sums = [0.0] * self.max_spec_tokens
        self.survival_counts = [0] * self.max_spec_tokens
        self.scheduled_counts = [0] * self.max_spec_tokens

    def observe(
        self,
        confidence_rows: Sequence[Sequence[float]],
        schedule_result: DSparkScheduleResult,
    ) -> None:
        if len(confidence_rows) != len(schedule_result.lengths):
            raise ValueError(
                "confidence_rows and schedule_result.lengths must have the same length"
            )

        self.num_steps += 1
        self.num_requests += len(confidence_rows)
        self.num_possible_draft_tokens += sum(len(row) for row in confidence_rows)
        self.num_scheduled_draft_tokens += sum(schedule_result.lengths)
        self.total_expected_accepted_tokens += schedule_result.expected_accepted_tokens
        self.total_expected_tokens_per_second += (
            schedule_result.expected_tokens_per_second
        )

        for request_index, (confidences, scheduled_length) in enumerate(
            zip(confidence_rows, schedule_result.lengths, strict=True)
        ):
            if len(confidences) > self.max_spec_tokens:
                raise ValueError(
                    f"confidence_rows[{request_index}] has {len(confidences)} "
                    f"tokens, exceeding max_spec_tokens={self.max_spec_tokens}"
                )
            if scheduled_length < 0 or scheduled_length > len(confidences):
                raise ValueError(
                    f"scheduled length {scheduled_length} is invalid for "
                    f"confidence_rows[{request_index}]"
                )

            self.scheduled_length_histogram[scheduled_length] += 1
            survivals = cumulative_survival(confidences)
            for position, confidence in enumerate(confidences):
                self.confidence_sums[position] += float(confidence)
                self.confidence_counts[position] += 1
                self.survival_sums[position] += survivals[position]
                self.survival_counts[position] += 1
                if position < scheduled_length:
                    self.scheduled_counts[position] += 1

    def snapshot(self) -> DSparkDiagnosticsSnapshot:
        avg_confidence_per_pos = _safe_average_tuple(
            self.confidence_sums,
            self.confidence_counts,
        )
        avg_survival_per_pos = _safe_average_tuple(
            self.survival_sums,
            self.survival_counts,
        )
        scheduled_fraction_per_pos = _safe_average_tuple(
            [float(value) for value in self.scheduled_counts],
            self.confidence_counts,
        )

        avg_scheduled_length = (
            self.num_scheduled_draft_tokens / self.num_requests
            if self.num_requests > 0
            else 0.0
        )
        draft_token_prune_rate = (
            1.0 - (self.num_scheduled_draft_tokens / self.num_possible_draft_tokens)
            if self.num_possible_draft_tokens > 0
            else 0.0
        )
        expected_acceptance_length = (
            self.total_expected_accepted_tokens / self.num_requests
            if self.num_requests > 0
            else 0.0
        )
        avg_expected_tokens_per_second = (
            self.total_expected_tokens_per_second / self.num_steps
            if self.num_steps > 0
            else 0.0
        )

        return DSparkDiagnosticsSnapshot(
            num_steps=self.num_steps,
            num_requests=self.num_requests,
            num_possible_draft_tokens=self.num_possible_draft_tokens,
            num_scheduled_draft_tokens=self.num_scheduled_draft_tokens,
            scheduled_length_histogram=tuple(self.scheduled_length_histogram),
            avg_scheduled_length=avg_scheduled_length,
            draft_token_prune_rate=draft_token_prune_rate,
            expected_acceptance_length=expected_acceptance_length,
            avg_expected_tokens_per_second=avg_expected_tokens_per_second,
            avg_confidence_per_pos=avg_confidence_per_pos,
            avg_survival_per_pos=avg_survival_per_pos,
            scheduled_fraction_per_pos=scheduled_fraction_per_pos,
        )


@dataclass
class DSparkPosition0Diagnostics:
    """Accumulate first-draft-token agreement and confidence diagnostics."""

    num_tokens: int = 0
    num_matches: int = 0
    confidence_sum: float = 0.0
    confidence_count: int = 0
    matched_confidence_sum: float = 0.0
    matched_confidence_count: int = 0
    missed_confidence_sum: float = 0.0
    missed_confidence_count: int = 0
    num_confidence_logits_normalized: int = 0

    def observe(
        self,
        matches: Sequence[bool],
        confidences: Sequence[float] | None = None,
    ) -> None:
        if confidences is not None and len(confidences) != len(matches):
            raise ValueError("confidences and matches must have the same length")

        for index, match in enumerate(matches):
            self.num_tokens += 1
            matched = bool(match)
            if matched:
                self.num_matches += 1

            if confidences is None:
                continue

            confidence = float(confidences[index])
            if not math.isfinite(confidence):
                continue
            if confidence < 0.0 or confidence > 1.0:
                # Diagnostic-only path: preserve the observation without
                # turning an unexpected raw confidence logit into an engine
                # failure. Production schedulers still validate strictly.
                confidence = _sigmoid_scalar(confidence)
                self.num_confidence_logits_normalized += 1
            self.confidence_sum += confidence
            self.confidence_count += 1
            if matched:
                self.matched_confidence_sum += confidence
                self.matched_confidence_count += 1
            else:
                self.missed_confidence_sum += confidence
                self.missed_confidence_count += 1

    @staticmethod
    def _avg(total: float, count: int) -> float | None:
        return total / count if count > 0 else None

    def snapshot(self) -> DSparkPosition0DiagnosticsSnapshot:
        return DSparkPosition0DiagnosticsSnapshot(
            num_tokens=self.num_tokens,
            num_matches=self.num_matches,
            match_rate=(
                self.num_matches / self.num_tokens if self.num_tokens > 0 else 0.0
            ),
            avg_confidence=self._avg(self.confidence_sum, self.confidence_count),
            avg_confidence_when_matched=self._avg(
                self.matched_confidence_sum,
                self.matched_confidence_count,
            ),
            avg_confidence_when_missed=self._avg(
                self.missed_confidence_sum,
                self.missed_confidence_count,
            ),
            num_confidence_logits_normalized=self.num_confidence_logits_normalized,
        )


def _get_config_value(hf_config: Any, name: str, default: Any = ...):
    if isinstance(hf_config, dict):
        if default is ...:
            return hf_config[name]
        return hf_config.get(name, default)
    if default is ...:
        return getattr(hf_config, name)
    return getattr(hf_config, name, default)


def has_dspark_config(hf_config: Any) -> bool:
    return _get_config_value(hf_config, "dspark_block_size", None) is not None


def infer_dspark_weight_prefix(weight_names: Sequence[str]) -> str | None:
    prefixes = set()
    suffixes = (
        ".markov_head.markov_w1.weight",
        ".markov_head.markov_w2.weight",
        ".confidence_head.proj.weight",
    )
    for name in weight_names:
        for suffix in suffixes:
            if name.endswith(suffix):
                prefixes.add(name.removesuffix(suffix))

    if not prefixes:
        return None
    if len(prefixes) > 1:
        raise ValueError(
            f"DSpark weights were found under multiple prefixes: {sorted(prefixes)}"
        )
    return next(iter(prefixes))


def infer_dspark_num_draft_layers(weight_names: Sequence[str]) -> int | None:
    """Infer how many DSpark draft stages are present in a checkpoint.

    The DeepSeek-V4-Flash-DSpark release stores the DSpark draft stages under
    the historical `mtp.N.*` namespace. Unlike regular MTP, the HF config still
    reports `num_nextn_predict_layers=1`, so the weight map is the reliable
    source for the DSpark draft depth.
    """

    layer_ids: set[int] = set()
    for name in weight_names:
        if not name.startswith("mtp."):
            continue
        parts = name.split(".", 2)
        if len(parts) < 3:
            continue
        try:
            layer_ids.add(int(parts[1]))
        except ValueError:
            continue

    if not layer_ids:
        return None
    expected = set(range(max(layer_ids) + 1))
    if layer_ids != expected:
        raise ValueError(
            "DSpark MTP namespace must be contiguous from mtp.0; "
            f"found {sorted(layer_ids)}"
        )
    return max(layer_ids) + 1


def _safe_average_tuple(
    values: Sequence[float], counts: Sequence[int]
) -> tuple[float, ...]:
    if len(values) != len(counts):
        raise ValueError("values and counts must have the same length")
    return tuple(
        float(value) / count if count > 0 else 0.0
        for value, count in zip(values, counts, strict=True)
    )


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _sigmoid_scalar(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def speculative_acceptance_confidence(
    draft_probs: Sequence[float],
    target_probs: Sequence[float],
) -> float:
    """Return the exact one-step speculative acceptance probability.

    For normalized draft distribution q and target distribution p, the expected
    acceptance probability is sum_x min(p(x), q(x)), equivalently
    1 - TV(p, q).
    """

    if len(draft_probs) != len(target_probs):
        raise ValueError(
            "draft_probs and target_probs must have the same vocabulary size"
        )
    if len(draft_probs) == 0:
        raise ValueError("probability vectors must not be empty")

    draft_total = sum(float(prob) for prob in draft_probs)
    target_total = sum(float(prob) for prob in target_probs)
    if draft_total <= 0.0 or target_total <= 0.0:
        raise ValueError("probability vectors must have positive total mass")

    draft = [float(prob) / draft_total for prob in draft_probs]
    target = [float(prob) / target_total for prob in target_probs]
    if any(prob < 0.0 for prob in (*draft, *target)):
        raise ValueError("probabilities must be non-negative")

    return sum(
        min(draft_prob, target_prob)
        for draft_prob, target_prob in zip(draft, target, strict=True)
    )


def cumulative_survival(confidences: Sequence[float]) -> tuple[float, ...]:
    """Convert conditional per-token confidences to prefix survival rates."""

    survivals: list[float] = []
    survival = 1.0
    for index, confidence in enumerate(confidences):
        survival *= _validate_probability(confidence, f"confidences[{index}]")
        survivals.append(survival)
    return tuple(survivals)


def confidence_threshold_prefix_length(
    confidences: Sequence[float],
    threshold: float,
) -> int:
    """Choose the longest prefix whose cumulative confidence stays above a threshold.

    The decision is non-anticipating: position n is admitted using only
    confidences from positions <= n. This keeps DSpark's confidence scheduling
    compatible with rejection sampling.
    """

    threshold = _validate_probability(threshold, "threshold")
    admitted = 0
    survival = 1.0
    for index, confidence in enumerate(confidences):
        survival *= _validate_probability(confidence, f"confidences[{index}]")
        if survival < threshold:
            break
        admitted += 1
    return admitted


def score_prefix_lengths(
    confidence_rows: Sequence[Sequence[float]],
    lengths: Sequence[int],
    *,
    steps_per_second: StepCurve,
) -> DSparkScheduleResult:
    """Score a fixed per-request prefix schedule against a profiled step curve."""

    if len(confidence_rows) != len(lengths):
        raise ValueError("confidence_rows and lengths must have the same length")

    expected_accepted_tokens = float(len(confidence_rows))
    batch_tokens = len(confidence_rows)
    normalized_lengths: list[int] = []

    for request_index, (confidences, length) in enumerate(
        zip(confidence_rows, lengths, strict=True)
    ):
        length = int(length)
        if length < 0 or length > len(confidences):
            raise ValueError(
                f"lengths[{request_index}] must be in [0, {len(confidences)}], "
                f"got {length}"
            )

        expected_accepted_tokens += sum(cumulative_survival(confidences)[:length])
        batch_tokens += length
        normalized_lengths.append(length)

    step_rate = float(steps_per_second(batch_tokens))
    if step_rate < 0.0:
        raise ValueError("steps_per_second must return a non-negative value")

    return DSparkScheduleResult(
        lengths=tuple(normalized_lengths),
        expected_accepted_tokens=expected_accepted_tokens,
        batch_tokens=batch_tokens,
        expected_tokens_per_second=expected_accepted_tokens * step_rate,
    )


def hardware_aware_prefix_schedule(
    confidence_rows: Sequence[Sequence[float]],
    *,
    steps_per_second: StepCurve,
    early_stop: bool = True,
) -> DSparkScheduleResult:
    """Choose DSpark verification prefix lengths for a batch.

    `confidence_rows[r][j]` is the conditional acceptance probability for
    request r at draft position j. The planner greedily admits prefix tokens by
    descending cumulative survival, then keeps the best point along that path
    after applying the supplied profiled engine step-rate curve.

    `early_stop=True` is intended for smooth capacity curves where adding more
    verification tokens cannot recover from the first throughput drop. Set it to
    `False` for exhaustive traversal of the greedy prefix path when the profiled
    curve has jagged capacity cliffs.
    """

    request_count = len(confidence_rows)
    lengths = [0] * request_count
    best_lengths = tuple(lengths)
    batch_tokens = request_count

    base_step_rate = float(steps_per_second(batch_tokens))
    if base_step_rate < 0.0:
        raise ValueError("steps_per_second must return a non-negative value")

    expected_accepted_tokens = float(request_count)
    best_batch_tokens = batch_tokens
    best_expected_accepted_tokens = expected_accepted_tokens
    best_tokens_per_second = expected_accepted_tokens * base_step_rate

    candidates: list[tuple[float, int, int]] = []
    for request_index, confidences in enumerate(confidence_rows):
        for position, survival in enumerate(cumulative_survival(confidences), start=1):
            if survival > 0.0:
                candidates.append((survival, request_index, position))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    for survival, request_index, position in candidates:
        if position != lengths[request_index] + 1:
            continue

        lengths[request_index] = position
        batch_tokens += 1
        expected_accepted_tokens += survival

        step_rate = float(steps_per_second(batch_tokens))
        if step_rate < 0.0:
            raise ValueError("steps_per_second must return a non-negative value")
        tokens_per_second = expected_accepted_tokens * step_rate

        if tokens_per_second > best_tokens_per_second:
            best_lengths = tuple(lengths)
            best_batch_tokens = batch_tokens
            best_expected_accepted_tokens = expected_accepted_tokens
            best_tokens_per_second = tokens_per_second
            continue

        if early_stop:
            break

    return DSparkScheduleResult(
        lengths=best_lengths,
        expected_accepted_tokens=best_expected_accepted_tokens,
        batch_tokens=best_batch_tokens,
        expected_tokens_per_second=best_tokens_per_second,
    )
