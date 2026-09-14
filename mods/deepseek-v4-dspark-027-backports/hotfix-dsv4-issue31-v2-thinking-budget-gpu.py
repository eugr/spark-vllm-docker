#!/usr/bin/env python3
"""Enable V2 ``thinking_token_budget`` without per-token CPU synchronization.

The withdrawn issue #31 implementation copied GPU mappings and token slices to
Python during every decode step. At long context that serialized the sampler
and caused a severe decode-rate regression. This replacement keeps all hot-path
state on the GPU:

* requests without an explicit budget take one NumPy fast-path check and launch
  no extra kernel;
* budget enforcement is a small Triton kernel that only touches the vocabulary
  row when the exact budget boundary is reached; and
* accepted ``</think>`` tokens update the per-request state with another small
  Triton kernel, including accepted MTP/speculative tokens.

DeepSeek V4 uses single-token ``<think>`` and ``</think>`` delimiters. The
hotfix validates that property before accepting a budgeted request. It does not
install an omit-field default: clients must opt in per request.

Usage:
  python3 hotfix-dsv4-issue31-v2-thinking-budget-gpu.py
  python3 hotfix-dsv4-issue31-v2-thinking-budget-gpu.py /path/to/vllm
"""
from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
VLLM = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VLLM
MARK = "# [issue31-gpu-hotfix] V2 thinking_token_budget"

THINKING_BUDGET_PY = r'''# SPDX-License-Identifier: Apache-2.0
# [issue31-gpu-hotfix] V2 thinking_token_budget (DSpark)
"""GPU-resident thinking-budget enforcement for the V2 sampler."""

from __future__ import annotations

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


@triton.jit
def _force_thinking_end_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    pos_ptr,
    input_ids_ptr,
    budget_ptr,
    prompt_len_ptr,
    active_ptr,
    end_token_id: tl.constexpr,
    max_lookback: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req_idx = tl.load(expanded_idx_mapping_ptr + row)
    if req_idx < 0:
        return
    if tl.load(active_ptr + req_idx) == 0:
        return

    budget = tl.load(budget_ptr + req_idx)
    if budget < 0:
        return

    # If an earlier token in this speculative path is </think>, any later
    # accepted row is already outside reasoning. A rejected earlier draft also
    # rejects the later row, so suppressing the force remains correct.
    seen_end = 0
    for back in range(max_lookback):
        prev = row - back
        valid = prev >= 0
        prev_req = tl.load(
            expanded_idx_mapping_ptr + prev, mask=valid, other=-1
        )
        prev_token = tl.load(input_ids_ptr + prev, mask=valid, other=-1)
        seen_end = seen_end | (
            valid & (prev_req == req_idx) & (prev_token == end_token_id)
        )
    if seen_end:
        return

    pos = tl.load(pos_ptr + row)
    prompt_len = tl.load(prompt_len_ptr + req_idx)
    generated_index = pos + 1 - prompt_len
    if generated_index != budget:
        return

    # This loop runs once per request, at the boundary only. Every ordinary
    # decode step returns above without reading or writing the vocabulary row.
    for offset in range(0, vocab_size, block_size):
        cols = offset + tl.arange(0, block_size)
        tl.store(
            logits_ptr + row * logits_stride + cols,
            -float("inf"),
            mask=cols < vocab_size,
        )
    tl.debug_barrier()
    tl.store(logits_ptr + row * logits_stride + end_token_id, 1.0e9)


@triton.jit
def _observe_accepted_end_kernel(
    active_ptr,
    idx_mapping_ptr,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    num_sampled_ptr,
    end_token_id: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0 or tl.load(active_ptr + req_idx) == 0:
        return
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    for i in range(num_sampled):
        token_id = tl.load(
            sampled_tokens_ptr + batch_idx * sampled_tokens_stride + i
        )
        if token_id == end_token_id:
            tl.store(active_ptr + req_idx, 0)


class ThinkingBudgetState:
    """Per-request state with no device-to-host work on the decode path."""

    def __init__(
        self,
        max_num_reqs: int,
        device: torch.device,
        reasoning_config,
        max_lookback: int,
    ) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.max_lookback = max(1, int(max_lookback))
        start = getattr(reasoning_config, "reasoning_start_token_ids", None)
        end = getattr(reasoning_config, "reasoning_end_token_ids", None)
        self.start_ids = [int(x) for x in (start or [])]
        self.end_ids = [int(x) for x in (end or [])]
        self.supported = len(self.start_ids) == 1 and len(self.end_ids) == 1
        self.end_token_id = self.end_ids[0] if self.supported else -1

        self.budget = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.budget.np.fill(-1)
        self.budget.copy_to_uva()
        self.prompt_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.active = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # CPU-only dispatch flag. It is indexed with the sampler's existing
        # NumPy mapping, so the fast-path never synchronizes a CUDA tensor.
        self.use_budget = np.zeros(self.max_num_reqs, dtype=bool)

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        prompt_token_ids: list[int],
        prefill_token_ids: list[int],
        sampling_params,
    ) -> None:
        value = getattr(sampling_params, "thinking_token_budget", None)
        if value is None:
            self.budget.np[req_idx] = -1
            self.prompt_len.np[req_idx] = int(prompt_len)
            self.active.stage_write_elem(req_idx, 0)
            self.use_budget[req_idx] = False
            return
        if not self.supported:
            raise ValueError(
                "V2 GPU thinking_token_budget requires single-token reasoning "
                "start/end delimiters"
            )
        value = int(value)
        if value < 0:
            raise ValueError("thinking_token_budget must be non-negative")

        # DeepSeek V4's generation prompt ends in <think>. During a resumed
        # request, only inspect the already-generated suffix (never the long
        # prompt) for a natural </think>.
        prompt_ends_in_start = bool(prompt_token_ids) and (
            int(prompt_token_ids[-1]) == self.start_ids[0]
        )
        generated_suffix = prefill_token_ids[int(prompt_len) :]
        already_ended = self.end_token_id in generated_suffix
        is_active = prompt_ends_in_start and not already_ended

        self.budget.np[req_idx] = value
        self.prompt_len.np[req_idx] = int(prompt_len)
        self.active.stage_write_elem(req_idx, int(is_active))
        self.use_budget[req_idx] = True

    def apply_staged_writes(self) -> None:
        self.budget.copy_to_uva()
        self.prompt_len.copy_to_uva()
        self.active.apply_write()

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> None:
        if not self.supported or not np.any(self.use_budget[idx_mapping_np]):
            return
        num_rows, vocab_size = logits.shape
        _force_thinking_end_kernel[(num_rows,)](
            logits,
            logits.stride(0),
            vocab_size,
            expanded_idx_mapping,
            pos,
            input_ids,
            self.budget.gpu,
            self.prompt_len.gpu,
            self.active.gpu,
            end_token_id=self.end_token_id,
            max_lookback=self.max_lookback,
            block_size=8192,
            num_warps=1,
        )

    def observe_accepted(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if not self.supported or not np.any(self.use_budget[idx_mapping_np]):
            return
        _observe_accepted_end_kernel[(idx_mapping.shape[0],)](
            self.active.gpu,
            idx_mapping,
            sampled_token_ids,
            sampled_token_ids.stride(0),
            num_sampled,
            end_token_id=self.end_token_id,
            num_warps=1,
        )
'''


def _patch_file(path: Path, old: str, new: str, what: str) -> None:
    source = path.read_text(encoding="utf-8")
    if new in source:
        print(f"[issue31-gpu-hotfix] already applied: {what} ({path})")
        return
    if old not in source:
        raise SystemExit(
            f"[issue31-gpu-hotfix] anchor not found for {what} in {path}"
        )
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    print(f"[issue31-gpu-hotfix] patched {what} in {path}")


def main() -> None:
    dest = VLLM / "v1/worker/gpu/sample/thinking_budget_gpu.py"
    dest.write_text(THINKING_BUDGET_PY, encoding="utf-8")
    print(f"[issue31-gpu-hotfix] wrote {dest}")

    _patch_file(
        VLLM / "v1/engine/input_processor.py",
        '''                if self.use_v2_model_runner:
                    raise ValueError(
                        "thinking_token_budget is not yet supported by the V2 "
                        "model runner. Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 "
                        "to use thinking_token_budget."
                    )''',
        f'''                if self.use_v2_model_runner:
                    # {MARK}: enforced in the V2 GPU sampler.
                    pass''',
        "input-processor V2 gate",
    )

    _patch_file(
        VLLM / "config/vllm.py",
        '''        if self.reasoning_config is not None:
            logger.warning_once(
                "Model Runner V2 does not yet support the thinking_token_budget "
                "request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if this is required."
            )''',
        f'''        if self.reasoning_config is not None:
            # {MARK}: supported by the DSpark GPU sampler hotfix.
            pass''',
        "stale V2 warning",
    )

    sampler = VLLM / "v1/worker/gpu/sample/sampler.py"
    _patch_file(
        sampler,
        '''from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState''',
        f'''from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.sample.thinking_budget_gpu import ThinkingBudgetState  # {MARK}
from vllm.v1.worker.gpu.states import RequestState''',
        "sampler import",
    )
    _patch_file(
        sampler,
        '''        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
    ):''',
        f'''        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
        reasoning_config=None,  # {MARK}
    ):''',
        "sampler init signature",
    )
    _patch_file(
        sampler,
        '''        self.num_speculative_tokens = num_speculative_tokens
        self.use_flashinfer = flashinfer_sampler_supported()''',
        f'''        self.num_speculative_tokens = num_speculative_tokens
        self.use_flashinfer = flashinfer_sampler_supported()
        self.thinking_budget_state = ThinkingBudgetState(  # {MARK}
            max_num_reqs, device, reasoning_config, num_speculative_tokens
        )''',
        "sampler GPU state",
    )
    _patch_file(
        sampler,
        '''    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)''',
        f'''    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int],  # {MARK}
        prefill_token_ids: list[int],
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.thinking_budget_state.add_request(
            req_idx, prompt_len, prompt_token_ids, prefill_token_ids, sampling_params
        )''',
        "sampler add_request",
    )
    _patch_file(
        sampler,
        '''        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()''',
        f'''        self.sampling_states.apply_staged_writes()
        self.thinking_budget_state.apply_staged_writes()  # {MARK}
        self.penalties_state.apply_staged_writes()''',
        "sampler staged writes",
    )
    _patch_file(
        sampler,
        '''        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        if skip_top_k_top_p:
            return logits''',
        f'''        # Apply min_p in place.
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)

        # {MARK}: no CPU sync; only the exact boundary row is overwritten.
        self.thinking_budget_state.apply(
            logits, expanded_idx_mapping, idx_mapping_np, pos, input_ids
        )

        if skip_top_k_top_p:
            return logits''',
        "sampler budget enforcement",
    )

    runner = VLLM / "v1/worker/gpu/model_runner.py"
    _patch_file(
        runner,
        '''                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
            )''',
        f'''                num_speculative_tokens=self.decode_query_len,
                use_fp64_gumbel=self.model_config.use_fp64_gumbel,
                reasoning_config=self.vllm_config.reasoning_config,  # {MARK}
            )''',
        "model-runner sampler config",
    )
    _patch_file(
        runner,
        '''                self.sampler.add_request(
                    req_index, prompt_len, new_req_data.sampling_params
                )''',
        f'''                self.sampler.add_request(
                    req_index,
                    prompt_len,
                    new_req_data.sampling_params,
                    new_req_data.prompt_token_ids,  # {MARK}
                    new_req_data.prefill_token_ids,
                )''',
        "model-runner request tokens",
    )
    _patch_file(
        runner,
        '''            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected''',
        f'''            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        # {MARK}: observe only accepted tokens, entirely on device.
        self.sampler.thinking_budget_state.observe_accepted(
            sampler_output.sampled_token_ids,
            sampler_output.num_sampled,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
        )
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected''',
        "accepted-token observation",
    )
    print("[issue31-gpu-hotfix] done")


if __name__ == "__main__":
    main()
