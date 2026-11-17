# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpts from upstream b409a4f0d787fdddb75ec2e8ddb6dfcfd36235b7 (with preceding series entries).
# tests/models/qwen4_exp/test_config.py

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen4ExpConfig, Qwen4ExpTextConfig

from vllm.config import CacheConfig, ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.models.config import (
    Qwen4ExpForConditionalGenerationConfig,
)
from vllm.models.qwen4_exp.amd.model_state import (
    Qwen4ExpModelState as AmdQwen4ExpModelState,
)
from vllm.models.qwen4_exp.nvidia.model import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)
from vllm.models.qwen4_exp.nvidia.model_state import Qwen4ExpModelState
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

from ...utils import spawn_new_process_for_each_test



@pytest.mark.parametrize(
    "model_cls", [Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration]
)
def test_qkv_proj_mapping_allows_bf16_indexer(model_cls) -> None:
    """Regression test for https://github.com/vllm-project/vllm/issues/62308.

    Layout of e.g. ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ: q/k/v are
    quantized and only the indexer projection is in `ignore`.
    """
    assert not should_ignore_layer(
        "model.layers.3.self_attn.qkv_proj",
        ignore=["model.layers.3.self_attn.indexer.index_qk_proj"],
        fused_mapping=model_cls.packed_modules_mapping,
    )

@pytest.mark.parametrize(
    "state_cls",
    [Qwen4ExpModelState, AmdQwen4ExpModelState],
    ids=["nvidia", "amd"],
)
def test_qwen4_exp_model_state_pads_replayed_request_range(state_cls) -> None:
    """The PLE buffers must be valid for the capture-time request range.

    A PIECEWISE graph replays with the request count of the piece it was captured
    for, so its request lookup scans every row of `ple_query_start_loc` up to that
    count and reads the matching `ngram_context` rows.
    """
    model_state = object.__new__(state_cls)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 2
    model_state.ngram_eos_token_id = 99
    model_state.max_num_reqs = 8
    model_state.ngram_context = torch.empty((8, 2), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-2, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.zeros(9, dtype=torch.int32)
    # Empty: the disk-offload staging path is covered by test_ple.py.
    model_state.disk_offload_modules = []
    req_states = SimpleNamespace(
        # Rows 0 and 1 have prior context, so the context rows are not all sentinel.
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
        ),
        all_token_ids=SimpleNamespace(gpu=torch.arange(64).view(8, 8).to(torch.int32)),
    )

    # A wider earlier batch leaves a non-monotonic tail behind.
    wide_batch = SimpleNamespace(
        num_reqs=3,
        num_reqs_after_padding=3,
        num_tokens=3,
        idx_mapping=torch.arange(3),
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_state.prepare_inputs(wide_batch, req_states)

    # One request, 8 tokens: the 8-token piece was captured with R_cap = 8, but
    # PIECEWISE does not pad requests, so num_reqs_after_padding == num_reqs == 1.
    batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=1,
        num_tokens=8,
        idx_mapping=torch.tensor([0]),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
    )
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_state.prepare_inputs(batch, req_states)

    decoded = model_state.ple_query_start_loc
    assert decoded[0] == 0
    assert torch.all(decoded[1:] >= decoded[:-1]), decoded
    # Unused capacity must not capture real tokens.
    assert torch.all(decoded[2:] >= 8), decoded
    # A search over the captured 8-request range must resolve every real token
    # to request 0, not to the rows left over from the wider batch.
    for token in range(8):
        assert max(i for i in range(9) if int(decoded[i]) <= token) == 0
    # Row 0 keeps its live context; rows the batch does not own must be sentinels
    # (they are still read for padding tokens).
    expected_context = torch.full((8, 2), 99, dtype=torch.int32)
    expected_context[0] = torch.tensor([99, 0], dtype=torch.int32)
    torch.testing.assert_close(model_state.ngram_context, expected_context)

    # A request-padded batch (FULL graphs): the runner already pads its own copy,
    # so the buffer tail past it must be padded with the same value.
    padded_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=4,
        num_tokens=6,
        idx_mapping=torch.tensor([0, 1]),
        query_start_loc=torch.tensor([0, 2, 4, 6, 6], dtype=torch.int32),
    )
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_state.prepare_inputs(padded_batch, req_states)
    torch.testing.assert_close(
        model_state.ple_query_start_loc,
        torch.tensor([0, 2, 4, 6, 6, 6, 6, 6, 6], dtype=torch.int32),
    )
    assert torch.all(model_state.ngram_context[2:] == 99), model_state.ngram_context

    # A full batch leaves no tail to pad.
    full_batch = SimpleNamespace(
        num_reqs=8,
        num_reqs_after_padding=8,
        num_tokens=8,
        idx_mapping=torch.arange(8),
        query_start_loc=torch.arange(9, dtype=torch.int32),
    )
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_state.prepare_inputs(full_batch, req_states)
    torch.testing.assert_close(
        model_state.ple_query_start_loc, torch.arange(9, dtype=torch.int32)
    )

    # Capture-time dummy inputs follow the same convention.
    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        model_state.prepare_dummy_inputs(num_reqs=2, num_tokens=5)
    torch.testing.assert_close(
        model_state.ple_query_start_loc[:3], torch.tensor([0, 2, 5], dtype=torch.int32)
    )
    assert torch.all(model_state.ple_query_start_loc[3:] >= 5)
