#!/bin/bash
set -euo pipefail

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
PR41797_URL="https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/41797.diff"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SITE_PACKAGES"

echo "[fix-mimo-v2-vllm] Applying MiMo V2.5 vLLM fixes"

if [ ! -f "$MOD_DIR/chat_template.jinja" ]; then
    echo "[fix-mimo-v2-vllm] ERROR: fixed MiMo chat_template.jinja missing from mod directory: $MOD_DIR" >&2
    exit 1
fi

echo "[fix-mimo-v2-vllm] fixed MiMo chat template available at $MOD_DIR/chat_template.jinja"

# CyberTen forum note: vLLM's multimodal audio path uses soundfile + librosa,
# not torchcodec. Keep this harmless for text-only runs and necessary for audio.
python3 - <<'PY' || uv pip install --quiet soundfile librosa
import soundfile  # noqa: F401
import librosa  # noqa: F401
PY

# PR #42969: the "mimo" tool parser is an alias for Qwen3XMLToolParser.  When
# the streaming XML parser closes a <function> element it already marks the
# function closed, but older vLLM builds leave current_function_name populated.
# Fallback recovery can then treat the function as still active and emit an
# extra closing brace in streamed tool-call deltas.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3xml_tool_parser.py')
if not path.exists():
    print('[fix-mimo-v2-vllm] qwen3xml tool parser not present; skipping PR #42969')
    raise SystemExit
text = path.read_text()
old = '''            self.current_function_open = False

        elif name == "tool_call":
'''
new = '''            self.current_function_open = False
            self.current_function_name = None

        elif name == "tool_call":
'''
if new in text:
    print('[fix-mimo-v2-vllm] Qwen3XML/MiMo tool parser PR #42969 already patched')
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched Qwen3XML/MiMo streaming tool parser (#42969)')
else:
    raise SystemExit('[fix-mimo-v2-vllm] ERROR: Qwen3XML tool parser pattern not found for PR #42969')
PY

# vLLM currently aliases the MiMo reasoning parser to Qwen3ReasoningParser.
# That parser decides whether streaming output has already left the reasoning
# phase by scanning the whole prompt for the last <think>/</think> token.  The
# MiMo V2.5 template does NOT prefill <think> for thinking-on generation, but
# the rendered prompt can contain closed <think></think> spans in tool
# instructions and assistant history.  The generic scan then marks reasoning as
# already ended before generation starts, so generated <think>...</think> is
# routed through the tool parser as ordinary content.  Install a tiny MiMo
# subclass that treats the fresh assistant generation prompt as reasoning-open
# unless the prompt explicitly ends with the thinking-off <think></think> stub.
echo "[fix-mimo-v2-vllm] Installing MiMo-specific reasoning parser prompt-state fix"
cat > "$SITE_PACKAGES/vllm/reasoning/mimo_reasoning_parser.py" <<'PY'
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser


class MimoReasoningParser(Qwen3ReasoningParser):
    """MiMo V2.5 reasoning parser.

    MiMo's chat template leaves the assistant generation prompt at
    ``<|im_start|>assistant\n`` when thinking is enabled.  Closed think tags
    can still occur earlier in the prompt (tool instructions say
    ``<think></think>`` and assistant history is rendered with think spans), so
    Qwen3ReasoningParser.is_reasoning_end(prompt_ids) can return True before
    any new assistant tokens have been generated.  That leaks generated
    ``<think>`` text as normal content and confuses streamed tool-call parsing.
    """

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._assistant_generation_prefix_token_ids = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self._assistant_thinking_disabled_suffix_token_ids = tokenizer.encode(
            "<|im_start|>assistant\n<think></think>", add_special_tokens=False
        )

    @staticmethod
    def _endswith(input_ids: Sequence[int], suffix: Sequence[int]) -> bool:
        if not suffix or len(input_ids) < len(suffix):
            return False
        return list(input_ids[-len(suffix) :]) == list(suffix)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # This is the prompt-time check used by vLLM before streaming starts.
        # If generation is about to begin after the bare assistant prefix,
        # reasoning is open even if earlier prompt text contains </think>.
        if self._endswith(input_ids, self._assistant_generation_prefix_token_ids):
            return False

        # Thinking-off MiMo prompts end with an explicit empty reasoning span;
        # in that case output should be routed as content/tool text immediately.
        if self._endswith(
            input_ids, self._assistant_thinking_disabled_suffix_token_ids
        ):
            return True

        return super().is_reasoning_end(input_ids)
PY
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/reasoning/__init__.py')
text = path.read_text()
old = '''    "mimo": (
        "qwen3_reasoning_parser",
        "Qwen3ReasoningParser",
    ),
'''
new = '''    "mimo": (
        "mimo_reasoning_parser",
        "MimoReasoningParser",
    ),
'''
if new in text:
    print('[fix-mimo-v2-vllm] MiMo reasoning parser alias already patched')
elif old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched MiMo reasoning parser alias')
else:
    raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMo reasoning parser alias pattern not found')
PY

# Harden Qwen3XML/MiMo streaming deltas for OpenAI-compatible clients.  The
# parser emits terminal tool-call deltas with name=None and arguments=""; some
# clients/frontends treat those as separate empty calls (displayed as {}).  Also
# avoid constructing DeltaMessage(tool_calls=[]) for ordinary text chunks,
# because Pydantic marks the empty list as explicitly set and serializes it.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3xml_tool_parser.py')
if not path.exists():
    print('[fix-mimo-v2-vllm] qwen3xml tool parser not present; skipping empty-delta hardening')
    raise SystemExit
text = path.read_text()
orig = text
old = '''        return DeltaMessage(
            content=merged_content if merged_content else None,
            tool_calls=merged_tool_calls,
        )
'''
new = '''        kwargs = {}
        if merged_content:
            kwargs["content"] = merged_content
        if merged_tool_calls:
            kwargs["tool_calls"] = merged_tool_calls
        return DeltaMessage(**kwargs)
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: Qwen3XML merge return pattern not found')
    text = text.replace(old, new, 1)
old = '''        elif name.startswith("function") or name == "function":
            # if there are parameters, close JSON object
            if self.parameters:
'''
new = '''        elif name.startswith("function") or name == "function":
            # Ignore malformed/empty <function> blocks that never provided a
            # name; otherwise a bare <tool_call></tool_call> can become an
            # empty OpenAI tool call with `{}` arguments.
            if not self.current_function_name:
                self.current_function_open = False
                return

            # if there are parameters, close JSON object
            if self.parameters:
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: Qwen3XML function-end guard pattern not found')
    text = text.replace(old, new, 1)
old = '''        # Parse the delta text and get the result
        delta = self.parser.parse_single_streaming_chunks(delta_text)

        # Update tool call tracking arrays based on incremental parsing results
        if delta and delta.tool_calls:
'''
new = '''        # Parse the delta text and get the result
        delta = self.parser.parse_single_streaming_chunks(delta_text)

        # Drop terminal no-op tool-call deltas.  They carry no name and no
        # argument bytes; retaining them makes less defensive clients create a
        # second empty tool call.
        if delta and delta.tool_calls:
            meaningful_tool_calls = []
            for tool_call in delta.tool_calls:
                function = tool_call.function
                if function and (function.name is not None or function.arguments not in (None, "")):
                    meaningful_tool_calls.append(tool_call)
            if len(meaningful_tool_calls) != len(delta.tool_calls):
                delta.tool_calls = meaningful_tool_calls

        # Update tool call tracking arrays based on incremental parsing results
        if delta and delta.tool_calls:
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: Qwen3XML streaming filter insertion pattern not found')
    text = text.replace(old, new, 1)
if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] hardened Qwen3XML/MiMo streaming empty tool-call deltas')
else:
    print('[fix-mimo-v2-vllm] Qwen3XML/MiMo streaming empty-delta hardening already patched')
PY

# Optional MiMo reasoning budget default.  Some clients (including agent
# frontends) can enable reasoning but do not send vLLM's top-level
# `thinking_token_budget` request parameter.  Once the parser is correctly
# keeping generated <think> text in reasoning, MiMo may spend the full
# max_new_tokens budget in repetitive reasoning.  If
# MIMO_DEFAULT_THINKING_TOKEN_BUDGET is set, apply it only when the client did
# not provide an explicit budget.  This does not remove prior reasoning from
# the prompt; it only forces the current generation to close </think>.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py')
if not path.exists():
    print('[fix-mimo-v2-vllm] chat completion serving.py not present; skipping default thinking budget patch')
    raise SystemExit
text = path.read_text()
orig = text
if 'import os\n' not in text.split('import time\n', 1)[0] + 'import time\n':
    text = text.replace('import json\nimport time\n', 'import json\nimport os\nimport time\n', 1)
old = '''            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
'''
new = '''            default_thinking_token_budget = os.environ.get(
                "MIMO_DEFAULT_THINKING_TOKEN_BUDGET", ""
            )
            if (
                default_thinking_token_budget
                and request.thinking_token_budget is None
                and request.include_reasoning
                and reasoning_parser is not None
            ):
                try:
                    parsed_default_thinking_token_budget = int(
                        default_thinking_token_budget
                    )
                except ValueError as exc:
                    raise ValueError(
                        "MIMO_DEFAULT_THINKING_TOKEN_BUDGET must be a "
                        "non-negative integer"
                    ) from exc
                if parsed_default_thinking_token_budget < 0:
                    raise ValueError(
                        "MIMO_DEFAULT_THINKING_TOKEN_BUDGET must be a "
                        "non-negative integer"
                    )
                request.thinking_token_budget = parsed_default_thinking_token_budget

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: serving.py sampling params anchor not found for thinking budget default')
    text = text.replace(old, new, 1)
if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] patched OpenAI chat serving default MiMo thinking budget')
else:
    print('[fix-mimo-v2-vllm] OpenAI chat serving default MiMo thinking budget already patched')
PY

# Some current vLLM builds have the MiMo V2 model class but not a HF config
# registry entry for model_type=mimo_v2. Transformers then tries to fetch a
# nonexistent remote configuration_mimo_v2.py from the NVFP4 repo and aborts
# before vLLM can select its local model implementation.
echo "[fix-mimo-v2-vllm] Installing local MiMoV2Config registration if needed"
cat > "$SITE_PACKAGES/vllm/transformers_utils/configs/mimo_v2.py" <<'PY'
# SPDX-License-Identifier: Apache-2.0
from transformers import PretrainedConfig


class MimoV2Config(PretrainedConfig):
    model_type = "mimo_v2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
PY
python3 - <<'PY'
from pathlib import Path
site = Path('/usr/local/lib/python3.12/dist-packages')
init = site / 'vllm/transformers_utils/configs/__init__.py'
text = init.read_text()
if '"MimoV2Config": "vllm.transformers_utils.configs.mimo_v2"' not in text:
    text = text.replace(
        '    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",\n',
        '    "MiDashengLMConfig": "vllm.transformers_utils.configs.midashenglm",\n'
        '    "MimoV2Config": "vllm.transformers_utils.configs.mimo_v2",\n',
    )
if '    "MimoV2Config",\n' not in text:
    text = text.replace(
        '    "MiDashengLMConfig",\n',
        '    "MiDashengLMConfig",\n'
        '    "MimoV2Config",\n',
    )
init.write_text(text)

cfg = site / 'vllm/transformers_utils/config.py'
text = cfg.read_text()
if 'mimo_v2="MimoV2Config"' not in text:
    text = text.replace(
        '    midashenglm="MiDashengLMConfig",\n',
        '    midashenglm="MiDashengLMConfig",\n'
        '    mimo_v2="MimoV2Config",\n',
    )
cfg.write_text(text)
PY

# Respect an explicit text-only architecture override. Current vLLM's MiMoV2
# arch convertor unconditionally rewrites any config containing vision_config to
# MiMoV2OmniForCausalLM, even when the launch passes
# --hf-overrides '{"architectures":["MiMoV2ForCausalLM"]}'. That makes text-only
# bring-up load the vision/audio modules and currently fails on missing merger
# bias weights in this NVFP4 export.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/model_arch_config_convertor.py')
text = path.read_text()
old = '''class MimoV2ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def __init__(self, hf_config: PretrainedConfig, hf_text_config: PretrainedConfig):
        if getattr(hf_config, "vision_config", None):
            hf_config.architectures = ["MiMoV2OmniForCausalLM"]
        super().__init__(hf_config, hf_text_config)
        _strip_mimo_v2_attention_chunk_size(hf_config, hf_text_config)
'''
new = '''class MimoV2ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def __init__(self, hf_config: PretrainedConfig, hf_text_config: PretrainedConfig):
        # Preserve explicit text-only override for MiMo-V2.5 NVFP4 bring-up.
        # The checkpoint config includes vision/audio sections, but text-only
        # serving should use MiMoV2ForCausalLM when requested via hf_overrides.
        if getattr(hf_config, "vision_config", None) and getattr(
            hf_config, "architectures", None
        ) != ["MiMoV2ForCausalLM"]:
            hf_config.architectures = ["MiMoV2OmniForCausalLM"]
        super().__init__(hf_config, hf_text_config)
        _strip_mimo_v2_attention_chunk_size(hf_config, hf_text_config)
'''
if old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched MimoV2 arch convertor to preserve text-only override')
elif 'Preserve explicit text-only override for MiMo-V2.5 NVFP4 bring-up' in text:
    print('[fix-mimo-v2-vllm] MimoV2 arch convertor already patched')
else:
    raise SystemExit('[fix-mimo-v2-vllm] ERROR: MimoV2 arch convertor pattern not found')
PY

# Text-only MiMoV2ForCausalLM should ignore multimodal and MTP weights present
# in the Omni/NVFP4 checkpoint. AutoWeightsLoader otherwise treats e.g.
# visual.* as an unknown module and aborts.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py')
text = path.read_text()
old = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
'''
new = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def text_only_weights():
            skip_prefixes = (
                "visual.",
                "audio_encoder.",
                "speech_embeddings.",
                "model.mtp.",
            )
            for name, tensor in weights:
                if name.startswith(skip_prefixes):
                    continue
                yield name, tensor

        loader = AutoWeightsLoader(self)
        return loader.load_weights(text_only_weights())
'''
if old in text:
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched MiMoV2ForCausalLM.load_weights to skip non-text checkpoint tensors')
elif 'def text_only_weights():' in text and 'speech_embeddings.' in text:
    print('[fix-mimo-v2-vllm] MiMoV2 text-only weight filter already patched')
else:
    raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2 load_weights pattern not found')
PY

# Omni/multimodal fixes for this checkpoint:
# - The reference MiMo vision merger uses biased Linear layers and the NVFP4
#   checkpoint contains visual.merger.mlp.{0,2}.bias.  vLLM's local MiMo copy
#   had these as bias=False, causing unknown/missing bias handling and an
#   architecture mismatch.
# - Target Omni model loading should skip MTP weights; the MTP drafter loads
#   those separately.
# - The top-level Omni class is SupportsQuant, so it is the class that mutates
#   ModelOptMixedPrecisionConfig.  Without a packed_modules_mapping on Omni,
#   nested language_model.model.layers.*.mlp.gate_up_proj does not resolve the
#   checkpoint's separate gate_proj/up_proj quantization entries and is treated
#   as unquantized, corrupting the target text path even for text-only requests.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py')
text = path.read_text()
orig = text
old = '''class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant):
    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
'''
new = '''class MiMoV2OmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsQuant):
    # Ensure ModelOpt mixed-precision resolves fused language/MTP modules after
    # the Omni hf_to_vllm prefix mapper rewrites model.* -> language_model.model.*.
    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2Omni class anchor not found')
    text = text.replace(old, new, 1)
text = text.replace(
'''            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
''',
'''            ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.0",
''',
1)
text = text.replace(
'''            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
''',
'''            RowParallelLinear(
                self.hidden_size,
                d_model,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp.2",
''',
1)
old = '''        loader = AutoWeightsLoader(self, skip_prefixes=["audio_tokenizer."])
        auto_loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
new = '''        loader = AutoWeightsLoader(
            self,
            skip_prefixes=[
                "audio_tokenizer.",
                # After hf_to_vllm_mapper, checkpoint model.mtp.* becomes
                # language_model.model.mtp.*.  The target model should ignore
                # it; MiMoV2MTP/OmniMTP loads these weights separately.
                "language_model.model.mtp.",
            ],
        )
        auto_loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2Omni load_weights skip pattern not found')
    text = text.replace(old, new, 1)
if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] patched MiMoV2Omni vision merger bias and MTP skip')
else:
    print('[fix-mimo-v2-vllm] MiMoV2Omni multimodal fixes already patched')
PY

# MiMo-V2.5-NVFP4/chimera stores fused qkv_proj tensors as canonical
# [Q_all][K_all][V_all] (see checkpoint metadata: qkv-deinterleaved).  The
# upstream MiMoV2 loader has a Pro-format shortcut that blindly chunks the
# fused row dimension by TP rank; that is shape-correct but semantically wrong
# for this checkpoint because K/V slots receive Q rows.  Let QKVParallelLinear's
# native fused-QKV loader split Q/K/V and their MXFP8 scale rows correctly.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py')
text = path.read_text()
old = '''            # Support fused qkv_proj checkpoint (Pro format)
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    default_weight_loader(param, loaded_weight)
                continue
'''
new = '''            # MiMo-V2.5-NVFP4/chimera stores fused qkv_proj tensors as
            # canonical [Q_all][K_all][V_all] (checkpoint metadata says
            # qkv-deinterleaved), not the native FP8 Pro TP-prepacked layout
            # [Q0 K0 V0][Q1 K1 V1]...
            #
            # A blind chunk(tp_size, dim=0) is shape-correct for TP=2 but
            # semantically corrupts K/V rows and the row-aligned MXFP8
            # weight_scale_inv tensors.  Use QKVParallelLinear.weight_loader so
            # each rank receives [Q_rank][K_rank][V_rank].
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2 qkv_proj loader shortcut pattern not found')
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched MiMoV2 fused qkv_proj loader to use QKVParallelLinear.weight_loader')
else:
    print('[fix-mimo-v2-vllm] MiMoV2 fused qkv_proj loader already patched')
PY

# Apply the same qkv-deinterleaved handling to the MiMo-V2 MTP draft model.
# The NVFP4 checkpoint includes MTP qkv_proj tensors in the same canonical
# [Q_all][K_all][V_all] layout.  Also keep duplicate parameter aliases when
# building params_dict so `.weight_scale_inv` checkpoint tensors load into the
# MXFP8 `weight_scale` alias registered by fix-modelopt-mixed-mxfp8.
#
# Critical Omni+MTP quant fix: MiMoV2OmniForCausalLM's hf_to_vllm_mapper rewrites
# checkpoint quantized_layers from model.* to language_model.model.* so the Omni
# target layers quantize correctly.  That same global QuantizationConfig is then
# reused for the MTP drafter, whose prefixes are still model.mtp.layers.*.  Mirror
# language_model.model.mtp.* quant metadata back to model.mtp.* when the draft
# class is initialized, otherwise OmniMTP layers silently become unquantized.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_mtp.py')
text = path.read_text()
orig = text
old = '''from .utils import _merge_multimodal_embeddings, maybe_prefix
'''
new = '''from .utils import WeightsMapper, _merge_multimodal_embeddings, maybe_prefix
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP utils import pattern not found')
    text = text.replace(old, new, 1)

old = '''class MiMoV2MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
'''
new = '''class MiMoV2MTP(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["qkv_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Undo the Omni target mapper for MTP draft quant metadata only.
            "language_model.model.mtp.": "model.mtp.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP class anchor not found')
    text = text.replace(old, new, 1)

old = '''        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
'''
new = '''        # Keep duplicate aliases such as MXFP8 `weight_scale` /
        # `weight_scale_inv`; the checkpoint uses the latter.
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP params_dict pattern not found')
    text = text.replace(old, new, 1)

old = '''            # Support fused qkv_proj checkpoint (Pro format).
            # The checkpoint is stored pre-sharded for TP=8 as
            # [Q_rank0, K_rank0, V_rank0, Q_rank1, ...], so splitting along
            # dim 0 with chunk(tp_size) gives each rank its Q+K+V slice for
            # both the FP8 weight and the block weight_scale_inv. This matches
            # how the main model loads the same layout.
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    loaded_weight = loaded_weight.chunk(tp_size, dim=0)[tp_rank]
                    default_weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
'''
new = '''            # MiMo-V2.5-NVFP4/chimera MTP qkv_proj tensors are canonical
            # [Q_all][K_all][V_all], not TP-prepacked Pro layout.  Use
            # QKVParallelLinear.weight_loader for Q/K/V-aware TP slicing of
            # both FP8 weights and row-aligned MXFP8 weight_scale_inv tensors.
            if "qkv_proj" in name:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                continue
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: MiMoV2MTP qkv_proj loader shortcut pattern not found')
    text = text.replace(old, new, 1)

if text != orig:
    path.write_text(text)
    print('[fix-mimo-v2-vllm] patched MiMoV2MTP quant mapping and qkv-deinterleaved MXFP8 loader')
else:
    print('[fix-mimo-v2-vllm] MiMoV2MTP quant mapping and loader already patched')
PY

# MiMo-V2.5 Omni checkpoints may keep the raw HF architecture as
# MiMoV2ForCausalLM even though the target model is resolved to
# MiMoV2OmniForCausalLM because vision/audio config is present or because the
# launch passes an Omni hf override.  The speculative draft ModelConfig reloads
# the raw checkpoint config and applies only SpeculativeConfig.hf_config_override;
# without this patch the draft resolves to text-only MiMoV2MTPModel instead of
# the official multimodal MiMoV2OmniMTPModel wrapper.  Select OmniMTP whenever
# the draft checkpoint advertises vision_config.
python3 - <<'PY'
from pathlib import Path
path = Path('/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py')
text = path.read_text()
old = '''        if (arch := hf_config.architectures[0]) in (
            "MiMoV2ForCausalLM",
            "MiMoV2OmniForCausalLM",
        ):
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_PRO_NUM_MTP_LAYERS,
            )

            mtp_arch_maps = {
                "MiMoV2ForCausalLM": "MiMoV2MTPModel",
                "MiMoV2OmniForCausalLM": "MiMoV2OmniMTPModel",
            }

            hf_config.model_type = "mimo_v2_mtp"
'''
new = '''        if (arch := hf_config.architectures[0]) in (
            "MiMoV2ForCausalLM",
            "MiMoV2OmniForCausalLM",
        ):
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_PRO_NUM_MTP_LAYERS,
            )

            # The raw HF config for some Omni-capable MiMo-V2.5 exports still
            # says MiMoV2ForCausalLM even though vision/audio config is present.
            # The target may be resolved to MiMoV2OmniForCausalLM, but the MTP
            # draft config only sees the raw checkpoint architecture.  Mirror
            # the official Omni MTP path for such checkpoints.
            if arch == "MiMoV2ForCausalLM" and getattr(
                hf_config, "vision_config", None
            ):
                arch = "MiMoV2OmniForCausalLM"

            mtp_arch_maps = {
                "MiMoV2ForCausalLM": "MiMoV2MTPModel",
                "MiMoV2OmniForCausalLM": "MiMoV2OmniMTPModel",
            }

            hf_config.model_type = "mimo_v2_mtp"
'''
if new not in text:
    if old not in text:
        raise SystemExit('[fix-mimo-v2-vllm] ERROR: SpeculativeConfig MiMoV2 MTP mapping pattern not found')
    path.write_text(text.replace(old, new, 1))
    print('[fix-mimo-v2-vllm] patched SpeculativeConfig to select MiMoV2OmniMTPModel for Omni exports')
else:
    print('[fix-mimo-v2-vllm] SpeculativeConfig Omni MTP mapping already patched')
PY

# PR #41797: add TRITON_ATTN_DIFFKV and make MiMoV2 auto-fallback to it on
# non-FA3 hardware (GB10/sm_121a). Without this, MiMoV2's K/V head-dim split
# forces FlashAttentionDiffKV and fails on DGX Spark.
if python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('vllm.v1.attention.backends.triton_attn_diffkv') else 1)
PY
then
    echo "[fix-mimo-v2-vllm] TRITON_ATTN_DIFFKV already present; skipping PR #41797"
else
    echo "[fix-mimo-v2-vllm] Applying vLLM PR #41797 (TRITON_ATTN_DIFFKV)"
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT
    curl -fsL "$PR41797_URL" -o "$tmpdir/pr41797.diff"

    # The upstream diff contains docs; only apply package files under vllm/.
    python3 - "$tmpdir/pr41797.diff" "$tmpdir/pr41797-vllm-only.diff" <<'PY'
from pathlib import Path
import sys
src = Path(sys.argv[1])
dst = Path(sys.argv[2])
keep = False
out = []
for line in src.read_text().splitlines(True):
    if line.startswith('diff --git '):
        parts = line.strip().split()
        # format: diff --git a/path b/path
        b_path = parts[3][2:] if len(parts) >= 4 and parts[3].startswith('b/') else ''
        keep = b_path.startswith('vllm/')
    if keep:
        out.append(line)
dst.write_text(''.join(out))
PY

    if git apply --check --unsafe-paths "$tmpdir/pr41797-vllm-only.diff"; then
        git apply --unsafe-paths --whitespace=nowarn "$tmpdir/pr41797-vllm-only.diff"
    elif patch -p1 --dry-run --forward --batch < "$tmpdir/pr41797-vllm-only.diff" >/dev/null 2>&1; then
        patch -p1 --forward --batch < "$tmpdir/pr41797-vllm-only.diff"
    else
        echo "[fix-mimo-v2-vllm] ERROR: PR #41797 is not applicable to this vLLM install" >&2
        echo "[fix-mimo-v2-vllm] Rebuild with --apply-vllm-pr 41797 or update the base image." >&2
        exit 1
    fi
fi

# CyberTen's #41834 minimal fallback for V2 executor + MTP + cudagraph.
# Newer vLLM already contains this; older builds may not. Apply only when the
# exact anchor exists and the snippet is absent.
GPU_RUNNER="$SITE_PACKAGES/vllm/v1/worker/gpu_model_runner.py"
if [ -f "$GPU_RUNNER" ] && ! grep -q "sync_without_prev_positions" "$GPU_RUNNER"; then
    echo "[fix-mimo-v2-vllm] Applying minimal sync_without_prev_positions fallback"
    python3 - "$GPU_RUNNER" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
anchors = [
    "        prev_positions = self.prev_positions.np[: len(num_draft_tokens)]\n",
    "        prev_positions = self.prev_positions.np[:len(num_draft_tokens)]\n",
]
snippet = '''\
        sync_without_prev_positions = (\n            not self.use_async_scheduling and np.all(prev_positions < 0)\n        )\n        if sync_without_prev_positions:\n            if draft_probs.ndim == 2:\n                return draft_probs[:total_num_draft_tokens].contiguous()\n            if draft_probs.shape[0] >= len(num_draft_tokens):\n                prev_positions = np.arange(len(num_draft_tokens))\n            else:\n                packed_probs = []\n                draft_row = 0\n                for num_tokens in num_draft_tokens:\n                    if num_tokens == 0:\n                        continue\n                    if draft_row >= draft_probs.shape[0]:\n                        raise RuntimeError(\n                            "Spec decode metadata references more draft token "\n                            "rows than were recorded by the draft model."\n                        )\n                    packed_probs.append(draft_probs[draft_row, :num_tokens])\n                    draft_row += 1\n                if not packed_probs:\n                    return None\n                return torch.cat(packed_probs, dim=0).contiguous()\n'''
for anchor in anchors:
    if anchor in text:
        path.write_text(text.replace(anchor, anchor + snippet, 1))
        print("[fix-mimo-v2-vllm] patched", path)
        break
else:
    print("[fix-mimo-v2-vllm] anchor not found; skipping #41834 fallback (likely different vLLM version)")
PY
else
    echo "[fix-mimo-v2-vllm] sync_without_prev_positions already present or gpu_model_runner.py missing; skipping"
fi

find "$SITE_PACKAGES/vllm" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

python3 - <<'PY'
import importlib.util
assert importlib.util.find_spec('vllm.v1.attention.backends.triton_attn_diffkv'), 'TRITON_ATTN_DIFFKV not installed'
from vllm.reasoning import ReasoningParserManager
assert ReasoningParserManager.lazy_parsers.get('mimo') == (
    'vllm.reasoning.mimo_reasoning_parser',
    'MimoReasoningParser',
), ReasoningParserManager.lazy_parsers.get('mimo')
assert importlib.util.find_spec('vllm.reasoning.mimo_reasoning_parser'), 'MiMo reasoning parser not installed'
from vllm.transformers_utils.config import _CONFIG_REGISTRY
assert 'mimo_v2' in _CONFIG_REGISTRY, 'mimo_v2 config registry entry missing'
from vllm.config.speculative import SpeculativeConfig
from vllm.transformers_utils.configs.mimo_v2 import MimoV2Config
cfg = MimoV2Config(architectures=['MiMoV2ForCausalLM'], vision_config={'enabled': True})
out = SpeculativeConfig.hf_config_override(cfg)
assert out.architectures == ['MiMoV2OmniMTPModel'], out.architectures
from vllm.model_executor.models.mimo_v2_mtp import MiMoV2MTP
mapped = MiMoV2MTP.hf_to_vllm_mapper.apply_dict({
    'language_model.model.mtp.layers.0.self_attn.qkv_proj': {'quant_algo': 'MXFP8'},
    'language_model.model.layers.0.self_attn.qkv_proj': {'quant_algo': 'MXFP8'},
})
assert 'model.mtp.layers.0.self_attn.qkv_proj' in mapped, mapped
assert 'language_model.model.layers.0.self_attn.qkv_proj' in mapped, mapped
assert MiMoV2MTP.packed_modules_mapping['gate_up_proj'] == ['gate_proj', 'up_proj']
print('[fix-mimo-v2-vllm] validation OK')
PY
