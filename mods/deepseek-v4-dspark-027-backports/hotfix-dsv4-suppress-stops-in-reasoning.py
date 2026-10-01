#!/usr/bin/env python3
"""Keep client stop strings dormant until </think> (Anemll 0.1.1).

Port of tonyd2wild Patch 5 / Capicua25x
``0005-suppress-stops-in-reasoning.patch`` onto the Anemll image path:

  /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py

Do not bind-mount Tony's Stage-C file (``/opt/env/...``).

vLLM v1 matches ``stop`` against the whole stream. Think-in-prompt starts
inside ``<think>``; CoT often restates harness stops like ``Question:``.
The request then finishes mid-reason with ``content: null``.

Guard arms only when the last prompt token is the reasoning start marker.
EOS / max_tokens are unchanged. Spec-decode: when ``</think>`` arrives in
the same k+1 chunk as prior think tokens, ``stop_check_offset`` jumps past
the marker so a stop in that tail cannot fire.

Runtime opt-out (process-wide): ``DSPARK_SUPPRESS_STOPS_IN_REASONING=0``
or Tony's ``VLLM_SUPPRESS_STOPS_IN_REASONING=0``.
Skip applying this file: ``DSPARK_SKIP_SUPPRESS_STOPS_HOTFIX=1``.
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py")
MARK = "# [suppress-stops-in-reasoning]"

IMPORT_OLD = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "from abc import ABC, abstractmethod\n"
)
IMPORT_NEW = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "import os\n"
    "from abc import ABC, abstractmethod\n"
)

FACTORY_OLD = '''        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)
'''

FACTORY_NEW = '''        if USE_FAST_DETOKENIZER and isinstance(tokenizer, PreTrainedTokenizerFast):
            # Fast tokenizer => use tokenizers library DecodeStream.
            detok = FastIncrementalDetokenizer(tokenizer, request)
        else:
            # Fall back to slow python-based incremental detokenization.
            detok = SlowIncrementalDetokenizer(tokenizer, request)
        IncrementalDetokenizer._maybe_enable_reasoning_stop_guard(
            detok, tokenizer, request
        )
        return detok

    @staticmethod
    def _reasoning_stop_markers() -> tuple[str, str]:
        # [suppress-stops-in-reasoning] prefer --reasoning-config markers.
        start, end = "<think>", "</think>"
        try:
            from vllm.config import get_current_vllm_config_or_none

            cfg = get_current_vllm_config_or_none()
            rc = getattr(cfg, "reasoning_config", None) if cfg else None
            if rc is not None:
                start = getattr(rc, "reasoning_start_str", "") or start
                end = getattr(rc, "reasoning_end_str", "") or end
        except Exception as e:
            logger.debug(
                "suppress-stops-in-reasoning: no reasoning config (%s); using %r / %r",
                e,
                start,
                end,
            )
        return start, end

    @staticmethod
    def _suppress_stops_enabled() -> bool:
        for key in (
            "DSPARK_SUPPRESS_STOPS_IN_REASONING",
            "VLLM_SUPPRESS_STOPS_IN_REASONING",
        ):
            if key in os.environ:
                return os.environ.get(key) != "0"
        return True

    @staticmethod
    def _maybe_enable_reasoning_stop_guard(detok, tokenizer, request) -> None:
        # [suppress-stops-in-reasoning] last prompt token == <think> →
        # keep client stop strings dormant until </think>.
        if not IncrementalDetokenizer._suppress_stops_enabled():
            return
        try:
            stop = getattr(detok, "stop", None)
            ptids = getattr(request, "prompt_token_ids", None)
            if not stop or not ptids:
                return
            start_str, end_str = IncrementalDetokenizer._reasoning_stop_markers()
            think_id = None
            convert = getattr(tokenizer, "convert_tokens_to_ids", None)
            if callable(convert):
                think_id = convert(start_str)
            if think_id is None or think_id < 0:
                encode = getattr(tokenizer, "encode", None)
                if callable(encode):
                    try:
                        ids = encode(start_str, add_special_tokens=False)
                    except TypeError:
                        ids = encode(start_str)
                    if isinstance(ids, list) and len(ids) == 1:
                        think_id = ids[0]
            if think_id is not None and think_id >= 0 and ptids[-1] == think_id:
                detok._reasoning_stop_guard = True
                detok._reasoning_end_str = end_str
        except Exception as e:
            logger.debug("suppress-stops-in-reasoning: guard not armed (%s)", e)
'''

INIT_OLD = """        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = ""
"""

INIT_NEW = """        self._last_output_text_offset: int = 0

        # [suppress-stops-in-reasoning] client stops stay dormant while
        # generation is still inside <think> (think-in-prompt).
        self._reasoning_stop_guard: bool = False
        self._reasoning_closed: bool = False
        self._reasoning_end_str: str = "</think>"

        # Generation data
        self.output_text = ""
"""

STOP_OLD = """        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            stop = check_stop_strings(
"""

STOP_NEW = """        # 2) Evaluate stop strings.
        # [suppress-stops-in-reasoning] keep stops dormant while reasoning
        # is open. Advance stop_check_offset past </think> so a speculative
        # chunk that also carries the last think tokens cannot fire.
        if self._reasoning_stop_guard and not self._reasoning_closed:
            marker = self._reasoning_end_str
            window = max(0, stop_check_offset - (len(marker) - 1))
            idx = self.output_text.find(marker, window)
            if idx != -1:
                self._reasoning_closed = True
                stop_check_offset = max(stop_check_offset, idx + len(marker))
        stop_string = None
        if (
            self.stop
            and self.num_output_tokens() > self.min_tokens
            and (not self._reasoning_stop_guard or self._reasoning_closed)
        ):
            stop = check_stop_strings(
"""


def apply_text(src: str) -> tuple[str, str]:
    """Return (new_source, status): applied|skipped|missing."""
    if MARK in src and "_maybe_enable_reasoning_stop_guard" in src:
        return src, "skipped"
    missing = []
    if IMPORT_OLD not in src:
        missing.append("import")
    if FACTORY_OLD not in src:
        missing.append("factory")
    if INIT_OLD not in src:
        missing.append("init")
    if STOP_OLD not in src:
        missing.append("stop")
    if missing:
        return src, "missing:" + ",".join(missing)
    out = src.replace(IMPORT_OLD, IMPORT_NEW, 1)
    out = out.replace(FACTORY_OLD, FACTORY_NEW, 1)
    out = out.replace(INIT_OLD, INIT_NEW, 1)
    out = out.replace(STOP_OLD, STOP_NEW, 1)
    return out, "applied"


def apply_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    new, status = apply_text(text)
    if status == "applied":
        path.write_text(new, encoding="utf-8")
    return status


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else P
    if not target.is_file():
        print(f"[suppress-stops-in-reasoning] missing {target}", file=sys.stderr)
        return 0
    status = apply_file(target)
    print(f"[suppress-stops-in-reasoning] {status}: {target}")
    return 0 if status.startswith("applied") or status == "skipped" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
