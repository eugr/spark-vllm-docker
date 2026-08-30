#!/usr/bin/env python3
"""Backport vLLM's XGrammar speculative-decoding fixes.

The base image pins vLLM ``487ecf187``, whose XGrammar backend can continue
feeding tokens to a matcher after an EOS/stop token terminates it.  That is
especially visible with a multi-token speculative batch: the first trailing
token is rejected, subsequent advances can warn, and the cached termination
flag can become inconsistent with the matcher.

This applies two source-exact behavioral backports:

* vLLM PR #52805, merged as commit
  ``12f64b39d29282437e35be9aa5db432fb2a1a6e6``, stops token batches at
  grammar termination.
* vLLM PR #53046, merged as commit
  ``c6e19b3be24338759a443e03c8325d76da9ee202``, validates speculative
  drafts produced before a mid-window reasoning-end marker before advancing
  the grammar.  This avoids a spurious ``Failed to advance FSM`` error when
  such a draft is invalid under the newly active grammar.

https://github.com/vllm-project/vllm/pull/52805
https://github.com/vllm-project/vllm/commit/12f64b39d29282437e35be9aa5db432fb2a1a6e6
https://github.com/vllm-project/vllm/pull/53046
https://github.com/vllm-project/vllm/commit/c6e19b3be24338759a443e03c8325d76da9ee202

Only the exact upstream behavior is changed.  The patch is idempotent and
preflights both files before writing, failing closed if a pinned anchor drifts
or a partial patch is found.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


BACKEND_TARGET = Path(
    os.environ.get(
        "GLM53_XGRAMMAR_BACKEND_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
        "backend_xgrammar.py",
    )
)
MANAGER_TARGET = Path(
    os.environ.get(
        "GLM53_XGRAMMAR_MANAGER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/"
        "__init__.py",
    )
)
BACKEND_MARK = (
    "    # [glm53-xgrammar-termination] Source-exact vLLM 12f64b39 backport.\n"
)
MANAGER_MARK = (
    "                    # [glm53-xgrammar-reasoning] Source-exact vLLM "
    "c6e19b3 backport.\n"
)

ACCEPT_OLD = '''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True
'''

ACCEPT_UPSTREAM = '''    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        """Accepts a list of tokens and advances the FSM.

        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored. Returns False if the FSM
        failed to advance.
        """
        if self._is_terminated:
            return True
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            self._is_terminated = self.matcher.is_terminated()
            if self._is_terminated:
                break
        return True
'''

VALIDATE_OLD = '''    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''

VALIDATE_UPSTREAM = '''    def validate_tokens(self, tokens: list[int]) -> list[int]:
        """Checks if the list of tokens are accepted by the FSM in sequence.
        Will not advance the FSM.

        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if self._is_terminated:
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if self.matcher.is_terminated():
                    break
            else:
                break
        if len(accepted_tokens) > 0:
            # Rollback the FSM to the initial state
            self.matcher.rollback(len(accepted_tokens))
        return accepted_tokens
'''

RESET_OLD = '''    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
'''

RESET_UPSTREAM = '''    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        self._is_terminated = False
'''

MANAGER_OLD = '''                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )
'''

MANAGER_UPSTREAM = '''                    if advance_grammar and not grammar.is_terminated():
                        if post_reasoning_end_in_window:
                            accepted = bool(grammar.validate_tokens([token]))
                            if accepted:
                                accepted = grammar.accept_tokens(req_id, [token])
                        else:
                            accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )
'''


def backend_counts(text: str) -> tuple[list[int], list[int]]:
    old = [text.count(x) for x in (ACCEPT_OLD, VALIDATE_OLD, RESET_OLD)]
    new = [
        text.count(ACCEPT_UPSTREAM),
        text.count(VALIDATE_UPSTREAM),
        text.count(RESET_UPSTREAM),
    ]
    return old, new


def verified_backend_state(text: str) -> bool:
    old, new = backend_counts(text)
    return old == [0, 0, 0] and new == [1, 1, 1]


def verified_manager_state(text: str) -> bool:
    return text.count(MANAGER_OLD) == 0 and text.count(MANAGER_UPSTREAM) == 1


def prepare_backend(source: str) -> tuple[str, str]:
    old, new = backend_counts(source)
    marker_count = source.count(BACKEND_MARK)
    if marker_count:
        if marker_count != 1 or not verified_backend_state(source):
            raise ValueError(
                "partial/inconsistent xgrammar termination patch "
                f"(marker={marker_count}, old={old}, new={new})"
            )
        return source, "already present"
    if verified_backend_state(source):
        return source, "already upstream"
    if old != [1, 1, 1] or new != [0, 0, 0]:
        raise ValueError(
            "pinned xgrammar termination anchors drifted "
            f"(old={old}, new={new})"
        )
    patched = source.replace(ACCEPT_OLD, BACKEND_MARK + ACCEPT_UPSTREAM, 1)
    patched = patched.replace(VALIDATE_OLD, VALIDATE_UPSTREAM, 1)
    patched = patched.replace(RESET_OLD, RESET_UPSTREAM, 1)
    if not verified_backend_state(patched) or patched.count(BACKEND_MARK) != 1:
        raise ValueError("xgrammar termination post-patch verification failed")
    return patched, "patched"


def prepare_manager(source: str) -> tuple[str, str]:
    old = source.count(MANAGER_OLD)
    new = source.count(MANAGER_UPSTREAM)
    marker_count = source.count(MANAGER_MARK)
    if marker_count:
        if marker_count != 1 or not verified_manager_state(source):
            raise ValueError(
                "partial/inconsistent xgrammar reasoning patch "
                f"(marker={marker_count}, old={old}, new={new})"
            )
        return source, "already present"
    if verified_manager_state(source):
        return source, "already upstream"
    if old != 1 or new != 0:
        raise ValueError(
            "pinned xgrammar reasoning anchor drifted "
            f"(old={old}, new={new})"
        )
    patched = source.replace(MANAGER_OLD, MANAGER_MARK + MANAGER_UPSTREAM, 1)
    if not verified_manager_state(patched) or patched.count(MANAGER_MARK) != 1:
        raise ValueError("xgrammar reasoning post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-xgrammar.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    for target in (BACKEND_TARGET, MANAGER_TARGET):
        if not target.is_file():
            raise SystemExit(f"missing {target}")

    backend_source = BACKEND_TARGET.read_text()
    manager_source = MANAGER_TARGET.read_text()
    try:
        backend_patched, backend_action = prepare_backend(backend_source)
        manager_patched, manager_action = prepare_manager(manager_source)
    except ValueError as exc:
        raise SystemExit(f"xgrammar patch preflight failed: {exc}") from exc

    compile(backend_patched, str(BACKEND_TARGET), "exec")
    compile(manager_patched, str(MANAGER_TARGET), "exec")

    if backend_patched != backend_source:
        replace_file(BACKEND_TARGET, backend_patched)
    if manager_patched != manager_source:
        replace_file(MANAGER_TARGET, manager_patched)

    print(f"{BACKEND_TARGET.name}: termination fix {backend_action} (#52805)")
    print(f"{MANAGER_TARGET.name}: reasoning fix {manager_action} (#53046)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
