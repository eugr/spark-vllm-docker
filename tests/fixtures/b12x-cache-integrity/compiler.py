# Extracted from local-inference-lab/b12x at
# 4f3028b19c1d8290dc72b6f483aba40de23eae5a for CPU-only patch regression tests.
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from contextvars import ContextVar
from threading import RLock
from types import SimpleNamespace
from typing import Any

from .compile_plan import _RETAINED_PROGRAMS
from .runtime_patches import apply_cutlass_runtime_patches
from .program_cache import register_program_cache


def _cache_prefix(cache_key: str) -> str:
    return f"b12x_cute_{cache_key}"


def _cache_object_path(cache_key: str) -> Path:
    return _cute_compile_cache_dir() / cache_key[:2] / f"{cache_key}.o"


def _cache_manifest_path(cache_key: str) -> Path:
    return _cache_object_path(cache_key).with_suffix(".json")


def _cache_lock_path(cache_key: str) -> Path:
    return _cache_object_path(cache_key).with_suffix(".lock")


def _write_compile_manifest(
    cache_key: str,
    cache_payload: tuple[object, ...],
    func: Any,
    object_bytes: bytes,
    compiled: Any = None,
) -> None:
    manifest_path = _cache_manifest_path(cache_key)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_compile_manifest(
        cache_key, cache_payload, func, object_bytes, compiled=compiled
    )
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest_path.parent,
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_name = tmp_file.name
            json.dump(
                manifest,
                tmp_file,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            tmp_file.write("\n")
        os.replace(tmp_name, manifest_path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            with suppress(OSError):
                os.unlink(tmp_name)


def _ensure_cute_compile_manifest(
    cache_key: str,
    cache_payload: tuple[object, ...],
    func: Any,
) -> None:
    manifest_path = _cache_manifest_path(cache_key)
    if manifest_path.exists():
        return
    object_bytes = _cache_object_path(cache_key).read_bytes()
    _write_compile_manifest(cache_key, cache_payload, func, object_bytes)


@contextmanager
def _disk_cache_key_lock(cache_key: str):
    try:
        import fcntl
    except ImportError:
        yield
        return

    lock_path = _cache_lock_path(cache_key)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_cute_compile_from_disk(cache_key: str):
    from cutlass.base_dsl.export.external_binary_module import ExternalBinaryModule

    object_path = _cache_object_path(cache_key)
    if not object_path.exists():
        return None
    try:
        # CUTLASS may finalize or patch the ELF while loading it.  The cache
        # object is content-addressed and its digest is recorded in the compile
        # manifest, so never expose that canonical object to the loader.
        with tempfile.TemporaryDirectory(
            prefix="b12x-cute-cache-load-"
        ) as raw_stage:
            staged_object = Path(raw_stage) / object_path.name
            shutil.copy2(object_path, staged_object)
            module = ExternalBinaryModule(str(staged_object))
            return getattr(module, _cache_prefix(cache_key))
    except Exception:
        return None


def _store_cute_compile_to_disk(
    cache_key: str,
    compiled: Any,
    *,
    cache_payload: tuple[object, ...] | None = None,
    func: Any = None,
) -> None:
    if not hasattr(compiled, "dump_to_object"):
        return

    object_path = _cache_object_path(cache_key)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = object_path.with_suffix(".tmp")
    object_bytes = compiled.dump_to_object(_cache_prefix(cache_key))
    with open(tmp_path, "wb") as f:
        f.write(object_bytes)
    os.replace(tmp_path, object_path)
    if cache_payload is not None and func is not None:
        _write_compile_manifest(
            cache_key, cache_payload, func, object_bytes, compiled=compiled
        )


def compile(
    func: Any,
    *args: Any,
    compile_spec: KernelCompileSpec | None = None,
    dsl_compile_options: Any = None,
    **kwargs: Any,
) -> Any:
    import cutlass.cute as cute
    from .compile_plan import (
        DeferredCuTeKernel,
        ProgramKey,
        planning,
        record_program,
        tag_compiled,
    )

    global _DISK_CACHE_HITS
    global _COMPILE_MISSES
    compile_callable = cute.compile
    if dsl_compile_options is not None:
        # Subscript-style DSL compile options (e.g. OptLevel(2): ptxas -O3's
        # scheduler register-starves some scalar-heavy kernels; see the w4a8
        # dynamic MoE recipe).
        if hasattr(compile_callable, "__getitem__"):
            compile_callable = compile_callable[dsl_compile_options]
        else:
            # Some embedded runtimes expose cutlass.cute.compile as a plain
            # function instead of the CompileCallable instance installed by the
            # top-level module import.  Recreate the callable explicitly so DSL
            # options still take effect instead of crashing or silently falling
            # back to the default compiler options.
            from cutlass.base_dsl.compiler import CompileCallable

            compile_callable = CompileCallable(dsl_compile_options)
        kwargs = dict(kwargs)
        kwargs["__dsl_compile_options_key"] = _dsl_compile_options_kwargs_key(
            compile_callable
        )
    memory_cache_key = _compile_memory_cache_key(
        compile_callable, func, args, kwargs, compile_spec
    )
    if planning():
        from .runtime_control import raise_if_kernel_resolution_frozen

        raise_if_kernel_resolution_frozen("CuTe compilation planning", target=func)
        payload = _compile_disk_cache_payload(
            compile_callable, func, args, kwargs, compile_spec
        )
        if not _cute_compile_disk_cache_enabled_for_payload(payload):
            raise RuntimeError("compilation planning requires the normal device-bound object cache")
        cache_key = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
        program = ProgramKey(
            "cute", cache_key,
            compile_spec.kernel_id if compile_spec is not None else _compile_target_name(func),
        )
        record_program(program)
        return DeferredCuTeKernel(program, memory_cache_key)
    compiled = _memory_cache_get(memory_cache_key)
    if compiled is not None:
        for program in getattr(compiled, "__b12x_programs__", ()):
            record_program(program)
        return compiled
    from b12x._lib.runtime_control import (
        raise_if_kernel_resolution_frozen,
    )

    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=func,
        cache_key=compile_spec if compile_spec is not None else memory_cache_key,
    )

    post_engine_start_log = _cute_compile_post_engine_start_log_enabled()
    payload = _compile_disk_cache_payload(
        compile_callable, func, args, kwargs, compile_spec
    )
    cache_key = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    program = ProgramKey(
        "cute", cache_key,
        compile_spec.kernel_id if compile_spec is not None else _compile_target_name(func),
    )
    disk_cache_enabled = _cute_compile_disk_cache_enabled_for_payload(payload)

    if disk_cache_enabled:
        compiled = _load_cute_compile_from_disk(cache_key)
        if compiled is not None:
            compiled = tag_compiled(compiled, program)
            with suppress(Exception):
                _ensure_cute_compile_manifest(cache_key, payload, func)
            with _MEMORY_CACHE_LOCK:
                _DISK_CACHE_HITS += 1
            if post_engine_start_log:
                with suppress(Exception):
                    _log_cute_compile_event(
                        func,
                        args,
                        kwargs,
                        event="disk-hit",
                        cache_status="disk-cache-hit",
                        cache_payload=payload,
                        reason="post-engine-start",
                        cache_key=cache_key,
                    )
            _memory_cache_put(memory_cache_key, compiled)
            return compiled

        with _disk_cache_key_lock(cache_key):
            compiled = _memory_cache_get(memory_cache_key)
            if compiled is not None:
                for existing_program in getattr(compiled, "__b12x_programs__", ()):
                    record_program(existing_program)
                return compiled

            compiled = _load_cute_compile_from_disk(cache_key)
            if compiled is not None:
                compiled = tag_compiled(compiled, program)
                with suppress(Exception):
                    _ensure_cute_compile_manifest(cache_key, payload, func)
                with _MEMORY_CACHE_LOCK:
                    _DISK_CACHE_HITS += 1
                if post_engine_start_log:
                    with suppress(Exception):
                        _log_cute_compile_event(
                            func,
                            args,
                            kwargs,
                            event="disk-hit-after-wait",
                            cache_status="disk-cache-hit-after-wait",
                            cache_payload=payload,
                            reason="post-engine-start",
                            cache_key=cache_key,
                        )
                _memory_cache_put(memory_cache_key, compiled)
                return compiled

            cache_status = "disk-cache-miss"

            if _cute_compile_log_enabled() or post_engine_start_log:
                with suppress(Exception):
                    _log_cute_compile_miss(
                        func,
                        args,
                        kwargs,
                        cache_status=cache_status,
                        cache_payload=payload,
                        reason="post-engine-start" if post_engine_start_log else None,
                        cache_key=cache_key,
                    )

            with _MEMORY_CACHE_LOCK:
                _COMPILE_MISSES += 1
            call_kwargs = {
                k: v for k, v in kwargs.items() if k != "__dsl_compile_options_key"
            }
            compiled = _call_cute_compile(
                compile_callable,
                func,
                args,
                call_kwargs,
                compile_spec=compile_spec,
                cache_key=cache_key,
            )
            store_context = nullcontext() if _OFFLINE_CUTE_NO_JIT else suppress(Exception)
            with store_context:
                _store_cute_compile_to_disk(
                    cache_key,
                    compiled,
                    cache_payload=payload,
                    func=func,
                )
            compiled = tag_compiled(compiled, program)
            _memory_cache_put(memory_cache_key, compiled)
            return compiled
    else:
        cache_status = (
            "disk-cache-device-uuid-unavailable"
            if _cute_compile_disk_cache_enabled()
            else "disk-cache-disabled"
        )

    if _cute_compile_log_enabled() or post_engine_start_log:
        with suppress(Exception):
            _log_cute_compile_miss(
                func,
                args,
                kwargs,
                cache_status=cache_status,
                cache_payload=payload,
                reason="post-engine-start" if post_engine_start_log else None,
                cache_key=cache_key,
            )

    with _MEMORY_CACHE_LOCK:
        _COMPILE_MISSES += 1
    call_kwargs = {k: v for k, v in kwargs.items() if k != "__dsl_compile_options_key"}
    compiled = _call_cute_compile(
        compile_callable,
        func,
        args,
        call_kwargs,
        compile_spec=compile_spec,
        cache_key=cache_key,
    )
    if disk_cache_enabled:
        store_context = nullcontext() if _OFFLINE_CUTE_NO_JIT else suppress(Exception)
        with store_context:
            _store_cute_compile_to_disk(
                cache_key,
                compiled,
                cache_payload=payload,
                func=func,
            )
    compiled = tag_compiled(compiled, program)
    _memory_cache_put(memory_cache_key, compiled)
    return compiled


