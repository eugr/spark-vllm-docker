# Extracted from local-inference-lab/b12x at
# 4f3028b19c1d8290dc72b6f483aba40de23eae5a for CPU-only patch regression tests.
"""Plan production compilation by real cache identity, without compiling it.

Compile-only factories may return deferred kernels during discovery. They must
return their compilation carriers, and host launcher closures explicitly retain
those carriers through ``attach_programs``. No closure/source introspection or
alternative persistent cache format is involved.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


def compiled_program_available(program: ProgramKey) -> bool:
    """Check an executable or actual artifact, never an unresolved factory."""
    if program in _RESIDENT_PROGRAMS:
        return True
    if program.dialect == "cute":
        from .compiler import _cache_object_path
        return _cache_object_path(program.key).is_file()
    if program.dialect == "triton":
        from triton.compiler.compiler import get_cache_manager
        if not program.name:
            raise ValueError("Triton availability requires its current descriptor name")
        group = get_cache_manager(program.key).get_group(f"{program.name[:150]}.json")
        return bool(group) and all(Path(path).is_file() for path in group.values())
    raise ValueError(f"unsupported compiler dialect {program.dialect!r}")


