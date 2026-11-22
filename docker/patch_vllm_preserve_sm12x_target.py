#!/usr/bin/env python3
"""Preserve explicitly selected CUDA 13 Blackwell subarchitecture targets."""

import os
import sys
from pathlib import Path


if os.environ.get("VLLM_PRESERVE_SM12X_TARGET", "") not in {
    "1",
    "true",
    "TRUE",
    "yes",
    "YES",
}:
    print("Blackwell target preservation not requested; skipping")
    raise SystemExit(0)

source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "CMakeLists.txt"
cuda13_default = (
    'set(CUDA_SUPPORTED_ARCHS '
    '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")'
)
cuda13_sm121 = (
    'set(CUDA_SUPPORTED_ARCHS '
    '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")'
)
cuda13_subarchitectures = (
    'set(CUDA_SUPPORTED_ARCHS '
    '"7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1")'
)

text = target.read_text()
if cuda13_subarchitectures in text:
    print("CUDA 13 Blackwell subarchitecture allow-list already present; skipping")
elif text.count(cuda13_sm121) == 1:
    target.write_text(text.replace(cuda13_sm121, cuda13_subarchitectures, 1))
    print("Enabled selected SM103 and SM12x targets for CUDA 13 vLLM build")
elif text.count(cuda13_default) == 1:
    target.write_text(text.replace(cuda13_default, cuda13_subarchitectures, 1))
    print("Enabled selected SM103 and SM12x targets for CUDA 13 vLLM build")
else:
    raise SystemExit(
        "Unable to preserve the selected Blackwell target: expected CUDA 13 "
        "CUDA_SUPPORTED_ARCHS entry was not found exactly once"
    )

# A family match in vLLM's selector consumes only one requested GPU. Resolve
# each device as a single-target build would, keeping the kernel's own support
# list, PTX policy and exact-match preference on each call.
if len(os.environ.get("TORCH_CUDA_ARCH_LIST", "").replace(";", " ").split()) > 1:
    helper = source_root / "cmake/utils.cmake"
    signature = "function(cuda_archs_loose_intersection OUT_CUDA_ARCHS SRC_CUDA_ARCHS TGT_CUDA_ARCHS)\n"
    per_device = """  list(LENGTH TGT_CUDA_ARCHS _target_count)
  if(_target_count GREATER 1)
    set(_target_union)
    foreach(_device IN LISTS TGT_CUDA_ARCHS)
      cuda_archs_loose_intersection(_device_archs "${SRC_CUDA_ARCHS}" "${_device}")
      list(APPEND _target_union ${_device_archs})
    endforeach()
    list(REMOVE_DUPLICATES _target_union)
    set(${OUT_CUDA_ARCHS} "${_target_union}" PARENT_SCOPE)
    return()
  endif()
"""
    content = helper.read_text()
    if signature + per_device not in content:
        if content.count(signature) != 1:
            raise SystemExit("Expected one cuda_archs_loose_intersection function in cmake/utils.cmake")
        helper.write_text(content.replace(signature, signature + per_device, 1))
