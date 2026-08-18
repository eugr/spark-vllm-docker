#!/usr/bin/env python3
"""Fix the misplaced XPU-only return introduced by vLLM PR #49408."""

import sys
from pathlib import Path


source_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
target = source_root / "vllm/_custom_ops.py"
function_marker = "def topk_hash_softplus_sqrt("
direct_call = "\n    torch.ops._moe_C.topk_softplus_sqrt("
misplaced_return = "\n\n    return" + direct_call
fixed_return = "\n        return\n" + direct_call

if not target.exists():
    raise SystemExit(f"{target} not found; cannot inspect vLLM PR #49408 regression")

text = target.read_text()
start = text.find(function_marker)
if start == -1:
    print("topk_hash_softplus_sqrt is absent; vLLM PR #49408 is not applicable")
else:
    end = text.find("\ndef ", start + len(function_marker))
    end = len(text) if end == -1 else end
    function = text[start:end]

    if "is_padding" not in function:
        print("topk_hash_softplus_sqrt predates vLLM PR #49408; skipping workaround")
    elif fixed_return in function:
        print("vLLM topk_softplus_sqrt non-XPU path already fixed; skipping")
    elif (
        direct_call in function
        and "current_platform.is_xpu()" not in function
        and "\n    return" not in function
    ):
        print("topk_hash_softplus_sqrt has no XPU workaround; patch is not applicable")
    elif function.count(misplaced_return) == 1:
        function = function.replace(misplaced_return, fixed_return, 1)
        updated = text[:start] + function + text[end:]
        compile(updated, str(target), "exec")
        target.write_text(updated)
        print("Applied vLLM PR #49452 topk_softplus_sqrt control-flow fix")
    else:
        raise SystemExit(
            "Unknown is_padding-aware topk_hash_softplus_sqrt layout; "
            "refusing to guess whether vLLM PR #49408 is fixed"
        )
