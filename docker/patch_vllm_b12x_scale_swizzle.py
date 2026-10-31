#!/usr/bin/env python3
"""Resolve B12X scale swizzling after the intrinsics package split.

Patch vLLM source before wheel compilation. The shared accessor serves both
NVFP4 and MXFP4. Older FlashInfer refs still export the helper from intrinsics.py;
newer refs define it in intrinsics/reference.py without re-exporting it.
"""

import argparse
import ast
from pathlib import Path


TARGET_REL = Path("vllm/utils/b12x.py")
OLD_MODULE = "b12x._lib.intrinsics"
REFERENCE_MODULE = OLD_MODULE + ".reference"
MARKER = "spark-vllm-docker: resolve the B12X scale-swizzle export"
FIXED_HELPER = f'''def get_b12x_intrinsics() -> ModuleType | None:
    # {MARKER}.
    for name in ("{REFERENCE_MODULE}", "{OLD_MODULE}"):
        module = _get_submodule(name)
        if callable(getattr(module, "swizzle_block_scale", None)):
            return module
    return None
'''


def patch_source(source: str) -> str:
    tree = ast.parse(source)
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "get_b12x_intrinsics"]
    if not helpers:
        return source
    if len(helpers) != 1:
        raise ValueError("Expected one get_b12x_intrinsics accessor")
    helper = helpers[0]
    fixed = ast.parse(FIXED_HELPER).body[0]
    old = ast.parse(
        f'def get_b12x_intrinsics() -> ModuleType | None:\n'
        f'    return _get_submodule("{OLD_MODULE}")\n'
    ).body[0]
    direct = ast.parse(
        f'def get_b12x_intrinsics() -> ModuleType | None:\n'
        f'    return _get_submodule("{REFERENCE_MODULE}")\n'
    ).body[0]
    shape = ast.dump(helper)
    if shape not in (ast.dump(old), ast.dump(fixed), ast.dump(direct)):
        if any(isinstance(node, ast.Constant) and node.value == OLD_MODULE
               for node in ast.walk(helper)) or MARKER in source:
            raise ValueError("Unknown B12X intrinsics accessor; review upstream")
        return source

    registries = [node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_B12X_SUBMODULES"
                          for t in node.targets)]
    if (len(registries) != 1 or not isinstance(registries[0], ast.DictComp)
            or len(registries[0].generators) != 1
            or not isinstance(registries[0].generators[0].iter, ast.Tuple)):
        raise ValueError("Unknown B12X submodule registry; review upstream")
    entries = registries[0].generators[0].iter.elts
    old_entries = [node for node in entries if isinstance(node, ast.Constant)
                   and node.value == OLD_MODULE]
    references = [node for node in entries if isinstance(node, ast.Constant)
                  and node.value == REFERENCE_MODULE]
    if shape == ast.dump(direct):
        if len(references) != 1:
            raise ValueError("Scale-swizzle reference module is not registered once")
        return source
    if shape == ast.dump(fixed):
        if len(references) != 1 or len(old_entries) != 1:
            raise ValueError("Incomplete B12X scale-swizzle patch")
        return source
    if len(old_entries) != 1 or references:
        raise ValueError("Unexpected B12X intrinsics registry entries")

    lines = source.splitlines(keepends=True)
    entry = old_entries[0]
    if entry.lineno != entry.end_lineno or helper.decorator_list:
        raise ValueError("Unexpected B12X accessor source layout")
    replacements = [
        (helper.lineno - 1, helper.end_lineno, FIXED_HELPER),
        (entry.lineno - 1, entry.lineno,
         " " * entry.col_offset + f'"{REFERENCE_MODULE}",\n'
         + lines[entry.lineno - 1]),
    ]
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines)
    compile(result, str(TARGET_REL), "exec")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    target = args.source_root / TARGET_REL
    if not target.exists():
        print("B12X accessors are absent; scale-swizzle patch is not applicable")
        return
    source = target.read_text()
    try:
        patched = patch_source(source)
    except (ValueError, SyntaxError) as exc:
        raise SystemExit(f"Unable to patch {target}: {exc}") from exc
    if patched == source:
        print("B12X scale-swizzle accessor is unaffected or already fixed; skipping")
    else:
        target.write_text(patched)
        print("Patched shared B12X NVFP4/MXFP4 scale-swizzle accessor")


if __name__ == "__main__":
    main()
