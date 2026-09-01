#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[uma-fix] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

python3 - "$PYTHON_ROOT" <<'PY'
from pathlib import Path
import ast
import re
import sys


root = Path(sys.argv[1])
PIN_MEMORY_ENV = "VLLM_WSL2_ENABLE_PIN_MEMORY"
WSL_IMPORT = "from vllm.platforms.interface import in_wsl\n"


def ensure_import(text: str, import_line: str, path: Path) -> tuple[str, bool]:
    if import_line in text:
        return text, False

    tree = ast.parse(text)
    import_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    if not import_nodes:
        print(f"[uma-fix] Could not find import block in {path}.", file=sys.stderr)
        raise SystemExit(1)

    vllm_imports = [
        node
        for node in import_nodes
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "vllm" or node.module.startswith("vllm."))
    ]
    anchor = vllm_imports[-1] if vllm_imports else import_nodes[-1]
    lines = text.splitlines(keepends=True)
    lines.insert(anchor.end_lineno, import_line)
    return "".join(lines), True


def has_call(node: ast.AST, owner: str | None, name: str) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if owner is None and isinstance(child.func, ast.Name):
            if child.func.id == name:
                return True
        elif (
            isinstance(child.func, ast.Attribute)
            and child.func.attr == name
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == owner
        ):
            return True
    return False


def find_function(tree: ast.Module, class_name: str | None, name: str) -> ast.FunctionDef | None:
    body: list[ast.stmt]
    if class_name is None:
        body = tree.body
    else:
        class_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == class_name
            ),
            None,
        )
        if class_node is None:
            return None
        body = class_node.body

    return next(
        (
            node
            for node in body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def find_integrated_gpu_if(function: ast.FunctionDef) -> ast.If | None:
    return next(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and has_call(node.test, "current_platform", "is_integrated_gpu")
        ),
        None,
    )


def patch_pin_memory_default() -> None:
    path = root / "vllm/envs.py"
    if not path.exists():
        print("[uma-fix] vllm/envs.py not found.", file=sys.stderr)
        raise SystemExit(1)

    text = path.read_text()
    original = text
    runtime_default = re.compile(
        r'(os\.(?:getenv|environ\.get)\(\s*["\']'
        + PIN_MEMORY_ENV
        + r'["\']\s*,\s*["\'])0(["\']\s*\))'
    )
    enabled_default = re.compile(
        r'os\.(?:getenv|environ\.get)\(\s*["\']'
        + PIN_MEMORY_ENV
        + r'["\']\s*,\s*["\']1["\']\s*\)'
    )

    text, replacements = runtime_default.subn(r"\g<1>1\g<2>", text, count=1)
    if replacements == 0 and not enabled_default.search(text):
        print(
            f"[uma-fix] {PIN_MEMORY_ENV} runtime default was not found in {path}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    text = re.sub(
        rf"(\b{PIN_MEMORY_ENV}:\s*bool\s*=\s*)False\b",
        r"\g<1>True",
        text,
        count=1,
    )

    ast.parse(text)
    if text == original:
        print(f"[uma-fix] {PIN_MEMORY_ENV} already defaults to 1; skipping.")
        return

    path.write_text(text)
    print(f"[uma-fix] Set {PIN_MEMORY_ENV}=1 as the vLLM default.")


def patch_wsl_memory_accounting() -> None:
    path = root / "vllm/utils/mem_utils.py"
    if not path.exists():
        print("[uma-fix] vllm/utils/mem_utils.py not found.", file=sys.stderr)
        raise SystemExit(1)

    text = path.read_text()
    original = text
    text, _ = ensure_import(text, WSL_IMPORT, path)
    tree = ast.parse(text)

    release_func = find_function(
        tree, None, "release_device_memory_under_pressure"
    )
    if release_func is None:
        print(
            "[uma-fix] release_device_memory_under_pressure was not found.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    measure_func = find_function(tree, "MemorySnapshot", "measure")
    if measure_func is None:
        print("[uma-fix] MemorySnapshot.measure was not found.", file=sys.stderr)
        raise SystemExit(1)

    release_if = find_integrated_gpu_if(release_func)
    measure_if = find_integrated_gpu_if(measure_func)
    if release_if is None:
        print(
            "[uma-fix] Integrated-GPU release condition was not found.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if measure_if is None:
        print(
            "[uma-fix] MemorySnapshot UMA condition was not found.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    replacements: list[tuple[int, int, str]] = []
    if not has_call(release_if.test, None, "in_wsl"):
        release_header_end = release_if.body[0].lineno - 1
        replacements.append(
            (
                release_if.lineno - 1,
                release_header_end,
                "    if (\n"
                '        device.type != "cuda"\n'
                "        or in_wsl()\n"
                "        or not current_platform.is_integrated_gpu(device.index)\n"
                "    ):\n",
            )
        )

    if not has_call(measure_if.test, None, "in_wsl"):
        measure_header_end = measure_if.body[0].lineno - 1
        replacements.append(
            (
                measure_if.lineno - 1,
                measure_header_end,
                "        if (\n"
                "            not in_wsl()\n"
                "            and current_platform.is_integrated_gpu(device.index)\n"
                "        ):\n",
            )
        )

    lines = text.splitlines(keepends=True)
    for start_line, end_line, replacement in sorted(replacements, reverse=True):
        lines[start_line:end_line] = replacement.splitlines(keepends=True)
    text = "".join(lines)

    ast.parse(text)
    if text == original:
        print("[uma-fix] WSL memory accounting is already patched; skipping.")
        return

    path.write_text(text)
    print(
        "[uma-fix] Disabled host UMA accounting under WSL; "
        "vLLM will use CUDA's memory values."
    )


patch_pin_memory_default()
patch_wsl_memory_accounting()
PY
