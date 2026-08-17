#!/bin/bash
set -euo pipefail

PYTHON_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UVA_FALLBACK_PATCH_FILE="$MOD_DIR/wsl-uva-fallback.patch"

apply_patch_once() {
  local name="$1"
  local patch_file="$2"

  if [ ! -f "$patch_file" ]; then
    echo "[uma-fix] $name patch not found: $patch_file" >&2
    exit 1
  elif git apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "[uma-fix] $name patch is already applied; skipping."
  elif git apply --check "$patch_file"; then
    git apply "$patch_file"
    echo "[uma-fix] Applied $name patch."
  else
    echo "[uma-fix] $name patch could not be applied to installed vLLM." >&2
    exit 1
  fi
}

patch_uma_memory_fix() {
  python3 - "$PYTHON_ROOT" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
CUDA_IMPORT = "from vllm.utils.mem_utils import cuda_mem_get_info\n"
MEM_IMPORTS = (
    "from vllm.platforms.interface import in_wsl\n",
    "from vllm.utils.import_utils import import_pynvml\n",
)
HELPER_BLOCK = '''

def _device_index(device: torch.types.Device) -> int:
    resolved_device = torch.device(device)
    return resolved_device.index if resolved_device.index is not None else 0


def _nvml_mem_get_info(device_id: int) -> tuple[int, int]:
    pynvml = import_pynvml()
    pynvml.nvmlInit()
    try:
        try:
            map_visible_device = getattr(
                current_platform,
                "visible_device_id_to_physical_device_id",
                current_platform.device_id_to_physical_device_id,
            )
            physical_device_id = map_visible_device(device_id)
        except Exception:
            physical_device_id = device_id
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(memory_info.free), int(memory_info.total)
    finally:
        pynvml.nvmlShutdown()


def cuda_mem_get_info(device: torch.types.Device | None = None) -> tuple[int, int]:
    if device is None:
        try:
            device_fn = current_platform.current_device
            assert device_fn is not None
            current_device = device_fn()
            resolved_device = torch.device(current_device)
        except Exception:
            resolved_device = torch.device("cuda", torch.cuda.current_device())
    else:
        resolved_device = torch.device(device)

    get_memory_info = getattr(
        getattr(torch, "accelerator", None), "get_memory_info", None
    )
    if get_memory_info is not None:
        cuda_free_memory, cuda_total_memory = get_memory_info(resolved_device)
    else:
        try:
            cuda_free_memory, cuda_total_memory = current_platform.mem_get_info(
                resolved_device
            )
        except TypeError:
            cuda_free_memory, cuda_total_memory = current_platform.mem_get_info()
    if not in_wsl():
        return cuda_free_memory, cuda_total_memory

    device_id = _device_index(resolved_device)
    try:
        return _nvml_mem_get_info(device_id)
    except Exception:
        try:
            total_memory = current_platform.get_device_total_memory(device_id)
        except Exception:
            total_memory = cuda_total_memory
        return min(cuda_free_memory, total_memory), total_memory
'''
MEMORY_MEASURE_BLOCK = '''        device_id = _device_index(device)
        is_wsl = in_wsl()
        cuda_free_memory, cuda_total_memory = cuda_mem_get_info(device)
        self.free_memory = cuda_free_memory
        self.total_memory = cuda_total_memory
'''
UMA_FREE_BLOCK = '''            host_available_memory = psutil.virtual_memory().available
            self.free_memory = min(
                max(cuda_free_memory, host_available_memory),
                self.total_memory,
            )'''


def ensure_import(text: str, import_line: str, path: Path) -> tuple[str, bool]:
    if import_line in text:
        return text, False

    ast_mod = __import__("ast")
    tree = ast_mod.parse(text)
    import_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast_mod.Import, ast_mod.ImportFrom))
    ]
    if not import_nodes:
        print(f"[uma-fix] Could not find import block in {path}.", file=sys.stderr)
        raise SystemExit(1)

    vllm_imports = [
        node
        for node in import_nodes
        if isinstance(node, ast_mod.ImportFrom)
        and node.module is not None
        and (node.module == "vllm" or node.module.startswith("vllm."))
    ]
    anchor = vllm_imports[-1] if vllm_imports else import_nodes[-1]
    lines = text.splitlines(keepends=True)
    insert_at = anchor.end_lineno
    lines.insert(insert_at, import_line)
    return "".join(lines), True


def patch_mem_utils() -> None:
    path = root / "vllm/utils/mem_utils.py"
    if not path.exists():
        print("[uma-fix] vllm/utils/mem_utils.py not found.", file=sys.stderr)
        raise SystemExit(1)

    text = path.read_text()
    original = text

    for import_line in MEM_IMPORTS:
        text, _ = ensure_import(text, import_line, path)

    if "def cuda_mem_get_info(" not in text:
        marker = "\n@cache\ndef get_max_shared_memory_bytes"
        if marker not in text:
            print(
                "[uma-fix] Could not find insertion point for cuda_mem_get_info.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        text = text.replace(marker, HELPER_BLOCK + marker, 1)

    ast_mod = __import__("ast")
    tree = ast_mod.parse(text)

    def has_platform_call(node: object, name: str) -> bool:
        for child in ast_mod.walk(node):
            if (
                isinstance(child, ast_mod.Call)
                and isinstance(child.func, ast_mod.Attribute)
                and child.func.attr == name
                and isinstance(child.func.value, ast_mod.Name)
                and child.func.value.id == "current_platform"
            ):
                return True
        return False

    release_if = None
    for node in tree.body:
        if isinstance(node, ast_mod.FunctionDef) and node.name == "release_device_memory_under_pressure":
            for item in ast_mod.walk(node):
                if isinstance(item, ast_mod.If) and has_platform_call(item.test, "is_integrated_gpu"):
                    release_if = item
                    break
            break

    if release_if is not None:
        lines = text.splitlines(keepends=True)
        header_end = release_if.body[0].lineno - 1 if release_if.body else release_if.lineno
        lines[release_if.lineno - 1:header_end] = [
            '    if (\n',
            '        device.type != "cuda"\n',
            '        or in_wsl()\n',
            '        or not current_platform.is_integrated_gpu(_device_index(device))\n',
            '    ):\n',
        ]
        text = "".join(lines)
    elif "or in_wsl()" not in text:
        print(
            "[uma-fix] release_device_memory_under_pressure condition was not found.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tree = __import__("ast").parse(text)
    measure_func = None
    for node in tree.body:
        if isinstance(node, __import__("ast").ClassDef) and node.name == "MemorySnapshot":
            for item in node.body:
                if isinstance(item, __import__("ast").FunctionDef) and item.name == "measure":
                    measure_func = item
                    break
            break
    if measure_func is None:
        print("[uma-fix] MemorySnapshot.measure was not found.", file=sys.stderr)
        raise SystemExit(1)

    mem_assign = None
    uma_if = None
    ast_mod = __import__("ast")

    def target_names(target: object) -> set[str]:
        names = set()
        for child in ast_mod.walk(target):
            if (
                isinstance(child, ast_mod.Attribute)
                and isinstance(child.value, ast_mod.Name)
                and child.value.id == "self"
            ):
                names.add(child.attr)
        return names

    def is_memory_info_call(value: object) -> bool:
        if not isinstance(value, ast_mod.Call):
            return False
        func = value.func
        if not isinstance(func, ast_mod.Attribute):
            return False
        if (
            func.attr == "mem_get_info"
            and isinstance(func.value, ast_mod.Name)
            and func.value.id == "current_platform"
        ):
            return True
        if func.attr != "get_memory_info":
            return False
        owner = func.value
        return (
            isinstance(owner, ast_mod.Attribute)
            and owner.attr == "accelerator"
            and isinstance(owner.value, ast_mod.Name)
            and owner.value.id == "torch"
        )

    for node in ast_mod.walk(measure_func):
        if isinstance(node, ast_mod.Assign):
            value = node.value
            assigned_names = set().union(*(target_names(t) for t in node.targets))
            if {"free_memory", "total_memory"}.issubset(
                assigned_names
            ) and is_memory_info_call(value):
                mem_assign = node
        elif isinstance(node, ast_mod.If):
            test = node.test
            if (
                isinstance(test, ast_mod.Call)
                and isinstance(test.func, ast_mod.Attribute)
                and test.func.attr == "is_integrated_gpu"
                and isinstance(test.func.value, ast_mod.Name)
                and test.func.value.id == "current_platform"
            ):
                uma_if = node

    lines = text.splitlines(keepends=True)
    replacements = []
    if mem_assign is not None:
        replacements.append((mem_assign.lineno - 1, mem_assign.end_lineno, MEMORY_MEASURE_BLOCK))
    elif "cuda_free_memory, cuda_total_memory = cuda_mem_get_info(device)" not in text:
        print("[uma-fix] MemorySnapshot mem_get_info assignment was not found.", file=sys.stderr)
        raise SystemExit(1)

    if uma_if is not None:
        header_end = uma_if.lineno
        depth = 0
        while header_end <= len(lines):
            header = lines[header_end - 1]
            depth += header.count("(") - header.count(")")
            if depth == 0 and header.rstrip().endswith(":"):
                break
            header_end += 1
        replacements.append((uma_if.lineno - 1, header_end, "        if not is_wsl and current_platform.is_integrated_gpu(device_id):\n"))
    elif "if not is_wsl and current_platform.is_integrated_gpu(device_id):" not in text:
        print("[uma-fix] MemorySnapshot UMA condition was not found.", file=sys.stderr)
        raise SystemExit(1)

    for start_line, end_line, replacement in sorted(replacements, reverse=True):
        lines[start_line:end_line] = replacement.splitlines(keepends=True)
    text = "".join(lines)

    text = text.replace(
        "            # Use psutil to get the true available memory.\n",
        "            # Use the larger of CUDA's allocatable-device view and host\n"
        "            # available memory. In some cases, psutil can underreport relative\n"
        "            # to CUDA's actual allocation budget.\n",
        1,
    )
    text = text.replace(
        "            self.free_memory = psutil.virtual_memory().available",
        UMA_FREE_BLOCK,
        1,
    )

    if text == original:
        print("[uma-fix] vllm/utils/mem_utils.py is already patched; skipping.")
        return

    path.write_text(text)
    print("[uma-fix] Patched vllm/utils/mem_utils.py.")


def patch_file(relative_path: str, replacements: tuple[tuple[str, str], ...]) -> None:
    path = root / relative_path
    if not path.exists():
        print(f"[uma-fix] {relative_path} not found; skipping.")
        return

    text = path.read_text()
    original = text

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    if "cuda_mem_get_info" in text:
        text, _ = ensure_import(text, CUDA_IMPORT, path)

    if text == original:
        if "cuda_mem_get_info" in text:
            print(f"[uma-fix] {relative_path} is already patched; skipping.")
        else:
            print(f"[uma-fix] {relative_path} has no legacy memory calls; skipping.")
        return

    path.write_text(text)
    print(f"[uma-fix] Patched {relative_path}.")


patch_mem_utils()
patch_file(
    "vllm/model_executor/models/gemma4_mm.py",
    (
        (
            r"(?:current_platform\.mem_get_info|torch\.accelerator\.get_memory_info)\(\s*\)",
            "cuda_mem_get_info()",
        ),
    ),
)
patch_file(
    "vllm/v1/worker/gpu/model_runner.py",
    (
        (
            r"(?:torch\.cuda\.mem_get_info|torch\.accelerator\.get_memory_info)\(\s*\)\[0\]",
            "cuda_mem_get_info(self.device)[0]",
        ),
    ),
)
patch_file(
    "vllm/v1/worker/gpu/spec_decode/eagle/utils.py",
    (
        (
            r"(?:torch\.cuda\.mem_get_info|torch\.accelerator\.get_memory_info)\(\s*w\.device\s*\)\[0\]",
            "cuda_mem_get_info(w.device)[0]",
        ),
        (
            r"(?:torch\.cuda\.mem_get_info|torch\.accelerator\.get_memory_info)\(\s*device\s*=\s*w\.device\s*\)\[0\]",
            "cuda_mem_get_info(w.device)[0]",
        ),
    ),
)
patch_file(
    "vllm/v1/worker/gpu_model_runner.py",
    (
        (
            r"(?:torch\.cuda\.mem_get_info|torch\.accelerator\.get_memory_info)\(\s*\)\[0\]",
            "cuda_mem_get_info(self.device)[0]",
        ),
    ),
)
patch_file(
    "vllm/v1/worker/gpu_worker.py",
    (
        (
            r"(?:torch\.cuda\.mem_get_info|torch\.accelerator\.get_memory_info)\(\s*\)\[0\]",
            "cuda_mem_get_info(self.device)[0]",
        ),
        (
            r"(?:torch\.cuda\.mem_get_info|torch\.accelerator\.get_memory_info)\(\s*\)",
            "cuda_mem_get_info(self.device)",
        ),
    ),
)
PY
}

if ! command -v git >/dev/null 2>&1; then
  echo "[uma-fix] git is required to apply this mod." >&2
  echo "[uma-fix] Apply mods/use-official-vllm first if this container does not include git." >&2
  exit 1
fi

if [ ! -d "$PYTHON_ROOT/vllm" ]; then
  echo "[uma-fix] vLLM package not found at $PYTHON_ROOT/vllm" >&2
  exit 1
fi

cd "$PYTHON_ROOT"

patch_uma_memory_fix
apply_patch_once "WSL UVA fallback" "$UVA_FALLBACK_PATCH_FILE"
