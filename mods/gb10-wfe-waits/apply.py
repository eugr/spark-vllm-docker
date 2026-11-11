"""gb10-wfe-waits: park busy-waiting host threads in WFET instead of spinning (aarch64 FEAT_WFxT).

Applied inside every container (head and workers) before the engine starts. Four independent parts; each one
either applies completely, is already applied, or is skipped with a reason and leaves that file stock:

  1. b12x RoCE proxy (`b12x/comm/roce/_roce_proxy.c`, standalone b12x or FlashInfer's experimental b12x): the idle
     doorbell wait arms the exclusive monitor (LDXR) and parks in WFET instead of spinning. b12x compiles the proxy
     from this source at runtime (cache keyed by source hash), so only the source changes.
  2. vLLM `spinloop` extension: rebuilt from the bundled spinloop.cpp, which adds an aarch64 branch with the same
     monitor + WFET wait for the shared-memory broadcast queue (EngineCore <-> local workers).
  3. vLLM `shm_broadcast.py`: a reader passes its own read flag to `spinloop()` as well (`rank + 2` instead of
     `rank + 1`); `check()` reads that byte, and the WFET path compares exactly the bytes it is given.
  4. vLLM `envs.py`: `VLLM_USE_SPINLOOP_EXT` defaults to 1 (only if part 2 is in place), so the queue uses it.

Runtime opt-outs: B12X_ROCE_WFE=0 (proxy spins again), VLLM_SPINLOOP_WFE=0 (spinloop yields again),
VLLM_USE_SPINLOOP_EXT=0 (stock Python wait). B12X_ROCE_DEBUG=1 prints the proxy's wait mode.
WFE_WAITS_STRICT=1 makes the mod exit 1 if any part had to be skipped.
"""
from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

TAG = "[gb10-wfe-waits]"
ROOT = Path(sys.argv[1])
MOD = Path(sys.argv[2])
STRICT = os.environ.get("WFE_WAITS_STRICT") == "1"
skipped: list[str] = []


def log(msg: str) -> None:
    print(f"{TAG} {msg}", flush=True)


class Fatal(Exception):
    """A state the mod cannot leave behind: always exits non-zero, regardless of WFE_WAITS_STRICT."""


def skip(part: str, why: str) -> None:
    log(f"{part}: SKIPPED, left stock - {why}")
    skipped.append(part)


def first_tool(*names: str | None) -> str | None:
    for n in names:
        if n and shutil.which(n):
            return n
    return None


def run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Bounded subprocess; raises OSError/TimeoutExpired, callers turn both into a skipped part."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _atomic_write(dest: Path, data: bytes, mode_from: Path) -> None:
    """Write data to a staging file beside dest, fsync it, copy the mode, then os.replace() it into place."""
    tmp = dest.with_name(f".{dest.name}.wfe-{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        shutil.copymode(mode_from, tmp)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def install(dest: Path, data: bytes) -> None:
    """Publish a complete .stock backup once (never a partial one), then replace dest atomically."""
    backup = dest.with_name(dest.name + ".stock")
    if not backup.exists():
        _atomic_write(backup, dest.read_bytes(), dest)
    _atomic_write(dest, data, dest)


def has_wfxt() -> bool:
    try:
        libc = ctypes.CDLL(None)
        libc.getauxval.restype = ctypes.c_ulong
        return bool(libc.getauxval(26) & (1 << 31))  # AT_HWCAP2, HWCAP2_WFXT
    except Exception:
        return False


def sub_once(text: str, pattern: str, repl, what: str) -> str:
    rx = re.compile(pattern, re.M)
    hits = rx.findall(text)
    if len(hits) != 1:
        raise LookupError(f"anchor '{what}' found {len(hits)} times, expected 1")
    return rx.sub(repl, text, count=1)


# ---------------------------------------------------------------- 1. RoCE proxy
PROXY_HELPERS = r"""// gb10-wfe-waits: idle doorbell wait.  On aarch64 cores with FEAT_WFxT the proxy
// parks in WFET with the exclusive monitor armed on ctrl[0]'s cache line instead
// of spinning; the GPU's store of the next sequence clears the monitor and wakes
// the core.  The absolute 1 ms deadline bounds every park so the running flag
// and the CQ drain stay serviced.  B12X_ROCE_WFE=0 restores the spin.
#if defined(__aarch64__)
static inline uint64_t roce_cntvct(void) {
@@uint64_t t;
@@__asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(t)::"memory");
@@return t;
}
static inline uint64_t roce_cntfrq(void) {
@@uint64_t f;
@@__asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
@@return f;
}
// Same sequence as Linux __cmpwait_case_32, with WFET bounded by an absolute CNTVCT deadline.
static inline void roce_wait_change(volatile uint32_t* p, uint32_t old, uint64_t deadline) {
@@uint32_t tmp;
@@__asm__ volatile("mov x9, %[dl]\n\t"
@@@@"sevl\n\t"
@@@@"wfe\n\t"
@@@@"ldxr %w[tmp], %[v]\n\t"
@@@@"eor %w[tmp], %w[tmp], %w[old]\n\t"
@@@@"cbnz %w[tmp], 1f\n\t"
@@@@".inst 0xd5031009\n"  // wfet x9
@@@@"1:"
@@@@: [tmp] "=&r"(tmp), [v] "+Q"(*p)
@@@@: [old] "r"(old), [dl] "r"(deadline)
@@@@: "x9", "memory");
}
static int roce_use_wfe(void) {
@@const char* e = getenv("B12X_ROCE_WFE");
@@if (e != NULL && e[0] == '0') {
@@@@return 0;
@@}
@@return (getauxval(AT_HWCAP2) & HWCAP2_WFXT) != 0;
}
#endif

"""
PROXY_INCLUDES = """#if defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP2_WFXT
#define HWCAP2_WFXT (1UL << 31)
#endif
#endif
"""


def patch_proxy_text(text: str) -> str:
    m = re.search(r"^([ \t]+)const struct timespec nap = \{0, 20000\};\n", text, re.M)
    if not m:
        raise LookupError("anchor 'nap' not found")
    u = m.group(1)  # one indent level of this file (4 spaces in b12x, 2 in FlashInfer)
    text = sub_once(text, r"^#include <time\.h>\n", lambda mm: mm.group(0) + PROXY_INCLUDES, "#include <time.h>")
    text = sub_once(text, r"^static void\s*\*\s*proxy_main\(void\s*\*\s*arg\)\s*\{\n",
                    lambda mm: PROXY_HELPERS.replace("@@", u) + mm.group(0), "proxy_main")
    setup = (f"{u}const struct timespec nap = {{0, 20000}};\n"
             "#if defined(__aarch64__)\n"
             f"{u}const int use_wfe = roce_use_wfe();\n"
             f"{u}const uint64_t wfe_slice = roce_cntfrq() / 1000;  // bounded park: 1 ms\n"
             f"{u}if (getenv(\"B12X_ROCE_DEBUG\") != NULL) {{\n"
             f"{u}{u}fprintf(stderr, \"[b12x roce proxy] doorbell wait: %s\\n\", use_wfe ? \"WFET\" : \"spin\");\n"
             f"{u}}}\n"
             "#else\n"
             f"{u}const int use_wfe = 0;\n"
             "#endif\n"
             f"{u}// A parked iteration lasts until the next store to the line, an event-stream\n"
             f"{u}// event (~100-130 us on GB10) or the 1 ms deadline; drain the CQs every 8\n"
             f"{u}// parked iterations instead of every 64 spins.\n"
             f"{u}const uint64_t drain_every = use_wfe ? 8 : 64;\n")
    text = sub_once(text, r"^[ \t]+const struct timespec nap = \{0, 20000\};\n", lambda mm: setup, "nap")
    text = sub_once(text, r"^([ \t]+)if \(idle % 64 == 0\) \{\n",
                    lambda mm: f"{mm.group(1)}if (idle % drain_every == 0) {{\n", "idle % 64")

    def wfe_block(mm):
        i = mm.group(1)
        return ("#if defined(__aarch64__)\n"
                f"{i}if (use_wfe) {{\n"
                f"{i}{u}roce_wait_change(&ctrl[0], seq, roce_cntvct() + wfe_slice);\n"
                f"{i}{u}continue;\n"
                f"{i}}}\n"
                "#endif\n" + mm.group(0))
    return sub_once(text, r"^([ \t]+)if \(idle >= ROCE_IDLE_SPINS\) \{\n", wfe_block, "ROCE_IDLE_SPINS")


def proxy_part() -> None:
    cands = [ROOT / "b12x/comm/roce/_roce_proxy.c", ROOT / "flashinfer/experimental/b12x/comm/roce/_roce_proxy.c"]
    files, seen = [], set()
    for c in cands:
        if c.is_file() and c.resolve() not in seen:
            seen.add(c.resolve())
            files.append(c)
    if not files:
        skip("roce-proxy", "no b12x/comm/roce/_roce_proxy.c (image without b12x RoCE all-reduce)")
        return
    cc = first_tool(os.environ.get("CC"), "gcc", "cc", "clang")
    for f in files:
        text = f.read_text()
        if "roce_wait_change" in text:
            log(f"roce-proxy: already applied ({f})")
            continue
        try:
            new = patch_proxy_text(text)
        except LookupError as e:
            skip("roce-proxy", f"{f}: {e}")
            continue
        if not cc:
            # a changed source changes b12x's build-cache key; without a compiler that would break startup
            skip("roce-proxy", f"{f}: no C compiler to test-build the patched source")
            continue
        try:
            with tempfile.TemporaryDirectory() as td:  # same command b12x uses at runtime
                src = Path(td) / "_roce_proxy.c"
                src.write_text(new)
                r = run([cc, "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", str(Path(td) / "p.so"),
                         str(src), "-libverbs", "-lpthread"], timeout=300)
            if r.returncode != 0:
                skip("roce-proxy", f"{f}: patched source does not compile: {r.stderr.strip()[:300]}")
                continue
            install(f, new.encode())
        except (OSError, subprocess.TimeoutExpired) as e:
            skip("roce-proxy", f"{f}: {e!r}")
            continue
        log(f"roce-proxy: applied ({f})")


# ---------------------------------------------------------------- 2. spinloop extension
SELFTEST = """
import importlib.util, sys, time
spec = importlib.util.spec_from_file_location("vllm.spinloop", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
v = memoryview(bytearray(64))
if m.spinloop(v[0:2], lambda: True, timeout=0.1) is not True:
    sys.exit("ready callback did not return True")
t = time.perf_counter()
r = m.spinloop(v[0:2], lambda: False, timeout=0.05)
dt = time.perf_counter() - t
if r is not False or not 0.04 < dt < 1.0:
    sys.exit(f"timeout path returned {r!r} after {dt:.3f} s")
"""


def selftest(path: Path) -> str | None:
    """Load the extension in a fresh interpreter and exercise it; None on success, else the reason."""
    try:
        r = run([sys.executable, "-c", SELFTEST, str(path)], timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return repr(e)
    if r.returncode != 0:
        # never empty: an import-time SIGSEGV/SIGILL exits with a negative code and no output
        return f"self-test exit code {r.returncode}: {(r.stderr or r.stdout).strip()[-300:] or '(no output)'}"
    if "aarch64 wait: WFET" not in r.stderr:
        return f"extension did not select WFET: {r.stderr.strip()[-200:]}"
    return None


def spinloop_part() -> bool:
    targets = sorted((ROOT / "vllm").glob("spinloop*.so"))
    if not targets:
        skip("spinloop", "vllm/spinloop*.so not found (this vLLM has no spinloop extension)")
        return False
    target = targets[0]
    if b"aarch64 wait: %s" in target.read_bytes():
        why = selftest(target)
        if why is not None:
            # a broken extension is installed and may already be enabled by default: refuse to start
            raise Fatal(f"spinloop: installed {target} carries the patch marker but fails its self-test: {why}; "
                        f"restore {target.name}.stock or recreate the container")
        log(f"spinloop: already applied and self-tested ({target})")
        return True
    cxx = first_tool(os.environ.get("CXX"), "g++", "c++", "clang++")
    inc = Path(sysconfig.get_paths()["include"])
    if not cxx or not (inc / "Python.h").is_file():
        skip("spinloop", f"needs a C++ compiler and Python headers (compiler={cxx}, {inc}/Python.h present="
                         f"{(inc / 'Python.h').is_file()})")
        return False
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / target.name
            cmd = [cxx, "-O2", "-fPIC", "-shared", f"-I{inc}", "-o", str(out), str(MOD / "spinloop.cpp")]
            if ".abi3." in target.name:
                cmd.insert(4, "-DPy_LIMITED_API=0x030B0000")
            r = run(cmd, timeout=300)
            if r.returncode != 0:
                skip("spinloop", f"build failed: {r.stderr.strip()[:300]}")
                return False
            why = selftest(out)
            if why is not None:
                skip("spinloop", f"self-test of the new build failed: {why}")
                return False
            install(target, out.read_bytes())
    except (OSError, subprocess.TimeoutExpired) as e:
        skip("spinloop", repr(e))
        return False
    why = selftest(target)  # the installed file, not just the staged one
    if why is not None:
        try:
            install(target, target.with_name(target.name + ".stock").read_bytes())
        except OSError as e:
            raise Fatal(f"spinloop: installed extension failed its self-test ({why}) and restoring "
                        f"{target.name}.stock failed: {e!r}")
        skip("spinloop", f"installed extension failed its self-test, stock restored: {why}")
        return False
    log(f"spinloop: applied and self-tested ({target})")
    return True


# ---------------------------------------------------------------- 3. shm_broadcast reader slice
def slice_part() -> None:
    f = ROOT / "vllm/distributed/device_communicators/shm_broadcast.py"
    if not f.is_file():
        skip("shm-slice", f"{f} not found")
        return
    text = f.read_text()
    if "metadata_buffer[0 : self.local_reader_rank + 2]" in text:
        log("shm-slice: already applied")
        return
    try:
        new = sub_once(text, r"metadata_buffer\[0 : self\.local_reader_rank \+ 1\],",
                       lambda mm: "metadata_buffer[0 : self.local_reader_rank + 2],", "reader spinloop slice")
    except LookupError as e:
        skip("shm-slice", f"{e} (the WFET wait still works; a rare race can then wait up to its 1 ms deadline)")
        return
    try:
        install(f, new.encode())
    except OSError as e:
        skip("shm-slice", repr(e))
        return
    log("shm-slice: applied")


# ---------------------------------------------------------------- 4. VLLM_USE_SPINLOOP_EXT default
def env_default_part() -> None:
    f = ROOT / "vllm/envs.py"
    text = f.read_text() if f.is_file() else ""
    if re.search(r'os\.getenv\("VLLM_USE_SPINLOOP_EXT",\s*"1"\)', text):
        log("spinloop-default: already applied")
        return
    try:
        new = sub_once(text, r'os\.getenv\("VLLM_USE_SPINLOOP_EXT",\s*"0"\)',
                       lambda mm: 'os.getenv("VLLM_USE_SPINLOOP_EXT", "1")', "VLLM_USE_SPINLOOP_EXT default")
    except LookupError as e:
        skip("spinloop-default", f"{e}; set VLLM_USE_SPINLOOP_EXT=1 in the container environment instead")
        return
    try:
        install(f, new.encode())
    except OSError as e:
        skip("spinloop-default", f"{e!r}; set VLLM_USE_SPINLOOP_EXT=1 in the container environment instead")
        return
    log("spinloop-default: applied (VLLM_USE_SPINLOOP_EXT now defaults to 1; set it to 0 to opt out)")


def main() -> int:
    if platform.machine() != "aarch64":
        log(f"{platform.machine()} host: nothing to do")
        return 0
    if not has_wfxt():
        log("this CPU has no FEAT_WFxT (HWCAP2_WFXT): nothing to do")
        return 0
    proxy_part()
    if spinloop_part():
        slice_part()
        env_default_part()
    else:
        log("shm-slice and spinloop-default not applied because the spinloop extension is stock")
    if skipped:
        log(f"done with skipped parts: {', '.join(skipped)}")
        return 1 if STRICT else 0
    log("done: all parts applied")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fatal as e:
        log(f"FATAL: {e}")
        sys.exit(1)
