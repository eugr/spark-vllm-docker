#include <Python.h>

extern "C" {

#include <stdbool.h>
#include <time.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__i386__) || defined(__x86_64__)
  #include <cpuid.h>
  #include <x86intrin.h>
#endif

#if defined(__aarch64__)
  #include <sys/auxv.h>
  #ifndef HWCAP2_WFXT
    #define HWCAP2_WFXT (1UL << 31)
  #endif
#endif

#if defined(CLOCK_MONOTONIC_RAW)
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC_RAW
#else
  #define TIMEOUT_CLOCK CLOCK_MONOTONIC
#endif

#define CPU_SUPPORT_NONE 0
#define CPU_SUPPORT_MONITORX 1
#define CPU_SUPPORT_WFET 2

#define MWAITX_DEFAULT_TIMEOUT_CYCLES 1000000

typedef struct {
  unsigned int cpu_support;
  unsigned int max_monitor_line_size;
  uint64_t wfet_slice_ticks;
} spinloop_state_t;

#if defined(__aarch64__)
// Arm: park in WFET (FEAT_WFxT) with the exclusive monitor armed on the
// buffer, the same pattern as Linux __cmpwait()/smp_cond_load_acquire(). A
// store by another core (or a coherent device) to the monitored granule clears
// the monitor and wakes the waiter; WFET's deadline bounds every park.
// VLLM_SPINLOOP_WFE=0 keeps the plain "yield" busy poll.
static inline uint64_t spinloop_cntvct(void) {
  uint64_t t;
  __asm__ volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(t)::"memory");
  return t;
}

static inline uint64_t spinloop_cntfrq(void) {
  uint64_t f;
  __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f));
  return f;
}

// Arm the monitor on *word and park until it is written, an event arrives or
// CNTVCT reaches `deadline`. Returns at once if *word no longer equals `old`.
static inline void spinloop_wait_change_u64(volatile uint64_t* word,
                                            uint64_t old, uint64_t deadline) {
  uint64_t tmp;
  __asm__ volatile(
      "mov x9, %[dl]\n\t"
      "sevl\n\t"
      "wfe\n\t"                 // consume a stale event
      "ldxr %[tmp], %[v]\n\t"    // arm the exclusive monitor on the granule
      "eor %[tmp], %[tmp], %[old]\n\t"
      "cbnz %[tmp], 1f\n\t"      // changed since the snapshot: do not park
      ".inst 0xd5031009\n"       // wfet x9
      "1:"
      : [tmp] "=&r"(tmp), [v] "+Q"(*word)
      : [old] "r"(old), [dl] "r"(deadline)
      : "x9", "memory");
  (void)tmp;
}
#endif

static void determine_cpu_support(spinloop_state_t* state) {
  state->cpu_support = CPU_SUPPORT_NONE;
  state->max_monitor_line_size = 0;
  state->wfet_slice_ticks = 0;

#if defined(__i386__) || defined(__x86_64__)
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 1) {
    // AMD CPU (possible monitorx/mwaitx support)
    if (ebx == 0x68747541 && edx == 0x69746e65 && ecx == 0x444d4163) {
      if (__get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx) == 1 &&
          eax >= 0x80000001 &&
          __get_cpuid(0x80000001, &eax, &ebx, &ecx, &edx) == 1) {
        if ((ecx & (1 << 29)) != 0) {
          state->cpu_support = CPU_SUPPORT_MONITORX;
        }
      }
    }
  }

  if (state->cpu_support == CPU_SUPPORT_MONITORX) {
    if (__get_cpuid(5, &eax, &ebx, &ecx, &edx) == 1) {
      state->max_monitor_line_size = ebx & 0xff;
    }
  }
#endif

#if defined(__aarch64__)
  const char* env = getenv("VLLM_SPINLOOP_WFE");
  const bool disabled = (env != NULL && env[0] == '0');
  if (!disabled && (getauxval(AT_HWCAP2) & HWCAP2_WFXT) != 0) {
    state->cpu_support = CPU_SUPPORT_WFET;
    // The waiter compares one naturally aligned 64-bit word, so a buffer
    // qualifies when it lies inside a single such word (checked per call).
    state->max_monitor_line_size = 8;
    state->wfet_slice_ticks = spinloop_cntfrq() / 1000;  // park <= 1 ms
  }
  fprintf(stderr, "[spinloop] aarch64 wait: %s\n",
          state->cpu_support == CPU_SUPPORT_WFET ? "WFET"
                                                 : "yield");
#endif
}

static inline bool wfet_parks(const spinloop_state_t* state, bool qualifies) {
  return qualifies && state->cpu_support == CPU_SUPPORT_WFET;
}

static PyObject* method_spinloop(PyObject* self, PyObject* args,
                                 PyObject* kwargs) {
  Py_buffer buffer;
  PyObject* callback;
  double timeout = 0.;

  spinloop_state_t* state = (spinloop_state_t*)PyModule_GetState(self);
  if (state == NULL) {
    PyErr_SetString(PyExc_TypeError, "Failed to retrieve module state!");
    return NULL;
  }

  static const char* keywords[] = {"buffer", "callback", "timeout", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*O|d", (char**)keywords,
                                   &buffer, &callback, &timeout)) {
    return NULL;
  }

  if (!PyCallable_Check(callback)) {
    PyErr_SetString(PyExc_TypeError, "callback parameter must be callable!");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  struct timespec t_start;
  if (clock_gettime(TIMEOUT_CLOCK, &t_start) != 0) {
    PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
    PyBuffer_Release(&buffer);
    return NULL;
  }

  bool result = false;
  bool error = false;
  bool have_timeout = (timeout > 1e-9);
  unsigned int iteration = 0;
  bool buffer_qualifies = (buffer.len <= state->max_monitor_line_size);

#if defined(__aarch64__)
  // WFET compares the aligned 64-bit word that holds the whole buffer.
  volatile uint64_t* wfet_word = NULL;
  uint64_t wfet_snapshot = 0;
  if (state->cpu_support == CPU_SUPPORT_WFET) {
    const uintptr_t first = (uintptr_t)buffer.buf;
    const uintptr_t last = first + (uintptr_t)buffer.len - 1;
    buffer_qualifies = buffer.len > 0 && (first & ~(uintptr_t)7) == (last & ~(uintptr_t)7);
    if (buffer_qualifies) {
      wfet_word = (volatile uint64_t*)(first & ~(uintptr_t)7);
    }
  }
#endif

  while (true) {
#if defined(__aarch64__)
    // Snapshot BEFORE the callback reads the buffer: any store that lands
    // after this point either is seen by the callback, or makes the word
    // differ from the snapshot, or clears the monitor armed below.
    if (wfet_word != NULL) {
      wfet_snapshot = __atomic_load_n(wfet_word, __ATOMIC_ACQUIRE);
    }
#endif
    PyObject* res = PyObject_CallNoArgs(callback);
    if (res == NULL) {
      error = true;
      break;
    }
    int ok = (res == Py_True);
    Py_DECREF(res);

    if (ok) {
      result = true;
      break;
    }

    // Check timeout at most every 16 iterations to avoid clock_gettime and
    // comparison cost (every iteration when parking: each one can be long)
    if (have_timeout && ((iteration & 15u) == 0 || wfet_parks(state, buffer_qualifies))) {
      struct timespec t_now;
      if (clock_gettime(TIMEOUT_CLOCK, &t_now) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "clock_gettime() failed!");
        error = true;
        break;
      }

      const double elapsed = (double)(t_now.tv_sec - t_start.tv_sec) +
                             (t_now.tv_nsec - t_start.tv_nsec) * 1e-9;
      if (elapsed >= timeout) {
        result = false;
        break;
      }
    }
    ++iteration;

#if defined(__i386__) || defined(__x86_64__)
    // monitorx + mwaitx with qualified buffer
    if (buffer_qualifies && state->cpu_support == CPU_SUPPORT_MONITORX) {
      _mm_monitorx(buffer.buf, 0, 0);

      // Check once more in case the buffer has been modified while we were
      // arming the monitor hardware
      res = PyObject_CallNoArgs(callback);
      if (res == NULL) {
        error = true;
        break;
      }
      ok = (res == Py_True);
      Py_DECREF(res);

      if (ok) {
        result = true;
        break;
      }

      // Run mwaitx with enabled timeout (bit 1). The actual timeout value
      // is not very important, we just want to ensure we don't lock up
      // here for too long.
      Py_BEGIN_ALLOW_THREADS _mm_mwaitx((1 << 1), 0,
                                        MWAITX_DEFAULT_TIMEOUT_CYCLES);
      Py_END_ALLOW_THREADS
    }

    // Fallback: Busy poll
    else {
#endif
#if defined(__aarch64__)
    if (buffer_qualifies && state->cpu_support == CPU_SUPPORT_WFET) {
      uint64_t deadline = spinloop_cntvct() + state->wfet_slice_ticks;
      if (have_timeout) {
        struct timespec t_now;
        clock_gettime(TIMEOUT_CLOCK, &t_now);
        const double left = timeout - ((double)(t_now.tv_sec - t_start.tv_sec) +
                                       (t_now.tv_nsec - t_start.tv_nsec) * 1e-9);
        const double left_ticks = left > 0 ? left * (double)(state->wfet_slice_ticks * 1000) : 0;
        if (left_ticks < (double)state->wfet_slice_ticks) {
          deadline = spinloop_cntvct() + (uint64_t)left_ticks;
        }
      }
      // GIL released before the monitor is armed: the release itself
      // touches other memory and must not run between LDXR and WFET.
      Py_BEGIN_ALLOW_THREADS
      spinloop_wait_change_u64(wfet_word, wfet_snapshot, deadline);
      Py_END_ALLOW_THREADS
      continue;
    }
#endif
      // Give other threads a chance to be scheduled
      Py_BEGIN_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield" :: : "memory");
#endif
      Py_END_ALLOW_THREADS
#if defined(__i386__) || defined(__x86_64__)
    }
#endif
  }

  PyBuffer_Release(&buffer);

  if (error) {
    return NULL;
  }

  if (result) {
    Py_RETURN_TRUE;
  }

  Py_RETURN_FALSE;
}

static PyMethodDef spinloop_methods[] = {
    {"spinloop", (PyCFunction)method_spinloop, METH_VARARGS | METH_KEYWORDS,
     "Wait for store with callback"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef spinloop_module = {
    PyModuleDef_HEAD_INIT, "spinloop",
    "Hardware-optimized spinloops for Python", sizeof(spinloop_state_t),
    spinloop_methods};

PyMODINIT_FUNC PyInit_spinloop(void) {
  PyObject* m = PyModule_Create(&spinloop_module);
  if (m != NULL) {
    spinloop_state_t* state = (spinloop_state_t*)PyModule_GetState(m);
    if (state != NULL) {
      determine_cpu_support(state);
    }
  }
  return m;
}

}  // extern "C"
