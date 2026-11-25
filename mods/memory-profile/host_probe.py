#!/usr/bin/env python3
"""Read-only Linux/CUDA inventory, also executable over SSH on stdin.

Uses the driver API without creating a CUDA context or importing Torch.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import socket


def inventory():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('MemTotal', 'MemAvailable'):
            memory[key] = int(value.split()[0]) * 1024
    result = {
        'hostname': socket.gethostname(),
        'host_id': hashlib.sha256(Path('/proc/sys/kernel/random/boot_id').read_bytes().strip()).hexdigest()[:16],
        'memory': memory, 'gpus': [],
        'wsl': 'microsoft' in platform.release().lower(),
    }
    try:
        cuda = ctypes.CDLL('libcuda.so.1')

        def call(name, *args):
            status = getattr(cuda, name)(*args)
            if status:
                raise RuntimeError(f'{name} failed (CUDA status {status})')

        call('cuInit', 0)
        count = ctypes.c_int()
        call('cuDeviceGetCount', ctypes.byref(count))
        for index in range(count.value):
            device = ctypes.c_int()
            call('cuDeviceGet', ctypes.byref(device), index)
            name, total = ctypes.create_string_buffer(256), ctypes.c_size_t()
            call('cuDeviceGetName', name, len(name), device)
            call('cuDeviceTotalMem_v2', ctypes.byref(total), device)
            attrs = []
            for attribute in (75, 76, 18):  # compute capability major/minor, integrated
                value = ctypes.c_int()
                call('cuDeviceGetAttribute', ctypes.byref(value), attribute, device)
                attrs.append(value.value)
            result['gpus'].append({'name': name.value.decode(), 'total_memory_bytes': total.value,
                                   'compute_capability': attrs[:2], 'integrated': bool(attrs[2])})
    except (OSError, RuntimeError, AttributeError) as error:
        result['gpu_error'] = str(error)
    return result


if __name__ == '__main__':
    print(json.dumps(inventory()))
