# SPDX-License-Identifier: Apache-2.0
# Source: scitix/InstantTensor, PyPI 0.2.3 aarch64 CPython 3.12 wheel.
# Wheel SHA256: 6d71a64e7b406af396b71400b9d56d0cd9e963cc9f1509a4701c47faad49e18a
"""Negotiate CPU, host-memory, and device-memory budgets among group ranks."""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import math
import os
import struct

import torch

from . import cgroup


@dataclass(frozen=True)
class RankResources:
    host_boot_id: str
    cpu_count: int
    affinity: frozenset[int]
    avail_mem: int
    cgroups: cgroup.CgroupSnapshot
    device_free_mem: int
    is_integrated: bool
    max_free_mem_usage: float


@dataclass(frozen=True)
class ResourceBudget:
    avail_cpus: int
    avail_mem: int
    avail_device_mem: int


def host_available_bytes():
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise ValueError("Missing MemAvailable in /proc/meminfo")


def query_rank_resources(device, *, max_free_mem_usage):
    boot_id = cgroup.get_host_boot_id()
    try:
        snapshot = cgroup.query_resources(host_boot_id=boot_id)
    except (OSError, ValueError):
        # Preserve host/affinity limits when no accessible v2 ancestry exists.
        snapshot = cgroup.CgroupSnapshot((), False)
    try:
        avail_mem = host_available_bytes()
    except (OSError, ValueError):
        # Participate in negotiation even if this rank cannot read MemAvailable.
        avail_mem = 0
    try:
        # Managed-memory support alone does not imply shared physical memory.
        integrated = torch.cuda.get_device_properties(device).is_integrated
        free_mem = torch.cuda.mem_get_info(device)[0]
    except (OSError, ValueError, AttributeError, RuntimeError):
        # A negative value reports query failure to every rank before admission.
        free_mem, integrated = -1, False
    return RankResources(boot_id, os.cpu_count() or 1,
                         frozenset(os.sched_getaffinity(0)), avail_mem, snapshot,
                         free_mem, integrated, max_free_mem_usage)


def allocate_resources(ranks: list[RankResources]) -> tuple[ResourceBudget, ...]:
    """Return each rank's bounds, counting only ranks in the supplied group."""
    hosts = Counter(rank.host_boot_id for rank in ranks)
    groups = Counter(key for rank in ranks
                     for key in {entry.key for entry in rank.cgroups.entries})
    cores = Counter((rank.host_boot_id, cpu) for rank in ranks for cpu in rank.affinity)
    budgets = []
    for rank in ranks:
        host_users = hosts[rank.host_boot_id]
        # Exact reciprocals avoid flooring e.g. six shares of 1/6 to zero.
        affinity_share = sum((Fraction(1, cores[rank.host_boot_id, cpu])
                              for cpu in rank.affinity), Fraction())
        cpu_bounds = [max(rank.cpu_count // host_users, 1),
                      max(affinity_share.numerator // affinity_share.denominator, 1)]
        mem_bounds = [max(rank.avail_mem // host_users, 0)]
        for entry in rank.cgroups.entries:
            users = groups[entry.key]
            if entry.avail_cpus is not None:
                cpu_bounds.append(max(entry.avail_cpus // users, 1))
            if entry.avail_mem is not None:
                mem_bounds.append(max(entry.avail_mem // users, 0))
        budgets.append(ResourceBudget(min(cpu_bounds), min(mem_bounds), rank.device_free_mem))
    return tuple(budgets)


def encode_rank(rank):
    values = cgroup.encode_key(cgroup.CgroupKey(rank.host_boot_id, 0, 0, 0))[:4]
    values += [rank.cpu_count, rank.avail_mem, len(rank.affinity),
               len(rank.cgroups.entries), int(rank.cgroups.ancestors_complete)]
    # Preserve the fraction's float64 bits in the int64 resource record.
    values += [rank.device_free_mem, int(rank.is_integrated),
               struct.unpack("q", struct.pack("d", rank.max_free_mem_usage))[0]]
    values.extend(sorted(rank.affinity))
    for entry in rank.cgroups.entries:
        values.extend(cgroup.encode_key(entry.key))
        values.extend([entry.avail_cpus if entry.avail_cpus is not None else -1,
                       entry.avail_mem if entry.avail_mem is not None else -1])
    return values


def decode_rank(values):
    boot_id = cgroup.decode_key(values[:4] + [0] * 4).host_boot_id
    cpu_count, avail_mem, affinity_size, entry_count, complete = values[4:9]
    device_free_mem, is_integrated, fraction_bits = values[9:12]
    fraction = struct.unpack("d", struct.pack("q", fraction_bits))[0]
    offset = 12 + affinity_size
    entries = []
    for index in range(entry_count):
        start = offset + 10 * index
        key = cgroup.decode_key(values[start:start + 8])
        cpus, memory = values[start + 8:start + 10]
        entries.append(cgroup.CgroupResources(key, "", None if cpus == -1 else cpus,
                                               None if memory == -1 else memory))
    return RankResources(boot_id, cpu_count, frozenset(values[12:offset]), avail_mem,
                         cgroup.CgroupSnapshot(tuple(entries), bool(complete)),
                         device_free_mem, bool(is_integrated), fraction)


def negotiate_resources(process_group, *, device, max_free_mem_usage) -> ResourceBudget:
    """Agree on CPU, host-memory, and CUDA device-memory budgets.

    Exchange snapshots once, then compute the same per-rank bounds and group
    minimum locally. No additional collective is needed for the final minimum.
    """
    local = query_rank_resources(device, max_free_mem_usage=max_free_mem_usage)
    rows = cgroup.gather_records(encode_rank(local), process_group, device)
    ranks = [decode_rank(row) for row in rows]
    if any(rank.device_free_mem < 0 for rank in ranks):
        raise RuntimeError("Cannot query CUDA resources on at least one rank")
    if any(not math.isfinite(rank.max_free_mem_usage) or
           not 0 < rank.max_free_mem_usage <= 1 for rank in ranks):
        raise ValueError("max_free_mem_usage must be finite and satisfy 0 < value <= 1 on every rank")
    if any(rank.max_free_mem_usage != ranks[0].max_free_mem_usage for rank in ranks):
        raise ValueError("max_free_mem_usage must be identical on every rank")
    if any(rank.is_integrated != ranks[0].is_integrated for rank in ranks):
        raise ValueError("is_integrated must be identical on every rank")
    budgets = allocate_resources(ranks)
    fraction = ranks[0].max_free_mem_usage
    host_memory = int(min(b.avail_mem for b in budgets) * fraction)
    if ranks[0].is_integrated:
        # Host staging and the device buffer consume the same physical memory.
        host_memory //= 2
        device_memory = host_memory
    else:
        device_memory = int(min(b.avail_device_mem for b in budgets) * fraction)
    if device_memory == 0:
        # Every rank must reject before the subsequent backend collectives.
        raise RuntimeError("Cannot select memory budget: device memory budget is zero on at least one rank")
    return ResourceBudget(min(b.avail_cpus for b in budgets),
                          host_memory, device_memory)
