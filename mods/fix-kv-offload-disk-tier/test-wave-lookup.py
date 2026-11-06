#!/usr/bin/env python3
"""Checks for patch 04 (wave readiness evaluated at load, never remembered).

1. Replays a remembered-then-evicted wave key on vLLM's real
   CPUOffloadingManager: a wave key becomes resident (ref_cnt 0, evictable)
   while the rest of its wave is still promoting, another request's store
   evicts it, then the rest lands. A driver that remembered the first key as
   ready would call prepare_load and die on `assert block is not None`.
   wave_lookup must answer HIT_PENDING while a promotion is writing, MISS
   after the eviction, and HIT only when prepare_load is safe.
2. The wave-retry counter reaches Prometheus with its `reason` label.

Run inside an image with the patch applied (no GPU, no model):
  docker run --rm --entrypoint python3 -v "$PWD/<this file>:/t.py:ro" <image> /t.py
"""
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
    _ConnectorMetricName,
)
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager

# 1. readiness
A, B, C, D = (bytes([i]) * 36 for i in range(4))
ctx = ReqContext(req_id="wave")
other = ReqContext(req_id="other")
primary = CPUOffloadingManager(num_blocks=2)
tiering = SimpleNamespace(primary_tier=primary)


def wave_lookup(keys):
    return TieringOffloadingManager.wave_lookup(tiering, keys, ctx)


def store(keys, req_context, complete=True):
    assert primary.prepare_store(keys, req_context) is not None
    if complete:
        primary.complete_store(keys, req_context)


store([A], ctx)                       # A resident, evictable
store([B], ctx, complete=False)       # B still promoting
assert wave_lookup([A, B]) is LookupResult.HIT_PENDING

store([C], other)                     # another request's store evicts A (LRU)
primary.complete_store([B], ctx)      # the rest of the wave lands
assert wave_lookup([A, B]) is LookupResult.MISS
try:
    primary.prepare_load([A, B], ctx)
except AssertionError as e:           # what a remembered readiness would hit
    assert "not found in cache" in str(e)
else:
    raise SystemExit("expected prepare_load to reject an evicted key")

store([A], ctx)                       # re-staged: evicts C, A and B resident
assert wave_lookup([A, B]) is LookupResult.HIT
primary.prepare_load([A, B], ctx)     # pins both; safe
assert primary.prepare_store([D], other) is None  # pinned rows are not evictable
print("OK wave_lookup: pending -> evicted MISS -> re-staged HIT, prepare_load safe")

# 2. counter
stats = OffloadingConnectorStats()
for reason in ("miss", "miss", "promote_refused"):
    stats.increase_counter(_ConnectorMetricName.WAVE_RETRY, labelvalues=(reason,))


class NoSpecMetrics:  # OffloadPromMetrics requires a spec class
    @staticmethod
    def build_metric_definitions(extra_config):
        return {}


with patch.object(OffloadingSpecFactory, "get_spec_cls", return_value=NoSpecMetrics):
    prom = OffloadPromMetrics(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
        ),
        metric_types={Gauge: Gauge, Counter: Counter, Histogram: Histogram},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["m", "0"]},
    )
prom.observe(stats.data)
text = generate_latest(REGISTRY).decode()
for reason, value in (("miss", "2.0"), ("promote_refused", "1.0")):
    sample = (f'vllm:kv_offload_wave_retry_total{{engine="0",model_name="m",'
              f'reason="{reason}"}} {value}')
    assert sample in text, sample
print("OK vllm:kv_offload_wave_retry_total{reason} exported")
