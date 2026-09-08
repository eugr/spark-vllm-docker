"""World-size-dispatched PCIe all-reduce runtime."""

from __future__ import annotations

import logging
import os

from contextlib import ExitStack, contextmanager
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .pcie_hierarchical import (
    SUPPORTED_WORLD_SIZES as HIERARCHICAL_WORLD_SIZES,
)
from .pcie_hierarchical import (
    PCIeHierarchicalAllReduce,
)
from .pcie_island_rs import (
    CROSSOVER_ELEMENTS as ISLAND_RS_CROSSOVER_ELEMENTS,
)
from .pcie_island_rs import (
    PREFERRED_ALIGNMENT_ELEMENTS as ISLAND_RS_PREFERRED_ALIGNMENT_ELEMENTS,
)
from .pcie_island_rs import (
    SUPPORTED_WORLD_SIZES as ISLAND_RS_WORLD_SIZES,
)
from .pcie_island_rs import (
    PCIeIslandRSAllReduce,
)
from .pcie_oneshot import (
    DEFAULT_MAX_SIZE,
    DEFAULT_RANK_DATA_BYTES,
    SUPPORTED_WORLD_SIZES as ONESHOT_WORLD_SIZES,
    TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_BYTES,
    PCIeOneshotAllReducePool,
    _tp2_plain_remote_push_enabled,
)


logger = logging.getLogger(__name__)


# Maximum message capacity qualified for the TP16 island runtime. Callers use
# this shared policy instead of duplicating an implementation-specific limit.
ISLAND_RS_MAX_BYTES = 160 * 1024


def _algorithm_override() -> str:
    """Select the established runtime or enable size-routed island dispatch."""

    choice = os.getenv("B12X_PCIE_ALLREDUCE_ALGORITHM", "auto").strip().lower()
    if choice not in ("auto", "hierarchical", "island_rs"):
        raise ValueError(
            "B12X_PCIE_ALLREDUCE_ALGORITHM must be auto, hierarchical or "
            f"island_rs, got {choice!r}"
        )
    return choice


def recommended_max_bytes(world_size: int, *, default: int = DEFAULT_MAX_SIZE) -> int:
    """Largest all-reduce this runtime expects to win at, for this world size.

    Callers that force the equal-quarter implementation advertise its complete
    qualified capacity. Automatic consumers retain their existing limit
    because a CUDA-graph caller without explicit output storage cannot use the
    equal-quarter implementation for large messages and may otherwise replace
    a faster fallback collective with hierarchical all-reduce.
    """

    if world_size in ISLAND_RS_WORLD_SIZES and _algorithm_override() == "island_rs":
        return max(default, ISLAND_RS_MAX_BYTES)
    if world_size == 2 and _tp2_plain_remote_push_enabled():
        return max(default, TP2_PLAIN_REMOTE_PUSH_AUTO_MAX_BYTES)
    return default


MAX_DIRECT_WORLD_SIZE = 8
DIRECT_WORLD_SIZES = tuple(
    world_size
    for world_size in ONESHOT_WORLD_SIZES
    if world_size <= MAX_DIRECT_WORLD_SIZE
)
SUPPORTED_WORLD_SIZES = (*DIRECT_WORLD_SIZES, *HIERARCHICAL_WORLD_SIZES)


def _algorithm_for_world_size(world_size: int) -> str:
    if world_size in DIRECT_WORLD_SIZES:
        return "oneshot"
    if world_size in HIERARCHICAL_WORLD_SIZES:
        return "hierarchical"
    raise ValueError(
        f"unsupported PCIe all-reduce world size {world_size}; "
        f"supported world sizes are {SUPPORTED_WORLD_SIZES}"
    )


class PCIeAllReduce:
    """Select a peer-safe all-reduce implementation from the world size.

    Worlds through TP8 use the low-latency all-peer oneshot runtime. TP12 and
    TP16 use bounded-degree four-GPU islands so no CUDA context maps more than
    six peers. Other worlds fail closed instead of exceeding the CUDA peer
    connection limit.
    """

    def __init__(
        self,
        runtime: Any,
        algorithm: str,
        island_rs: Any = None,
        *,
        algorithm_override: Optional[str] = None,
    ) -> None:
        self._runtime = runtime
        # Optional second implementation for the same world. When present the
        # dispatcher routes by message size instead of exposing a knob.
        self._island_rs = island_rs
        resolved_override = (
            _algorithm_override()
            if algorithm_override is None
            else str(algorithm_override)
        )
        if resolved_override not in ("auto", "hierarchical", "island_rs"):
            raise ValueError(
                "algorithm_override must be auto, hierarchical or island_rs, "
                f"got {resolved_override!r}"
            )
        self._algorithm_override = resolved_override
        self.algorithm = algorithm
        self.rank = runtime.rank
        self.world_size = runtime.world_size
        self.device = runtime.device

    @classmethod
    def from_exchange_group(
        cls,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        eager_buffer_bytes: int = DEFAULT_MAX_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
        max_concurrent_channels: int = 1,
    ) -> "PCIeAllReduce":
        world_size = dist.get_world_size(group=exchange_group)
        algorithm = _algorithm_for_world_size(world_size)
        algorithm_override = _algorithm_override()
        if algorithm == "oneshot":
            runtime = PCIeOneshotAllReducePool.from_exchange_group(
                exchange_group=exchange_group,
                device=device,
                eager_buffer_bytes=eager_buffer_bytes,
                max_size=max_size,
                rank_data_bytes=rank_data_bytes,
                ext_module=ext_module,
                single_channel=single_channel,
                max_concurrent_channels=max_concurrent_channels,
            )
        else:
            if int(max_concurrent_channels) != 1:
                raise ValueError(
                    "hierarchical all-reduce supports exactly one concurrent channel"
                )
            if max_size < torch.bfloat16.itemsize:
                raise ValueError("max_size must hold at least one BF16 element")
            runtime = PCIeHierarchicalAllReduce(
                exchange_group=exchange_group,
                device=device,
                max_elements=max_size // torch.bfloat16.itemsize,
                ext_module=ext_module,
            )
        island_rs = cls._maybe_island_rs(
            exchange_group=exchange_group,
            device=device,
            max_size=max_size,
            algorithm_override=algorithm_override,
        )
        return cls(
            runtime,
            algorithm,
            island_rs,
            algorithm_override=algorithm_override,
        )

    @staticmethod
    def _maybe_island_rs(
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_size: int,
        algorithm_override: str,
    ) -> Any:
        """Attach the equal-quarter runtime after an explicit policy opt-in.

        Opt-in capacity includes :data:`ISLAND_RS_MAX_BYTES` independently of
        the caller's ``max_size``. A coordinated construction failure leaves
        every rank on the hierarchical runtime and records the reason in the
        process log.
        """

        world_size = dist.get_world_size(group=exchange_group)
        if world_size not in ISLAND_RS_WORLD_SIZES:
            return None
        # The equal-quarter runtime owns an additional CUDA IPC slab and has a
        # stricter CUDA-graph output contract than the hierarchical runtime.
        # Construct it only for an explicit opt-in so unrelated auxiliary IPC
        # collectives retain the established peer-mapping and setup behavior.
        if algorithm_override != "island_rs":
            return None
        capacity = max(int(max_size), ISLAND_RS_MAX_BYTES)
        elements = capacity // torch.bfloat16.itemsize
        try:
            return PCIeIslandRSAllReduce(
                exchange_group=exchange_group,
                device=device,
                max_elements=elements - (elements % 2),
            )
        except RuntimeError as exc:
            if dist.get_rank(group=exchange_group) == 0:
                logger.warning(
                    "PCIe island reduce-scatter is unavailable; using the "
                    "hierarchical all-reduce runtime: %s",
                    exc,
                )
            return None

    @classmethod
    def from_process_group(
        cls,
        *,
        process_group: ProcessGroup,
        device: torch.device | int | str,
        max_input_bytes: int = DEFAULT_MAX_SIZE,
        eager_buffer_bytes: Optional[int] = None,
        max_size: int = DEFAULT_MAX_SIZE,
        rank_data_bytes: int = DEFAULT_RANK_DATA_BYTES,
        ext_module=None,
        single_channel: bool = False,
        max_concurrent_channels: int = 1,
    ) -> "PCIeAllReduce":
        return cls.from_exchange_group(
            exchange_group=process_group,
            device=device,
            eager_buffer_bytes=(
                max_input_bytes if eager_buffer_bytes is None else eager_buffer_bytes
            ),
            max_size=max_size,
            rank_data_bytes=rank_data_bytes,
            ext_module=ext_module,
            single_channel=single_channel,
            max_concurrent_channels=max_concurrent_channels,
        )

    @property
    def supports_all_peer_auxiliary(self) -> bool:
        """Whether another runtime may safely map every rank as a peer."""

        return self.algorithm == "oneshot"

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        """Prepare the runtime's semantic channel owners."""
        self._runtime.prepare_channels(channel_ids)
        if self._island_rs is not None:
            self._island_rs.prepare_channels(channel_ids)

    def for_stream(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ):
        if self._island_rs is not None:
            self._island_rs.for_stream(stream, channel_id=channel_id)
        return self._runtime.for_stream(stream, channel_id=channel_id)

    def _use_island_rs(
        self,
        inp: torch.Tensor,
    ) -> bool:
        """Route aligned large messages to the equal-quarter runtime.

        Small ones stay on the hierarchy, whose critical path is shorter for the
        ranks that are not the island leader. Unaligned quarters also stay on
        the hierarchy because the equal-quarter kernel's partial transfer group
        costs more than the leader path. Large aligned vectors would otherwise
        funnel the whole vector through the island leader's PCIe link.
        """

        if self._island_rs is None:
            return False
        override = self._algorithm_override
        if override == "hierarchical":
            return False
        island_accepts = self._island_rs.should_allreduce(inp)
        if not island_accepts:
            return False
        hierarchy_accepts = self._runtime.should_allreduce(inp)
        if not hierarchy_accepts:
            return True
        return (
            inp.numel() > ISLAND_RS_CROSSOVER_ELEMENTS
            and inp.numel() % ISLAND_RS_PREFERRED_ALIGNMENT_ELEMENTS == 0
        )

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._runtime.should_allreduce(inp):
            return True
        return self._use_island_rs(inp)

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        peer_input_ptrs: Optional[Sequence[int]] = None,
        blocks: Optional[int] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        if self.algorithm == "hierarchical":
            if peer_input_ptrs is not None:
                raise ValueError(
                    "peer_input_ptrs are unavailable for hierarchical all-reduce"
                )
            if self._use_island_rs(inp):
                return self._island_rs.all_reduce(
                    inp,
                    out=out,
                    blocks=blocks,
                    stream=stream,
                    channel_id=channel_id,
                )
            return self._runtime.all_reduce(
                inp,
                out=out,
                blocks=blocks,
                stream=stream,
                channel_id=channel_id,
            )
        if blocks is not None:
            raise ValueError("blocks is only available for hierarchical all-reduce")
        return self._runtime.all_reduce(
            inp,
            out=out,
            peer_input_ptrs=peer_input_ptrs,
            stream=stream,
            channel_id=channel_id,
        )

    @contextmanager
    def capture(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ):
        with ExitStack() as stack:
            stack.enter_context(
                self._runtime.capture(stream=stream, channel_id=channel_id)
            )
            if self._island_rs is not None:
                stack.enter_context(
                    self._island_rs.capture(stream=stream, channel_id=channel_id)
                )
            # Callers must retain message-size dispatch while recording a
            # graph; yielding the hierarchy would bypass the island runtime.
            yield self

    def close(self) -> None:
        if self._island_rs is not None:
            self._island_rs.close()
            self._island_rs = None
        self._runtime.close()

    def __getattr__(self, name: str):
        runtime = self.__dict__.get("_runtime")
        if runtime is None:
            raise AttributeError(name)
        return getattr(runtime, name)


__all__ = [
    "DIRECT_WORLD_SIZES",
    "MAX_DIRECT_WORLD_SIZE",
    "PCIeAllReduce",
    "SUPPORTED_WORLD_SIZES",
]
