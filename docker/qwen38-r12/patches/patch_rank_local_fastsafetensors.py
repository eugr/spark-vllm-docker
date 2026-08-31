#!/usr/bin/env python3
"""Make fastsafetensors safe and rank-local for vLLM expert parallelism.

The upstream vLLM integration uses the world process group.  On two DGX
Sparks that makes one node read a checkpoint file and broadcast all of its
tensors to the other node, even when each EP rank only owns half the experts
and both nodes have a local copy on NVMe.  It also lets PyTorch retain the
temporary file buffers in its CUDA caching allocator, which is especially
dangerous on unified-memory systems.

This opt-in patch:

* passes vLLM's already-computed ``local_expert_ids`` to fastsafetensors;
* lets every rank read its own NVMe with a per-rank tensor filter;
* permits vLLM's immediate-copy consumer to avoid a redundant clone; and
* optionally empties temporary CUDA allocator cache at bounded file-batch
  intervals, including the one-shard-prefetch mode.

All behavior is gated by environment variables set only by the audited hybrid
candidate launcher.
"""

from __future__ import annotations

import argparse
from pathlib import Path


WEIGHT_UTILS_MARKER = "QWEN_RANK_LOCAL_FASTSAFETENSORS_V1"
PARALLEL_LOADER_MARKER = "QWEN_FASTSAFETENSORS_BATCH_CACHE_RELEASE_V1"


def patch_weight_utils(source: str) -> str:
    if WEIGHT_UTILS_MARKER in source:
        raise ValueError("weight_utils.py is already rank-local patched")

    old_signature = '''def fastsafetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
'''
    new_signature = f'''def fastsafetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    local_expert_ids: set[int] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    # {WEIGHT_UTILS_MARKER}
'''

    old_setup = '''    device = torch.device(f"cuda:{current_platform.current_device()}")
    hf_weights_files = sorted(hf_weights_files, key=_natural_sort_key)

    # Use nogds=True for TP > 1 to avoid cuFileDriverOpen() which
'''
    new_setup = '''    device = torch.device(f"cuda:{current_platform.current_device()}")
    hf_weights_files = sorted(hf_weights_files, key=_natural_sort_key)

    explicit_num_experts = int(
        os.environ.get("QWEN_FASTSAFETENSORS_EP_NUM_EXPERTS", "0")
    )
    if local_expert_ids is None and explicit_num_experts > 0 and pg.size() > 1:
        # Some unreleased model adapters do not yet expose their expert count
        # through ModelConfig.get_num_experts(), so vLLM's generic prefilter
        # remains unset.  The audited launcher may provide the checkpoint's
        # frozen expert count explicitly; derive the same linear placement used
        # by this service rather than silently reverting to world-group loading.
        ep_size = pg.size()
        ep_rank = pg.rank()
        base, remainder = divmod(explicit_num_experts, ep_size)
        start = ep_rank * base + min(ep_rank, remainder)
        local_count = base + (1 if ep_rank < remainder else 0)
        local_expert_ids = set(range(start, start + local_count))

    rank_local_ep = (
        local_expert_ids is not None
        and os.environ.get("QWEN_FASTSAFETENSORS_EP_LOCAL", "0") == "1"
    )
    tensor_filter = None
    if rank_local_ep:
        tensor_filter = lambda name: not should_skip_weight(name, local_expert_ids)
        logger.warning_once(
            "Using rank-local fastsafetensors EP loading: each rank reads its "
            "own NVMe and skips non-local expert weights before I/O."
        )

    # Use nogds=True for TP > 1 to avoid cuFileDriverOpen() which
'''

    old_loader = '''            device=str(device),
            nogds=nogds,
        )
'''
    new_loader = '''            device=str(device),
            nogds=nogds,
            tensor_filter=tensor_filter,
            all_local=rank_local_ep,
        )
'''

    old_construct = '''            pl = _make_loader(nogds)
            for name, tensor in pl.iterate_weights():
'''
    new_construct = '''            pl = _make_loader(nogds)
            if rank_local_ep and os.environ.get(
                "QWEN_FASTSAFETENSORS_ASSUME_IMMEDIATE_COPY", "0"
            ) == "1":
                # vLLM copies every yielded tensor into its preallocated model
                # parameter before asking the iterator for another tensor.
                # The file batch therefore remains alive for the full copy and
                # an additional clone would only increase peak unified memory.
                pl.need_clone = False
            for name, tensor in pl.iterate_weights():
'''

    replacements = (
        (old_signature, new_signature, "iterator signature"),
        (old_setup, new_setup, "rank-local setup"),
        (old_loader, new_loader, "ParallelLoader construction"),
        (old_construct, new_construct, "immediate-copy contract"),
    )
    for old, new, label in replacements:
        if source.count(old) != 1:
            raise ValueError(f"expected {label} exactly once")
        source = source.replace(old, new)
    return source


def patch_default_loader(source: str) -> str:
    if WEIGHT_UTILS_MARKER in source:
        raise ValueError("default_loader.py is already rank-local patched")
    old = '''                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                )
'''
    new = f'''                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                    self.load_config.use_tqdm_on_load,
                    local_expert_ids=self.local_expert_ids,
                )  # {WEIGHT_UTILS_MARKER}
'''
    if source.count(old) != 2:
        raise ValueError("expected two fastsafetensors dispatches")
    return source.replace(old, new)


def patch_parallel_loader(source: str) -> str:
    if PARALLEL_LOADER_MARKER in source:
        raise ValueError("parallel_loader.py is already cache-release patched")
    old = '''        # sync
        if self.queue_size < 0 and self.consumer_processed is not None:
            self.consumer_processed.set()
'''
    new = f'''        # {PARALLEL_LOADER_MARKER}. With queue_size < 0 the
        # just-consumed file buffer is closed here. In queue_size=0 mode at
        # most one next shard may already be staged; empty_cache preserves that
        # live allocation while returning older cached blocks. This prevents
        # pinned staging from accumulating until it crowds out the model on
        # unified-memory systems.
        cache_release_interval = max(
            1,
            int(
                os.environ.get(
                    "QWEN_FASTSAFETENSORS_EMPTY_CACHE_INTERVAL", "1"
                )
            ),
        )
        if (
            self.queue_size <= 0
            and os.environ.get(
                "QWEN_FASTSAFETENSORS_EMPTY_CACHE_PER_BATCH", "0"
            ) == "1"
            and self.loader.framework.get_name() == "pytorch"
            and (
                (batch.batch_id + 1) % cache_release_interval == 0
                or batch.batch_id + 1 == len(self.weight_files_batches)
            )
        ):
            import torch

            torch.cuda.empty_cache()
        # sync
        if self.queue_size < 0 and self.consumer_processed is not None:
            self.consumer_processed.set()
'''
    if source.count(old) != 1:
        raise ValueError("expected serial batch release point exactly once")
    return source.replace(old, new)


def main() -> int:
    parser = argparse.ArgumentParser()
    site = Path("/usr/local/lib/python3.12/dist-packages")
    parser.add_argument(
        "--weight-utils",
        type=Path,
        default=site
        / "vllm/model_executor/model_loader/weight_utils.py",
    )
    parser.add_argument(
        "--default-loader",
        type=Path,
        default=site
        / "vllm/model_executor/model_loader/default_loader.py",
    )
    parser.add_argument(
        "--parallel-loader",
        type=Path,
        default=site / "fastsafetensors/parallel_loader.py",
    )
    args = parser.parse_args()
    args.weight_utils.write_text(patch_weight_utils(args.weight_utils.read_text()))
    args.default_loader.write_text(patch_default_loader(args.default_loader.read_text()))
    args.parallel_loader.write_text(
        patch_parallel_loader(args.parallel_loader.read_text())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
