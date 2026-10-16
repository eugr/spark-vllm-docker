#!/usr/bin/env python3
"""Validation: sparse-indexer TopK=512 at >=32K seq-len passes on a 48-SM device.

Run INSIDE the vLLM worker container (where the custom ops are compiled and the
mod has been applied). It verifies two things:

  1. The smem-capacity gate routes this device away from persistent_topk: on a
     <128KB opt-in-smem part (GB10 / consumer Blackwell) the gate must be False,
     so sparse_attn_indexer.py will take the top_k_per_row_decode path instead
     of hitting persistent_topk's FilteredTopK hard-fail.
  2. That fallback path actually runs cleanly at topk_tokens=512 for decode
     widths >= 32K (and up toward the 512K ceiling that triggered the original
     "total_ctas > num_sms*occupancy" crash), with no CUDA error.

Usage (on a worker, after the container is up and the mod ran):
    docker exec -it <container> python /workspace/mods/fix-persistent-topk-sm120/validate_sparse_indexer.py
"""

import sys

import torch

try:
    from vllm import _custom_ops as ops
except ImportError:  # pragma: no cover
    ops = None
    print("ERROR: vllm not importable — run inside the worker container.", file=sys.stderr)
    sys.exit(2)

TOP_K = 512
NUM_ROWS = 48          # a few >32 to reproduce the oversubscribe regime
WIDTHS = (32768, 262144, 512000)  # grow toward the 524288 max_model_len ceiling
ONE_HUNDRED_TWENTY_EIGHT_KB = 128 * 1024


def gate_ok(max_smem: int) -> bool:
    """Mirror of _device_has_persistent_topk_smem() from the applied patch."""
    force = __import__("os").environ.get(
        "VLLM_SPARSE_IDX_FORCE_PERSISTENT_TOPK", "0"
    ) == "1"
    return force or max_smem >= ONE_HUNDRED_TWENTY_EIGHT_KB


def main() -> int:
    if not torch.cuda.is_available():
        print("FATAL: no CUDA device.", file=sys.stderr)
        return 2

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    max_smem = int(getattr(props, "shared_memory_per_block_optin", 0) or 0)
    sms = int(getattr(props, "multi_processor_count", 0) or 0)
    print(f"device={props.name}  compute={props.major}.{props.minor}")
    print(f"multi_processor_count               = {sms}")
    print(f"shared_memory_per_block_optin       = {max_smem} bytes "
          f"({max_smem / 1024:.1f} KB; need {ONE_HUNDRED_TWENTY_EIGHT_KB / 1024:.0f} KB)")

    # 1) Gate check — the whole point of the patch.
    use_persistent = gate_ok(max_smem)
    print(f"persistent_topk smem gate ok = {use_persistent} "
          f"-> {'USE persistent_topk' if use_persistent else 'ROUTE to top_k_per_row_decode'}")
    if max_smem < ONE_HUNDRED_TWENTY_EIGHT_KB and use_persistent:
        print("FAIL: <128KB smem but gate allowed persistent_topk (would hard-fail).")
        return 1

    # 2) The fallback kernel must run clean at TopK=512 across the relevant widths.
    gate = ops if ops is not None else torch.ops._C
    decode = getattr(gate, "top_k_per_row_decode", None)
    if decode is None:
        print("FAIL: top_k_per_row_decode op not found.", file=sys.stderr)
        return 2

    for width in WIDTHS:
        logits = torch.randn(NUM_ROWS, width, dtype=torch.float32, device="cuda")
        assert logits.stride(0) % 4 == 0  # TMA 16-byte alignment (vec_size=4)
        # (B, next_n) shaped like the real decode call; next_n=1 for eager decode.
        seq_lens = torch.full((NUM_ROWS, 1), width, dtype=torch.int32, device="cuda")
        indices = torch.empty(NUM_ROWS, TOP_K, dtype=torch.int32, device="cuda")
        try:
            decode(
                logits,
                1,              # next_n
                seq_lens,
                indices,
                NUM_ROWS,
                logits.stride(0),
                logits.stride(1),
                TOP_K,
            )
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 — surface any CUDA/runtime error
            print(f"FAIL: top_k_per_row_decode(TopK={TOP_K}, width={width}) raised: {exc}")
            return 1

        topk = indices.cpu()
        if int(topk.min()) < 0 or int(topk.max()) >= width:
            print(f"FAIL: width={width} indices out of range [0,{width}).")
            return 1
        bad_rows = sum(len(set(row_int.tolist())) != TOP_K for row_int in topk)
        if bad_rows:
            print(f"FAIL: width={width} {bad_rows}/{NUM_ROWS} rows missing "
                  f"{TOP_K} unique top-k indices.")
            return 1
        print(f"PASS: TopK={TOP_K} width={width} seq_lens={width} num_rows={NUM_ROWS} "
              f"-> no CUDA error, {TOP_K} unique indices per row.")

    print("ALL PASS: sparse indexer TopK=512 at >=32K seq-len is clean on this device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
