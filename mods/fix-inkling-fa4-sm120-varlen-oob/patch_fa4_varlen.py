#!/usr/bin/env python3
"""Patch the vendored SM80/SM120 FA4 forward kernel for multi-request (varlen) batches.

    patch_fa4_varlen.py <path to inkling_sm120_fa4/flash_fwd.py>           # patch in place
    patch_fa4_varlen.py --check <path>                                     # report only

Two defects in FlashAttentionForwardSm80.kernel (also used by SM120), both only
reachable with more than one request in a batch. The SM90 and SM100 kernels in the
same bundle already do the right thing; this ports their behaviour.

A. Aux-tensor indices are wrapped "to not read OOB" using fastdiv_mods, which the
   host computes from the WHOLE batch (mQ.shape[0] rows). SM90/SM100 recompute them
   per request inside the kernel; SM80 did not. Inkling's score_mod reads
   rel_logits[offset_q + q_idx], so every request after the first indexed past the
   end of rel_logits on the padding rows of its tile (up to min(offset_q, padding)
   rows of 16-32 KB each). Harmless to the result (those rows are masked), but an
   MMU fault (Xid 31) whenever rel_logits sits at the edge of mapped memory.

B. SingleTileVarlenScheduler sizes the grid as an upper bound, so it can contain
   spare tiles. They come back as (block 0, head 0, batch == num_batch) with
   is_valid_tile False. SM90/SM100 loop `while work_tile.is_valid_tile`; SM80 never
   checked it and read cu_seqlens_q[num_batch + 1], seqused_k[num_batch] and
   page_table[num_batch] (one past the end), then worked with whatever it found.
"""
import sys

MARK = "oplc-fix-fa4-sm120-varlen-oob"

OLD_1 = """\
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, num_head, batch_size, split_idx = work_tile.tile_idx
"""
NEW_1 = OLD_1 + f"""\
        # [{MARK}] B: a spare grid tile carries batch index == num_batch. Point it at
        # a real request so the length lookups below stay in bounds; it is turned
        # into a no-op after get_n_block_min_max.
        if const_expr(mCuSeqlensQ is not None):
            batch_size = cutlass.min(batch_size, mCuSeqlensQ.shape[0] - 2)
        elif const_expr(mSeqUsedQ is not None):
            batch_size = cutlass.min(batch_size, mSeqUsedQ.shape[0] - 1)
"""

OLD_2 = """\
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(
            seqlen, m_block, split_idx, num_splits
        )
"""
NEW_2 = f"""\
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        # [{MARK}] A: recompute the aux-tensor wrap lengths for THIS request, as the
        # SM90/SM100 kernels do. The host-side values cover the whole batch.
        recompute_fastdiv_mods_q = const_expr(
            aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
        )
        recompute_fastdiv_mods_k = const_expr(
            aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
        )
        if const_expr(fastdiv_mods is not None):
            seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
            fastdiv_mods = (
                seqlen_q_divmod
                if not recompute_fastdiv_mods_q
                else FastDivmodDivisor(cutlass.max(seqlen.seqlen_q, 1)),
                seqlen_k_divmod
                if not recompute_fastdiv_mods_k
                else FastDivmodDivisor(cutlass.max(seqlen.seqlen_k, 1)),
            )
        n_block_min, n_block_max = block_info.get_n_block_min_max(
            seqlen, m_block, split_idx, num_splits
        )
        # [{MARK}] B: a spare grid tile does no work (empty K range -> the
        # `n_block_max > n_block_min` guard below skips load, compute and store).
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            n_block_max = n_block_max if work_tile.is_valid_tile else n_block_min
"""

WANT_MARKS = 3


def main():
    check = sys.argv[1] == "--check"
    path = sys.argv[2] if check else sys.argv[1]
    src = open(path).read()
    n = src.count(MARK)
    if n == WANT_MARKS:
        print(f"[fix-inkling-fa4-sm120-varlen-oob] {path} is already patched.")
        return 0
    if n != 0:
        print(f"[fix-inkling-fa4-sm120-varlen-oob ERROR] {path} is partially patched "
              f"({n} markers, want 0 or {WANT_MARKS}).", file=sys.stderr)
        return 1
    for name, old in (("tile index", OLD_1), ("seqlen/n_block", OLD_2)):
        if src.count(old) != 1:
            print(f"[fix-inkling-fa4-sm120-varlen-oob ERROR] expected exactly one '{name}' anchor in "
                  f"{path}, found {src.count(old)}. This is not the kernel source the fix was written for.",
                  file=sys.stderr)
            return 1
    if "recompute_fastdiv_mods" in src or "is_valid_tile" in src:
        print(f"[fix-inkling-fa4-sm120-varlen-oob ERROR] {path} already handles per-request fastdiv_mods "
              "or tile validity; refusing to patch an unknown variant.", file=sys.stderr)
        return 1
    if "from cutlass.cute import FastDivmodDivisor" not in src:
        print(f"[fix-inkling-fa4-sm120-varlen-oob ERROR] {path} does not import FastDivmodDivisor.",
              file=sys.stderr)
        return 1
    if check:
        print(f"[fix-inkling-fa4-sm120-varlen-oob] {path} is unpatched and compatible.")
        return 0
    src = src.replace(OLD_1, NEW_1).replace(OLD_2, NEW_2)
    assert src.count(MARK) == WANT_MARKS
    open(path, "w").write(src)
    print(f"[fix-inkling-fa4-sm120-varlen-oob] Patched {path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
