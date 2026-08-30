"""GB10 (sm_120/121) cannot launch persistent_topk: the persistent launch
needs total_ctas <= num_sms*occupancy (48 on GB10; glm5_next geometry wants
77-90) and the FilteredTopK fallback needs 128KB smem/block (GB10: 101KB).
The selector already excludes capability family 120 for cooperative_topk but
not for persistent_topk; add the same exclusion so sm120 routes to the
generic top_k_per_row_decode kernel that already sits in the else branch.
Without this, any request >32K tokens (sparse-indexer activation) kills the
engine: 'persistent_topk would oversubscribe and the FilteredTopK fallback
requires >=128KB smem per block (have 101376)'."""
import py_compile

P1 = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py"
src = open(P1).read()
a1 = """        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
"""
r1 = """        use_persistent_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and not current_platform.is_device_capability_family(120)
        )
"""
if "persistent_topk\n" in src and "is_device_capability_family(120)\n        )\n        if use_cooperative_topk" in src:
    pass
if a1 in src:
    assert src.count(a1) == 1, f"indexer anchor count {src.count(a1)}"
    src = src.replace(a1, r1)
    open(P1, "w").write(src)
    py_compile.compile(P1, doraise=True)
    print("sparse_attn_indexer sm120 persistent_topk exclusion applied + compiles")
else:
    assert "and not current_platform.is_device_capability_family(120)\n        )" in src
    print("sparse_attn_indexer already patched")

P2 = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
src2 = open(P2).read()
a2 = "        if current_platform.is_cuda() and select_k in (512, 1024, 2048):\n"
r2 = ("        if (current_platform.is_cuda() and select_k in (512, 1024, 2048)\n"
      "                and not current_platform.is_device_capability_family(120)):\n")
if a2 in src2:
    assert src2.count(a2) == 1, f"kpool anchor count {src2.count(a2)}"
    src2 = src2.replace(a2, r2)
    open(P2, "w").write(src2)
    py_compile.compile(P2, doraise=True)
    print("sparse_attn_indexer_kpool sm120 persistent_topk exclusion applied + compiles")
else:
    assert "and not current_platform.is_device_capability_family(120)):" in src2
    print("sparse_attn_indexer_kpool already patched")
