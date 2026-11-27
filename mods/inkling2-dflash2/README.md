# Inkling 2 Mini DFlash2 adapter

The adapter applies to the vLLM image pinned by the
[Inkling 2 Mini DFlash2 recipe](../../recipes/inkling-2-mini-dflash2.yaml).
It uses the draft checkpoint's three Inkling loop visits and the final
normalized state. The draft shares the target's input embedding table and
normalization, limits proposals to the target's usable vocabulary, and applies
the logit soft cap after the selector's transition correction.

The capture and scoring behavior follows
[Inco AI's Inkling integration](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/blob/4f9088ea66c21f2b99ed657d1941c306b03e84d2/sglang_inkling_dflash2.py).
