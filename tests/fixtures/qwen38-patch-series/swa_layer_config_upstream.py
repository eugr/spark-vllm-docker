# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Excerpts from b409a4f0d787fdddb75ec2e8ddb6dfcfd36235b7:
# vllm/model_executor/layers/attention/attention.py
from __future__ import annotations

def _largest_kernel_block_within(
    attn_backend: "type[AttentionBackend]",
    per_token_bytes: int,
    page_budget: int,
    fallback: int,
    kv_cache_spec: KVCacheSpec | None = None,
    divisor_of: int | None = None,
    layer_name: str | None = None,
) -> int:
    """Largest supported kernel block size whose page fits in `page_budget`.

    A padded spec (e.g. skip-quant layer) that pads its page up to a large shared page
    wastes `page_budget - block*per_token` bytes per block. Picking the largest kernel
    block whose natural page still fits under `page_budget` minimizes that waste.
    `MultipleOf` declarations are expanded to the largest aligned block that fits.
    Falls back to the smallest supported block when nothing fits.

    With `divisor_of`, prefer the largest fitting block that divides it, so the
    page can be scaled to `divisor_of` by an integer ratio instead of padded.
    """
    from vllm.v1.attention.backend import MultipleOf

    sizes = (
        attn_backend.get_supported_kernel_block_sizes(kv_cache_spec)
        if layer_name is None
        else attn_backend.get_supported_kernel_block_sizes_for_layers(
            kv_cache_spec, [layer_name]
        )
    )
    max_block_size = page_budget // per_token_bytes
    candidates = [s for s in sizes if isinstance(s, int)]
    candidates.extend(
        max(s.base, max_block_size // s.base * s.base)
        for s in sizes
        if isinstance(s, MultipleOf)
    )
    if not candidates:
        return fallback
    smallest = min(candidates)
    fitting = [b for b in candidates if b * per_token_bytes <= page_budget]
    if divisor_of is not None:
        dividing = [b for b in fitting if divisor_of % b == 0]
        dividing.extend(
            b
            for s in sizes
            if isinstance(s, MultipleOf)
            for b in range(s.base, max_block_size + 1, s.base)
            if divisor_of % b == 0
        )
        if dividing:
            return max(dividing)
    return max(fitting) if fitting else smallest

class Attention:
    def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig | None:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Encoder-only attention is prefill-only and keeps no autoregressive KV
        # cache. In hybrid models (e.g. Qwen3.5 / ColQwen3.5: GatedDeltaNet
        # linear_attention interleaved with full_attention) the runner iterates
        # every attention module to build the KV-cache spec, so an ENCODER_ONLY
        # full_attention layer reaches here; it contributes no KV cache group.
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return None
        # Should not be called for enc-dec attention.
        assert self.attn_type == AttentionType.DECODER
        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
        if self.sliding_window is not None:
            assert not self.attn_backend.is_mla(), (
                "MLA is not supported for sliding window"
            )
            # SW chooses its own block_size, decoupled from the user's
            # ``--block-size`` (which only constrains primary attention).
            # When this SW layer is a padded spec (skip-quant: its page is
            # padded up to ``skip_page_size_padded``), pick the largest kernel
            # block that still fits the shared page so we waste fewer padding
            # bytes per block. Otherwise (page_size_padded is None) take the
            # primary block size when the backend can run it unsplit: if this
            # page does not divide the primary page, ``unify`` then pads it
            # instead of scaling a small block
            # to a size coprime with the primary one, which inflates the
            # scheduler LCM (e.g. a 1024 B/token SWA draft next to a 1152
            # B/token MLA target: 1728 vs 1536 gives LCM 13824). Backends that
            # cannot run the primary block start from their smallest block and
            # ``unify`` scales it up by an integer ratio.
            shared_page = vllm_config.cache_config.skip_page_size_padded
            # The backend owns its packing
            kv_cache_spec = self.attn_backend.customize_spec(
                AttentionSpec(
                    block_size=1,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_size,
                    head_size_v=self.head_size_v,
                    dtype=self.kv_cache_torch_dtype,
                    kv_quant_mode=quant_mode,
                )
            )
            sw_per_token = kv_cache_spec.real_page_size_bytes
            page_budget = shared_page or sw_per_token * block_size
            # Without a shared page, prefer a block that divides the primary
            # block: ``unify`` pads any other block's page, and a padded page
            # disables fine-grained prefix-cache hits on this group.
            sw_block_size = _largest_kernel_block_within(
                self.attn_backend,
                sw_per_token,
                page_budget,
                block_size,
                kv_cache_spec,
                divisor_of=None if shared_page else block_size,
                layer_name=self.layer_name,
            )
            spec = AttentionSpec(
                block_size=sw_block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
                page_size_padded=shared_page,
            )
            return SlidingWindowManager.layer_config(
                spec, sliding_window=self.sliding_window
            )
        else:
            spec = AttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
            )
            return FullAttentionManager.layer_config(spec)
