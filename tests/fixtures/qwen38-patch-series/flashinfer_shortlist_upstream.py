# SPDX-License-Identifier: Apache-2.0
# Excerpt from FlashInfer 9bd0a68c, inside rank_tactics.
def lookup(self, runner, inputs, k, tuning_config, tactic_key, cache_key):
    if runner is not None:
        if k > 0:
            shortlist_identity = (self._profiling_policy(tuning_config), tactic_key)
            ranked_cache = self._ranked_cache()
            cached = ranked_cache.get(cache_key)
            if _all_ranks_hit_cache(
                cached is not None and cached[0] == shortlist_identity,
                (type(runner).__name__, k, cached[1][:k]) if cached else None,
            ):
                assert cached is not None
                return list(cached[1][:k])
