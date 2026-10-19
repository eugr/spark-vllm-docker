# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Loader tail after 14-pr60646.patch on upstream
# fe341a53dda00472c281a2d1f4bd34a74e7eae6e.

class DefaultModelLoader:
    def weights(self, source, weights_iterator):
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return (
            (source.prefix + name, tensor)
            for (name, tensor) in decode_csf_scale_streams(weights_iterator)
        )

    def get_all_weights(
        self,
    ):
        pass
