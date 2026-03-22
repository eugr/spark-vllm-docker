#!/bin/bash
set -e
echo "--- Applying mistral tokenizers patch..."
patch -p1 -d /usr/local/lib/python3.12/dist-packages < mistral_tokenizers.patch
echo "=== OK"
echo "--- Applying mistral_common patch for ccr..."
patch -p1 -d /usr/local/lib/python3.12/dist-packages < mistral_common_ccr.patch
echo "=== OK"
