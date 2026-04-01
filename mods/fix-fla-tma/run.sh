#!/bin/bash
set -e

# Patch vllm/model_executor/layers/fla/ops/utils.py to respect FLA_USE_TMA env var
UTILS_FILE=$(python3 -c "import vllm.model_executor.layers.fla.ops.utils as m; print(m.__file__)")
echo "[fix-fla-tma] Patching $UTILS_FILE"

python3 -c "
import re

with open('$UTILS_FILE') as f:
    content = f.read()

# Find the is_tma_supported line and add env var check
if 'FLA_USE_TMA' not in content:
    # Add os import if not present
    if 'import os' not in content:
        content = 'import os\n' + content
    
    # Replace the is_tma_supported assignment to check env var
    content = re.sub(
        r'(is_tma_supported\s*=\s*\(.*?\))',
        r'\1 and os.environ.get(\"FLA_USE_TMA\", \"1\") != \"0\"',
        content,
        flags=re.DOTALL
    )
    
    with open('$UTILS_FILE', 'w') as f:
        f.write(content)
    print('[fix-fla-tma] Patched successfully')
else:
    print('[fix-fla-tma] Already patched')
"

# Verify
python3 -c "
import os
os.environ['FLA_USE_TMA'] = '0'
# Force reimport
import importlib
import vllm.model_executor.layers.fla.ops.utils as u
importlib.reload(u)
print(f'[fix-fla-tma] is_tma_supported after patch: {u.is_tma_supported}')
"
