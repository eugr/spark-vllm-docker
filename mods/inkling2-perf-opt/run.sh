#!/usr/bin/env bash
set -euo pipefail
mod_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$mod_dir/patch_inkling2_opt.py"
