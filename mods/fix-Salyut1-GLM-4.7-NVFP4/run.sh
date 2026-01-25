#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PATCH_FILE="$DIR/glm4_moe.patch"

apply() {
    echo "Applying patch..."
    patch -p1 -d / < "$PATCH_FILE"
}

description() {
    echo "Fixes Salyut1/GLM-4.7-NVFP4 incompatibility by patching glm4_moe parser"
}

if [ "$1" == "description" ]; then
    description
else
    apply
fi