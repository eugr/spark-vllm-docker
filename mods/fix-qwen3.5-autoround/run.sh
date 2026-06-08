#!/bin/bash
set -e
patch -p1 --forward -d /usr/local/lib/python3.12/dist-packages < transformers.patch || {
  echo "Patch already applied or not needed for this transformers version — skipping"
}