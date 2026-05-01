#!/bin/bash

set -e

patch -p1 -d /usr/local/lib/python3.12/dist-packages < gpu_mem.patch \
  && echo "=====> You can now use --gpu-memory-utilization-gb parameter to specify reserved memory in GiB"