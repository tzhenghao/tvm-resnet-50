#!/bin/bash

set -euxo pipefail

# clone the repo
cd /usr
git clone --recursive https://github.com/apache/tvm tvm
cd /usr/tvm

# set flags
echo set\(USE_LLVM llvm-config-13\) >> cmake/config.cmake
echo set\(USE_CUDA ON\) >> cmake/config.cmake
echo set\(USE_CUDNN ON\) >> cmake/config.cmake
echo set\(USE_BLAS openblas\) >> cmake/config.cmake

# set debugging flags
echo set\(USE_GRAPH_EXECUTOR ON\) >> cmake/config.cmake
echo set\(USE_PROFILER ON\) >> cmake/config.cmake

cd /usr/tvm
mkdir -p build
cd build
cmake ..
make -j8