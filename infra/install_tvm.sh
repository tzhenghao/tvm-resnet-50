#!/bin/bash

set -euxo pipefail

# clone the repo
cd /usr
git clone --recursive https://github.com/apache/tvm tvm
cd /usr/tvm

# copy our own config.cmake to the root tvm directory.
cp /infra/config.cmake .

cd /usr/tvm
mkdir -p build
cd build
cmake .. -G Ninja
ninja