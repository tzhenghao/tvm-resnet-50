#!/bin/bash

set -euxo pipefail

echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal main\
    >> /etc/apt/sources.list.d/llvm.list

echo deb http://apt.llvm.org/focal/ llvm-toolchain-focal-13 main\
    >> /etc/apt/sources.list.d/llvm.list

apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 15CF4D18AF4F7421

apt-get update && apt-get install -y \
     llvm-11 llvm-12 llvm-13 \
     clang-11 libclang-11-dev \
     clang-12 libclang-12-dev \
     clang-13 libclang-13-dev