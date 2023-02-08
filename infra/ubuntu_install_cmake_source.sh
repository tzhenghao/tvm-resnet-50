#!/bin/bash

set -e
set -u
set -o pipefail

if [ -z ${1+x} ]; then
    version=3.18.4
else
    version=$1
fi

v=$(echo $version | sed 's/\(.*\)\..*/\1/g')
echo "Installing cmake $version ($v)"
wget https://cmake.org/files/v${v}/cmake-${version}.tar.gz
tar xvf cmake-${version}.tar.gz
cd cmake-${version}
./bootstrap
make -j$(nproc)
make install
cd ..
rm -rf cmake-${version} cmake-${version}.tar.gz