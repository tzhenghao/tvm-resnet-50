set -e
set -u
set -o pipefail

cd /usr
git clone https://github.com/apache/tvm tvm --recursive
cd /usr/tvm
# checkout a hash-tag
git checkout 4b13bf668edc7099b38d463e5db94ebc96c80470

echo set\(USE_LLVM llvm-config-8\) >> config.cmake
echo set\(USE_CUDA ON\) >> config.cmake
echo set\(USE_CUDNN ON\) >> config.cmake
echo set\(USE_BLAS openblas\) >> config.cmake
mkdir -p build
cd build
cmake ..
make -j10