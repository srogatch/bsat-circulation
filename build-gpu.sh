set -x
mkdir build
cd build

set -e
export CC=clang-17
export CXX=clang++-17
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 32
