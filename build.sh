# sudo apt-get install libjemalloc-dev libtbb-dev
mkdir -p bin
g++-11 -g BSAT.cpp -std=c++20 -march=native -fopenmp -ltbb -ljemalloc -o bin/BSAT-Debug
g++-11 -ggdb3 -O3 -funroll-loops -ffast-math \
  BSAT.cpp -std=c++20 -march=native  -fopenmp -ltbb -ljemalloc -o bin/BSAT-Release
