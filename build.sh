# sudo apt-get install libjemalloc-dev
mkdir -p bin
g++-11 -g BSAT.cpp -std=c++20 -march=native -fopenmp -ljemalloc -o bin/BSAT-Debug
g++-11 -ggdb3 -O3 BSAT.cpp -std=c++20 -march=native  -fopenmp -ljemalloc -o bin/BSAT-Release
