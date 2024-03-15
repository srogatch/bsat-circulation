mkdir -p bin
g++-11 -g BSAT.cpp -std=c++20 -march=native -fopenmp -o bin/BSAT-Debug
g++-11 -O3 BSAT.cpp -std=c++20 -march=native  -fopenmp -o bin/BSAT-Release
