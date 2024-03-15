mkdir -p bin
g++-11 -g BSAT.cpp -std=c++20 -march=native -o bin/BSAT-Debug
g++-11 -O3 BSAT.cpp -std=c++20 -march=native -o bin/BSAT-Release
