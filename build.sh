mkdir -p bin
g++ -g BSAT.cpp -std=c++20 -o bin/BSAT-Debug
g++ -O3 BSAT.cpp -std=c++20 -o bin/BSAT-Release
