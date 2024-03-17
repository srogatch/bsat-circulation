# sudo apt-get install libjemalloc-dev libtbb-dev
mkdir -p bin
# Anything below g++-12 may have a bug with unsigned __int128 arithmetic in STL std::map
g++-12 BSAT.cpp -g -std=c++20 -march=native -fopenmp -ljemalloc -ltbb -o bin/BSAT-Debug
g++-12 BSAT.cpp -ggdb3 -O3 -DNDEBUG -funroll-loops -ffast-math -march=native \
  -std=c++20 -fopenmp -ljemalloc -ltbb -o bin/BSAT-Release
