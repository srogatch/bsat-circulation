set -x
# sudo apt-get install libjemalloc-dev libtbb-dev libomp-17-dev
mkdir -p bin
# Anything below g++-12 may have a bug with unsigned __int128 arithmetic in STL std::map
set -e
clang++-17 BSAT.cpp -DNDEBUG -O3 -funroll-loops -ffast-math -march=native \
  -std=c++20 -fopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -o bin/Rogasat
clang++-17 BSAT.cpp -g -O3 -funroll-loops -ffast-math -march=native \
  -std=c++20 -fopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -o bin/BSAT-Release
clang++-17 BSAT.cpp -gdwarf-4 -O2 -fno-inline -fno-omit-frame-pointer -ffast-math -march=native \
  -std=c++20 -fopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -o bin/BSAT-Profiling
clang++-17 BSAT.cpp -ggdb3 -std=c++20 -march=native -fopenmp -Wl,--no-as-needed \
  -ldl -ljemalloc -ltbb -Wall -Wextra -o bin/BSAT-Debug
