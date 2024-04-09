set -x
found_seq=false
for arg in "$@"; do
    if [[ "$arg" == "seq" ]]; then
        found_seq=true
        break
    fi
done

# sudo apt-get install libjemalloc-dev libtbb-dev libomp-17-dev
mkdir -p bin
# Anything below g++-12 may have a bug with unsigned __int128 arithmetic in STL std::map
set -e
clang++-17 BSAT.cpp -DNDEBUG -O3 -funroll-loops -ffast-math -march=native \
  -std=c++20 -fopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -o bin/rogasat &
# Save the PID of the background process
clang_pid=$!
if $found_seq; then  
  # Use 'wait' to wait for the command to complete
  wait $clang_pid
  clang_exit_status=$?
  # Check the exit status
  if [ $clang_exit_status -ne 0 ]; then
      echo "clang failed with exit status $clang_exit_status"
      exit $clang_exit_status
  fi
fi

clang++-17 BSAT.cpp -g -O3 -funroll-loops -ffast-math -march=native \
  -std=c++20 -fopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -o bin/BSAT-Release &
clang++-17 BSAT.cpp -gdwarf-4 -O2 -fno-inline -fno-omit-frame-pointer -ffast-math -march=native \
  -std=c++20 -fopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -o bin/BSAT-Profiling &
clang++-17 BSAT.cpp -ggdb3 -O2 -fsanitize=address,undefined -std=c++20 -march=native -fopenmp \
  -Wl,--no-as-needed -ldl -ljemalloc -ltbb -o bin/BSAT-Sanitize &
clang++-17 BSAT.cpp -ggdb3 -std=c++20 -march=native -fopenmp -Wl,--no-as-needed \
  -ldl -ljemalloc -ltbb -Wall -Wextra -o bin/BSAT-Debug &
wait