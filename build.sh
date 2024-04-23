# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=aptpackagemanager
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
rm bin/Rogasat
rm bin/BSAT-Release
rm bin/BSAT-Profiling
rm bin/BSAT-Sanitize
rm bin/BSAT-Debug
# Anything below g++-12 may have a bug with unsigned __int128 arithmetic in STL std::map
set -e
icpx -fsycl BSAT.cpp -DNDEBUG -O3 -funroll-loops -ffast-math -march=native \
  -std=c++20 -qopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -losqp -o "bin/Rogasat" &
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

icpx -fsycl BSAT.cpp -g -O3 -funroll-loops -ffast-math -march=native \
  -std=c++20 -qopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -losqp -o bin/BSAT-Release &
icpx -fsycl BSAT.cpp -gdwarf-4 -O2 -fno-inline -fno-omit-frame-pointer -ffast-math -march=native \
  -std=c++20 -qopenmp -Wl,--no-as-needed -ldl -ljemalloc -ltbb -losqp -o bin/BSAT-Profiling &
clang++-17 BSAT.cpp -ggdb3 -O2 -fsanitize=address,undefined -std=c++20 -march=native -fopenmp \
  -Wl,--no-as-needed -ldl -ljemalloc -ltbb -losqp -o bin/BSAT-Sanitize &
icpx -fsycl BSAT.cpp -ggdb3 -std=c++20 -march=native -qopenmp -Wl,--no-as-needed \
  -ldl -ljemalloc -ltbb -losqp -Wall -Wextra -o bin/BSAT-Debug &
wait