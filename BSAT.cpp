#include "Circulation.h"
#include "MaxFlow.h"
#include "Reduction.h"

#include <iostream>

int main(int argc, char* argv[]) {
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  return 0;
}
