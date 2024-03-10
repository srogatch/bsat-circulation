#include "Reduction.h"

#include <iostream>

int main(int argc, char* argv[]) {
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  Formula task;
  task.Load(argv[1]);
  Reduction red(task);
  bool maybeSat = red.Circulate();
  std::ofstream ofs(argv[2]);
  if(maybeSat) {
    std::vector<bool> varVals = red.AssignVars();
    if(!varVals[0]) {
      ofs << "c UNCERTAIN" << std::endl;
    }
    if(varVals.size() > 1) {
      ofs << "s SATISFIABLE" << std::endl;
      ofs << "v ";
      for(int64_t i=1; i<varVals.size(); i++) {
        ofs << (varVals[i] ? i : -i) << " ";
      }
      ofs << "0" << std::endl;
      return 0;
    }
  }
  ofs << "s UNSATISFIABLE" << std::endl;
  return 0;
}
