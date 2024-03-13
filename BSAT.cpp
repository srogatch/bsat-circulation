#include "Reduction.h"

#include <iostream>

int main(int argc, char* argv[]) {
  if(argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.dimacs> <output.dimacs>" << std::endl;
    return 1;
  }
  Formula task;
  task.Load(argv[1]);
  bool satisfiable = true;
  do {
    Reduction red(task);
    satisfiable &= red.Circulate();
    if(!satisfiable) {
      std::cout << "No circulation." << std::endl;
      break;
    }
    int64_t nAssigned = 0;
    satisfiable &= red.AssignVars(nAssigned);
    if(!satisfiable) {
      std::cout << "Can't assign variable values." << std::endl;
      break;
    }
    std::cout << "Reflow succeeded." << std::endl;
  }
  while(false);

  std::ofstream ofs(argv[2]);
  if(!satisfiable) {
    ofs << "s UNSATISFIABLE" << std::endl;
    return 0;
  }

  assert(task.SolWorks());
  ofs << "s SATISFIABLE" << std::endl;
  ofs << "v ";
  for(int64_t i=1; i<task.ans_.size(); i++) {
    ofs << (task.ans_[i] ? i : -i) << " ";
  }
  ofs << "0" << std::endl;
  return 0;
}
