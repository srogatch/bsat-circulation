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
  for(;;) {
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
    // Assign one unknown variable arbitrarily
    int64_t nUnknown = 0;
    int64_t iToAssign = 0;
    for(int64_t i=1; i<=task.nVars_; i++) {
      if(!task.known_[i]) {
        nUnknown++;
        if(nUnknown == 1) {
          iToAssign = i;
        }
      }
    }
    if(nAssigned == 0 && iToAssign != 0) {
      std::cout << "\t" << nUnknown << " variables are still unknown. Assigning var #"
        << iToAssign << " arbitrarily." << std::endl;
      task.known_[iToAssign] = true;
      task.ans_[iToAssign] = false;
      nUnknown--;
    }
    satisfiable &= task.RemoveKnown();
    if(!satisfiable) {
      std::cout << "After removing known variable values, the formula is not satisfiable." << std::endl;
      break;
    }
    if(nUnknown == 0) {
      break;
    }
  }
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
