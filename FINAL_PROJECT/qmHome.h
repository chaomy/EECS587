#ifndef _QM_CUDA_

#include <bitset>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <iterator> 
#include <algorithm>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::ifstream;
using std::iostream;
using std::string;
using std::vector;

class QMParallel {
  vector<string> input, output; 
  vector<string> in_labels, out_labels;

 public:
  // read input true table 
  void readTrueTable(string fname);

  // write output true table 
  void writeTrueTable(string fname); 

  // run QMP parallel  
  void runQM(); 
};

inline void split(const string& s, const char* delim, vector<string>& v) {
  // duplicate original string, return a char pointer and free  memories
  char* dup = strdup(s.c_str());
  char* token = strtok(dup, delim);
  while (token != NULL) {
    v.push_back(string(token));
    // the call is treated as a subsequent calls to strtok:
    // the function continues from where it left in previous invocation
    token = strtok(NULL, delim);
  }
  free(dup);
}

#endif  //