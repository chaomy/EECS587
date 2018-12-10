#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::unordered_set;
using std::vector;

void writeTrueTable(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 14;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

int main(int argc, char** argv) { writeTrueTable("input.pla"); }
