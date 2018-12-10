#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <string>
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::unordered_set;
using std::vector;

void writeTrueTable4(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 4;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable5(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 5;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable6(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 6;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable7(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 7;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable8(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 8;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable9(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 9;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable10(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 10;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable11(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 11;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable12(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 12;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable13(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 13;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable14(string fname) {
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

void writeTrueTable15(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 15;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable16(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 16;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable17(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 17;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable18(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 18;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable19(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 19;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable20(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 20;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable21(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 21;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable22(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 22;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable23(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 23;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable24(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 24;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

void writeTrueTable25(string fname) {
  ofstream ofs(fname, std::iostream::out);
  string line;
  const int num_bits = 25;
  int total = 1 << num_bits;

  ofs << ".i " << num_bits << endl;
  ofs << ".o " << 1 << endl;
  for (int i = 0; i < total; ++i) {
    ofs << std::bitset<num_bits>(i).to_string() << " "
        << (std::rand() % 3 == 0 ? 1 : 0) << endl;
  }
  ofs << ".e" << endl;
}

int main(int argc, char** argv) {
		writeTrueTable4("input.pla4");
		writeTrueTable5("input.pla5");
		writeTrueTable6("input.pla6");
		writeTrueTable7("input.pla7");
		writeTrueTable8("input.pla8");
		writeTrueTable9("input.pla9");
		writeTrueTable10("input.pla10");
		writeTrueTable11("input.pla11");
		writeTrueTable12("input.pla12");
		writeTrueTable13("input.pla13");
		writeTrueTable14("input.pla14");
		writeTrueTable15("input.pla15");
		writeTrueTable16("input.pla16");
		writeTrueTable17("input.pla17");
		writeTrueTable18("input.pla18");
		writeTrueTable19("input.pla19");
		writeTrueTable20("input.pla20");
		writeTrueTable21("input.pla21");
		writeTrueTable22("input.pla22");
		writeTrueTable23("input.pla23");
		writeTrueTable24("input.pla24");
		writeTrueTable25("input.pla25");
}
