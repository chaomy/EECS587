#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <unordered_set>
#include <vector>
#include <ctime> 

using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::unordered_set;
using std::vector;

// inline void split(const string& s, const char* delim, vector<string>& v) {
//   // duplicate original string, return a char pointer and free  memories
//   char* dup = strdup(s.c_str());
//   char* token = strtok(dup, delim);
//   while (token != NULL) {
//     v.push_back(string(token));
//     // the call is treated as a subsequent calls to strtok:
//     // the function continues from where it left in previous invocation
//     token = strtok(NULL, delim);
//   }
//   free(dup);
// }

int in_bit_num, out_bit_num;
vector<string> in_labels, out_labels;
vector<string> input, output;

bool comp(int n, string a, string b) {
  for (int i = 0; i < n; i++) {
    if (a[i] != b[i] && (a[i] != '2' && b[i] != '2')) return false;
  }
  return true;
}

int checkBITs(int n, string a, string b) {
  int count = 0, temp;
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      if (++count > 1) return -1;
      temp = i;
    }
  }
  return count == 1 ? temp : -1;
}

void readTrueTable(string fname) {
  ifstream s(fname, std::iostream::in);
  string line;

  getline(s, line, ' ');
  getline(s, line);
  in_bit_num = stoi(line);

  getline(s, line, ' ');
  getline(s, line);
  out_bit_num = stoi(line);

  // std::copy(in_labels.begin(), in_labels.end(),
  // std::ostream_iterator<string>(std::cout, " "));
  // std::copy(out_labels.begin(), out_labels.end(),
  // std::ostream_iterator<string>(std::cout, " "));

  // read head
  string buff1, buff2;
  while (getline(s, buff1, ' ') && getline(s, buff2) && (buff1 != ".e")) {
    input.push_back(buff1);
    output.push_back(buff2);
  }
}

void prepInput(vector<string>& v) {
  size_t N{input.size()};
  v.reserve(N);
  for (int i = 0; i < N; ++i) {
    if (output[i][0] == '1' || output[i][0] == '2') {
      v.push_back(input[i]);
    }
  }
}

int main() {
  readTrueTable("input.pla");

  int count;
  string temp;
  vector<string> v;             // vector of strings that correponds to 1
  unordered_set<string> prime;  //
  vector<string> result;        // vector<char*> result;

  prepInput(v);
  vector<string> relative(v);

  // store according to num of 1 bits
  vector<vector<string>> buckets(17);

  cout << "Input " << endl;
  std::copy(v.begin(), v.end(), std::ostream_iterator<string>(cout, "\n"));

  for (int j = 0; j < v.size(); j++)
    buckets[std::count(v[j].begin(), v[j].end(), '1')].push_back(v[j]);

  std::clock_t start = std::clock();

  // to be parallelet
  for (int i = 0; i < 16; i++) {
    auto it = std::find_if(buckets.begin(), buckets.end(),
                           [](const vector<string>& a) { return a.size(); });
    if (it == buckets.end()) break;

    vector<vector<string>> next(17);
    unordered_set<string> flag;

    // update bucket
    for (int j = 0; j < 16; ++j) {
      for (auto str_a : buckets[j]) {
        for (auto str_b : buckets[j + 1]) {
          int res = checkBITs(16, str_a, str_b);
          if (res != -1) {  // can merge
            flag.insert(str_a);
            flag.insert(str_b);
            str_a[res] = '2';
            next[j].push_back(str_a);
            str_a[res] = '0';
          }
        }
        if (flag.find(str_a) == flag.end()) prime.insert(str_a);
      }
    }
    buckets = std::move(next);
  }

  double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
  for (auto num : prime) cout << num << endl;
  cout << "time: " << duration << endl;
  cout << "prime " << endl;
  return 0;

  // for (int i = 0; i < relative.size(); i++) {
  //   if (relative[i].empty()) continue;

  //   int count = 0, num = 0;
  //   for (int j = 0; j < prime.size(); j++) {
  //     if (prime.size() && comp(16, relative[i], prime[j])) {
  //       if (++count > 1) break;
  //       num = j;
  //     }
  //   }

  //   if (count == 1) {  // essential prime implicant
  //     result.push_back(prime[num]);
  //     for (int j = 0; j < relative.size(); j++) {
  //       if (relative[j].size() && comp(16, relative[j], prime[num])) {
  //         relative[j] = "";
  //       }
  //     }
  //     prime[num] = "";
  //   }
  // }

  // int cnt_empty = std::count_if(relative.begin(), relative.end(),
  //                               [](string a) { return a.size() == 0; });

  // while (cnt_empty < relative.size()) {
  //   do {
  //     temp = prime.back();
  //     prime.pop_back();
  //   } while (temp.size() == 0 && prime.size());

  //   count = 0;
  //   for (int i = 0; i < relative.size(); i++) {
  //     if (relative[i].size() && comp(16, relative[i], temp)) {
  //       relative[i] = "";
  //       cnt_empty++;
  //       count++;
  //     }
  //   }
  //   if (count > 0) {
  //     result.push_back(temp);
  //   }
  // }

  // cout << "result : " << endl;
  // for (auto item : result) cout << item << endl;
  // return 0;
}

// version 1
// bool* flag = new bool[v.size()];  // be able to improve
// for (int k = j + 1; k < v.size(); k++) {
//   int impt = imp(n, v[j], v[k]);
//   if (impt != -1) {
//     flag[j] = 1;
//     flag[k] = 1;
//     strcpy(temp, v[j]);
//     temp[impt] = '2';
//     if (find(v[i + 1].begin(), v[i + 1].end(), temp) == v[i + 1].end())
//       v[i + 1].push_back(temp);
//   }
// }

// for (int j = 0; j < v.size(); j++) {
//   if (!flag[j]) {
//     prime.push_back(v[j]);
//   }
// }
