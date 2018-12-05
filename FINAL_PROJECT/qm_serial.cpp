#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>

using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::unordered_set;
using std::vector;

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

  getline(s, line, ' ');
  getline(s, line);
  split(line, " ", in_labels);

  getline(s, line, ' ');
  getline(s, line);
  split(line, " ", out_labels);

  // std::copy(in_labels.begin(), in_labels.end(),
  // std::ostream_iterator<string>(std::cout, " "));
  // std::copy(out_labels.begin(), out_labels.end(),
  // std::ostream_iterator<string>(std::cout, " "));

  // read head
  while (getline(s, line) && (line != ".e")) {
    vector<string> buff;
    split(line, " ", buff);
    input.push_back(buff[0]);
    output.push_back(buff[1]);
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
  vector<string> v;
  vector<string> prime;   // vector<char*> prime;
  vector<string> result;  // vector<char*> result;

  prepInput(v);
  vector<string> relative(v);
  vector<vector<int>> buckets(17);

  cout << "Input " << endl; 
  std::copy(v.begin(), v.end(), std::ostream_iterator<string>(cout, "\n")); 

  for (int j = 0; j < v.size(); j++)
    buckets[std::count(v[j].begin(), v[j].end(), '1')].push_back(j);

  // to be parallelet
  for (int i = 0; i < 16; i++) {
    auto it = std::find_if(buckets.begin(), buckets.end(),
                           [](const vector<int>& a) { return a.size(); });
    if (it == buckets.end()) break;

    vector<vector<int>> next(17);
    vector<bool> flag(v.size());

    // update bucket
    for (int j = 0; j < 16; ++j) {
      for (auto a : buckets[j]) {
        for (auto b : buckets[j + 1]) {
          int res = checkBITs(16, v[a], v[b]);
          if (res != -1) {  // can merge
            flag[a] = 1, flag[b] = 1;
            v[a][res] = '2';
            next[j].push_back(a);
          }
        }
        if (flag[a] == 0) prime.push_back(v[a]);
      }
    }
    buckets = std::move(next);
  }

  for (int i = 0; i < relative.size(); i++) {
    if (relative[i].empty()) continue;

    int count = 0, num = 0;
    for (int j = 0; j < prime.size(); j++) {
      if (prime.size() && comp(16, relative[i], prime[j])) {
        if (++count > 1) break;
        num = j;
      }
    }

    if (count == 1) {  // essential prime implicant
      result.push_back(prime[num]);
      for (int j = 0; j < relative.size(); j++) {
        if (relative[j].size() && comp(16, relative[j], prime[num])) {
          relative[j] = "";
        }
      }
      prime[num] = "";
    }
  }

  int cnt_empty = std::count_if(relative.begin(), relative.end(),
                                [](string a) { return a.size() == 0; });

  while (cnt_empty < relative.size()) {
    do {
      temp = prime.back();
      prime.pop_back();
    } while (temp.size() == 0 && prime.size());

    count = 0;
    for (int i = 0; i < relative.size(); i++) {
      if (relative[i].size() && comp(16, relative[i], temp)) {
        relative[i] = "";
        cnt_empty++;
        count++;
      }
    }
    if (count > 0) {
      result.push_back(temp);
    }
  }

  cout << "result : " << endl;
  for (auto item : result) cout << item << endl;
  return 0;
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