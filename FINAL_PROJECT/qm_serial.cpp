#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <unordered_set>
#include <vector>

using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::unordered_set;
using std::vector;

// g++ qm_serial.cpp -std=c++11 -o qm_serial -o3 -g

double time1, time2;
int in_bit_num, out_bit_num;
vector<string> in_labels, out_labels;
vector<string> input, output;

bool comp(int n, const string& a, const string& b) {
  if (a.empty() || b.empty()) return false;
  for (int i = 0; i < n; i++) {
    if (a[i] != b[i] && (a[i] != '2' && b[i] != '2')) return false;
  }
  return true;
}

int checkbits(int n, const string& a, const string& b) {
  if (a.empty() || b.empty()) return -1;
  int count = 0, temp;
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      if (++count > 1) return -1;
      temp = i;
    }
  }
  return count == 1 ? temp : -1;
}

void readtruetable(string fname) {
  ifstream s(fname, std::iostream::in);
  string line;

  getline(s, line, ' ');
  getline(s, line);
  in_bit_num = stoi(line);

  getline(s, line, ' ');
  getline(s, line);
  out_bit_num = stoi(line);

  // read head
  string buff1, buff2;
  while (getline(s, buff1, ' ') && getline(s, buff2)) {
    input.push_back(buff1);
    output.push_back(buff2);
  }
}

void prepinput(vector<string>& v) {
  size_t n{input.size()};
  v.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (output[i][0] == '1' || output[i][0] == '2') {
      v.push_back(input[i]);
    }
  }
}

bool notempty(const string& a) { return a != "0"; }

struct compareprime {
  bool operator()(const string& a, const string& b) {
    if (a.empty() || b.empty()) return a < b;
    size_t score_a = std::count(a.begin(), a.end(), '2');
    size_t score_b = std::count(b.begin(), b.end(), '2');
    return score_a == score_b ? a < b : score_a < score_b;
  }
};

// qm step 1
void find_primes_serial(vector<string>& v, vector<string>& vec_primes,
                        int bit_num) {
  vector<vector<string>> buckets(bit_num + 1, vector<string>());

  // store according to num of 1 bits
  unordered_set<string> prime;

  for (auto key : v)
    buckets[std::count(key.begin(), key.end(), '1')].push_back(key);

  std::clock_t start = std::clock();

  // to be parallelet
  for (int i = 0; i < bit_num; i++) {
    auto it =
        std::find_if(buckets.begin(), buckets.end(),
                     [](const vector<string>& a) { return a.size() > 0; });
    if (it == buckets.end()) break;

    vector<vector<string>> next(bit_num + 1, vector<string>());
    vector<unordered_set<string>> vec_flags(bit_num + 1,
                                            unordered_set<string>());

    // update bucket
    for (int j = 0; j < bit_num; ++j) {
      for (auto str_a : buckets[j]) {
        for (auto str_b : buckets[j + 1]) {
          int res = checkbits(in_bit_num, str_a, str_b);
          if (res != -1) {  // can merge
            vec_flags[j].insert(str_a);
            vec_flags[j + 1].insert(str_b);
            str_a[res] = '2';
            next[j].push_back(str_a);
            str_a[res] = '0';
          }
        }
        if (vec_flags[j].find(str_a) == vec_flags[j].end()) prime.insert(str_a);
      }  // loop over all items on buket layer i
      vec_flags[j].clear();
    }  // loop over all layers
    buckets = std::move(next);
  }
  std::copy(prime.begin(), prime.end(), std::back_inserter(vec_primes));
  time1 = (std::clock() - start) / double(CLOCKS_PER_SEC);
  // std::sort(vec_primes.begin(), vec_primes.end(), compareprime());
}

// solve set cover problem by finding one solution
void solve_set_cover_one_solution(vector<string>& relative,
                                  vector<string>& vec_primes,
                                  vector<string>& result) {
  int cnt_empty = std::count_if(relative.begin(), relative.end(),
                                [](string a) { return a.size() == 0; });
  string temp;
  while (cnt_empty < relative.size()) {
    do {
      temp = vec_primes.back();
      vec_primes.pop_back();
    } while (temp.size() == 0 && vec_primes.size());

    int count = 0;
    for (int i = 0; i < relative.size(); i++) {
      if (relative[i].size() && comp(in_bit_num, relative[i], temp)) {
        relative[i] = "";
        cnt_empty++;
        count++;
      }
    }
    if (count > 0) {
      result.push_back(temp);
    }
  }
}

// qm step 2
void find_results_org(vector<string>& vec_primes, vector<string>& relative,
                      vector<string>& result) {
  std::clock_t start = std::clock();

  // find essential prime implicates
  for (int i = 0; i < relative.size(); i++) {
    if (relative[i].empty()) continue;

    int count = 0, num = 0;
    for (int j = vec_primes.size() - 1; j >= 0; --j) {
      if (vec_primes.size() && comp(in_bit_num, relative[i], vec_primes[j])) {
        if (++count > 1) break;
        num = j;
      }
    }

    if (count == 1) {  // essential prime implicant
      result.push_back(vec_primes[num]);
      for (int j = 0; j < relative.size(); j++) {
        if (relative[j].size() &&
            comp(in_bit_num, relative[j], vec_primes[num])) {
          relative[j] = "";
        }
      }
      vec_primes[num] = "";
    }
  }

  // solve_set_cover_one_solution(relative, vec_primes, result);
  // time2 = (std::clock() - start) / double(CLOCKS_PER_SEC);
}

void find_results_serial(vector<string>& vec_primes, vector<string>& relative,
                         vector<string>& result) {
  std::clock_t start = std::clock();
  result = vector<string>(vec_primes.size(), "0");

  // find essential prime implicates
  for (int i = 0; i < relative.size(); i++) {
    if (relative[i] == "0") continue;

    int count = 0, num = 0;
    for (int j = vec_primes.size() - 1; j >= 0; --j) {
      if (vec_primes[j] != "0" &&
          comp(in_bit_num, relative[i], vec_primes[j])) {
        if (++count > 1) break;
        num = j;
      }
    }

    if (count == 1) {  // essential prime implicant
      for (int j = 0; j < relative.size(); j++) {
        if (relative[j] != "0" &&
            comp(in_bit_num, relative[j], vec_primes[num])) {
          relative[j] = "0";
        }
      }
      result[num] = vec_primes[num];
      vec_primes[num] = "0";
    }
  }

  for (int i = relative.size() - 1; i >= 0; --i) {
    if (relative[i] == "0") continue;
    for (int j = vec_primes.size() - 1; j >= 0; --j) {
      if (vec_primes[j] == "0") continue;
      if (comp(in_bit_num, relative[i], vec_primes[j])) {
        result[j] = vec_primes[j];
        break;
      }
    }
  }

  auto last = std::partition(result.begin(), result.end(), notempty);
  result.erase(last, result.end());
  time2 = (std::clock() - start) / double(CLOCKS_PER_SEC);
}

void runQM(int jobid) {
  readtruetable("input.pla" + std::to_string(jobid));

  vector<string> v;           // vector of strings that correponds to 1
  vector<string> vec_primes;  // primes in string format
  vector<string> result;      // vector<char*> result;

  prepinput(v);  // parse inputs that respond to output is 1

  vector<string> relative(v);

  // step 1
  find_primes_serial(v, vec_primes, in_bit_num);

  // step 2
  find_results_serial(vec_primes, relative, result);

  cout << std::setprecision(8) << std::setw(10);
  cout << in_bit_num << " " << time1 << " " << time2 << endl;
}

int main(int argc, char* argv[]) {
  // for (int i = 4; i < 28; ++i) runQM(i);
  runQM(atoi(argv[1]));
  // sort(result.begin(), result.end());
  // for (auto item : result) cout << item << endl;
  return 0;
}

/*
1) let i represents set of elements included so far.  initialize i = {}

2) do following while i is not same as u.
    a) find the set si in {s1, s2, ... sm} whose cost effectiveness is
       smallest, i.e., the ratio of cost c(si) and number of newly added
       elements is minimum.
       basically we pick the set for which following value is minimum.
       cost(si) / |si - i|
    b) add elements of above picked si to i, i.e.,  i = i u si
*/

void solve_set_cover_approx_greedy(vector<string>& relative,
                                   vector<string>& vec_primes,
                                   vector<string>& result) {}
