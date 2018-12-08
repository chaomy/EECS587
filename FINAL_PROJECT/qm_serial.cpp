#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
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

struct {
  bool operator()(string a, string b) {
    size_t score_a = std::count(a.begin(), a.end(), '2');
    size_t score_b = std::count(b.begin(), b.end(), '2');
    return score_a == score_b ? true : score_a < score_b;
  }
} comparePrime;

// QM step 1
void find_primes(vector<string>& v, vector<string>& vec_primes) {
  vector<vector<string>> buckets(17);

  // store according to num of 1 bits
  unordered_set<string> prime;
  for (int j = 0; j < v.size(); j++)
    buckets[std::count(v[j].begin(), v[j].end(), '1')].push_back(v[j]);

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
          int res = checkBITs(in_bit_num, str_a, str_b);
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
  std::copy(prime.begin(), prime.end(), std::back_inserter(vec_primes));
  std::sort(vec_primes.begin(), vec_primes.end(), comparePrime);
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

/*
1) Let I represents set of elements included so far.  Initialize I = {}

2) Do following while I is not same as U.
    a) Find the set Si in {S1, S2, ... Sm} whose cost effectiveness is
       smallest, i.e., the ratio of cost C(Si) and number of newly added
       elements is minimum.
       Basically we pick the set for which following value is minimum.
       Cost(Si) / |Si - I|
    b) Add elements of above picked Si to I, i.e.,  I = I U Si
*/

void solve_set_cover_approx_greedy(vector<string>& relative,
                                   vector<string>& vec_primes,
                                   vector<string>& result) {}

// QM step 2
void find_results(vector<string>& vec_primes, vector<string>& relative,
                  vector<string>& result) {
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

  solve_set_cover_one_solution(relative, vec_primes, result);
}

int main() {
  readTrueTable("input.pla");

  int count;
  vector<string> v;           // vector of strings that correponds to 1
  vector<string> vec_primes;  // primes in string format
  vector<string> result;      // vector<char*> result;

  prepInput(v);
  vector<string> relative(v);

  std::clock_t start = std::clock();

  // step 1
  find_primes(v, vec_primes);

  // step 2
  find_results(vec_primes, relative, result);

  double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

  sort(result.begin(), result.end());
  for (auto item : result) cout << item << endl;
  cout << "time: " << duration << endl;
  return 0;
}