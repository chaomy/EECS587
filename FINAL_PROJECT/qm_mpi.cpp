#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/datatype_fwd.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/operations.hpp>
#include <boost/mpi/timer.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/unordered_set.hpp>
#include <boost/serialization/vector.hpp>
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
#define ROOT 0

namespace mpi = boost::mpi;

// mpic++ qm_mpi.cpp  -std=c++11 -lboost_mpi -lboost_serialization -lmpi  -o
// qm_mpi  -O3 -g mpic++ qm_mpi.cpp  -std=c++11 -lboost_mpi-mt
// -lboost_serialization-mt  -o qm_mpi  -O3 -g -L${BOOST_LIB} -I${BOOST_INCLUDE}

bool comp(int n, const string& a, const string& b) {
  for (int i = 0; i < n; i++) {
    if (a[i] != b[i] && (a[i] != '2' && b[i] != '2')) return false;
  }
  return true;
}

int checkbits(int n, const string& a, const string& b) {
  int count = 0, temp;
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      if (++count > 1) return -1;
      temp = i;
    }
  }
  return count == 1 ? temp : -1;
}

int readtruetable(string fname, vector<string>& input, vector<string>& output) {
  ifstream s(fname, std::iostream::in);
  string line;

  getline(s, line, ' ');
  getline(s, line);
  int in_bit_num = stoi(line);

  getline(s, line, ' ');
  getline(s, line);
  int out_bit_num = stoi(line);

  // read head
  string buff1, buff2;
  while (getline(s, buff1, ' ') && getline(s, buff2)) {
    input.push_back(buff1);
    output.push_back(buff2);
  }
  return in_bit_num;
}

void prepinput(vector<string>& v, vector<string>& input,
               vector<string>& output) {
  size_t n{input.size()};
  v.reserve(n);
  for (int i = 0; i < n; ++i) {
    if (output[i][0] == '1' || output[i][0] == '2') {
      v.push_back(input[i]);
    }
  }
}

struct {
  bool operator()(const string& a, const string& b) {
    size_t score_a = std::count(a.begin(), a.end(), '2');
    size_t score_b = std::count(b.begin(), b.end(), '2');
    return score_a == score_b ? false : score_a < score_b;
  }
} compareprime;

template <const int Base>
int convertStrToNum(const string& input) {
  int res{0};
  for (int i = input.size() - 1, mul = 1; i >= 0; --i, mul *= Base)
    res = res + (input[i] - '0') * mul;
  return res;
}

template <const int Base>
string convertNumToStr(int num, int in_bit_num) {
  string res(in_bit_num, '0');
  for (int p = in_bit_num - 1; num; num /= Base) res[p--] = (num % Base) + '0';
  return res;
}

string maskEmpty(const string& a, const string& b) {
  return a.empty() || b.empty() ? "" : a;
}

string mergeItem(const string& a, const string& b) {
  return a.empty() && b.empty() ? "" : a.empty() ? b : a;
}

bool notEmpty(const string& a) { return a.size(); }

void find_results(mpi::communicator& cmm, vector<string>& vec_primes,
                  vector<string>& relative, vector<string>& result,
                  int in_bit_num) {
  int worker_num = cmm.size(), id = cmm.rank();
  int workload = relative.size();
  int remains = workload % worker_num;
  int local_len = workload / worker_num + (id < remains);
  int start_id = local_len * id + (id < remains ? 0 : remains);
  int end_id = start_id + local_len - 1;

  vector<string> masked_relative(relative);
  vector<string> masked_primes(vec_primes);
  vector<string> local_res(vec_primes.size());
  result = local_res;

  mpi::timer myclock;

  for (int i = start_id; i <= end_id; ++i) {
    if (relative[i].empty()) continue;

    int count = 0, record_id = 0;

    // go through all primes backward to check for essential implicates
    for (int j = vec_primes.size() - 1; j >= 0; --j) {
      if (vec_primes.size() && comp(in_bit_num, relative[i], vec_primes[j])) {
        if (++count > 1) break;
        record_id = j;
      }
    }

    // use essential prime implicates to mask relatives
    if (count == 1) {
      for (auto& org : relative) {
        if (org.size() && comp(in_bit_num, org, vec_primes[record_id]))
          org = "";
      }
      std::swap(local_res[record_id], vec_primes[record_id]);
    }
  }

  // communicates the relative and vec_primes
  all_reduce(cmm, relative.data(), relative.size(), masked_relative.data(),
             maskEmpty);
  all_reduce(cmm, vec_primes.data(), vec_primes.size(), masked_primes.data(),
             maskEmpty);

  // find one solution
  for (int i = start_id; i <= end_id; ++i) {
    if (masked_relative[i].empty()) continue;

    for (int j = masked_primes.size() - 1; j >= 0; --j) {
      if (masked_primes[j].empty()) continue;  // ignore essential implicates
      if (comp(in_bit_num, masked_relative[i], masked_primes[j])) {
        local_res[j] = masked_primes[j];
        break;
      }
    }
  }

  reduce(cmm, local_res.data(), local_res.size(), result.data(), mergeItem,
         ROOT);

  if (id == ROOT) {
    cout << myclock.elapsed() << " ";
    auto last = std::partition(result.begin(), result.end(), notEmpty);
    result.erase(last, result.end());
    cout << result.size() << endl;
  }
}

void find_primes(mpi::communicator& cmm, vector<string>& v,
                 vector<string>& vec_primes, int in_bit_num) {
  int worker_num = cmm.size(), id = cmm.rank();
  int firstworker{0}, lastworker{worker_num - 1};

  int bucketsize = in_bit_num + 1;
  int workload = bucketsize - 1;

  int remains = workload % worker_num;
  int local_len = workload / worker_num + (id < remains);
  int start_id = local_len * id + (id < remains ? 0 : remains);
  int end_id = start_id + local_len - 1;

  vector<vector<string>> buckets(bucketsize), next(bucketsize);
  vector<unordered_set<string>> vec_flags(bucketsize);
  vector<unordered_set<string>> vec_buffs(bucketsize);

  for (auto key : v)
    buckets[std::count(key.begin(), key.end(), '1')].push_back(key);

  // record time for finding primes
  mpi::timer myclock;

  // store according to num of 1 bits
  unordered_set<string> prime;
  bool totaldone{false};

  for (int i = 0; i < in_bit_num; ++i) {
    bool localdone{false};
    auto it =
        std::find_if(buckets.begin() + start_id, buckets.begin() + end_id + 1,
                     [](const vector<string>& a) { return a.size(); });
    if (it == buckets.begin() + end_id + 1) localdone = true;

    all_reduce(cmm, localdone, totaldone, std::logical_and<bool>());
    if (totaldone) break;

    // update buckets
    // for (int j = end_id; j >= start_id; --j) {
    //   // before do the start_id must recieve from its neigh
    //   // cout << "I am " << cmm.rank() << ", j = " << j << endl;
    //   if (j == start_id && id != firstworker) {
    //     cmm.recv(id - 1, 0, vec_buffs[start_id]);
    //   }

    //   for (auto str_b : buckets[j + 1]) {
    //     for (auto str_a : buckets[j]) {
    //       int res = checkbits(in_bit_num, str_a, str_b);
    //       if (res != -1) {
    //         vec_flags[j].insert(str_a);
    //         vec_flags[j + 1].insert(str_b);
    //         str_a[res] = '2';
    //         next[j].push_back(str_a);
    //         str_a[res] = '0';
    //       }
    //     }  // loop over level j

    //     // add prime
    //     if (j != end_id &&
    //         vec_flags[j + 1].find(str_b) == vec_flags[j + 1].end()) {
    //       prime.insert(str_b);
    //       cout << "I am " << cmm.rank() << ", insert " << str_b << endl;
    //     }
    //   }  // loop over level j + 1

    //   if (j == start_id) {
    //     for (auto str_a : buckets[j]) {
    //       if (vec_flags[j].find(str_a) == vec_flags[j].end() &&
    //           vec_buffs[j].find(str_a) == vec_buffs[j].end()) {
    //         cout << "I am " << cmm.rank() << ", insert " << str_a << endl;
    //         prime.insert(str_a);
    //       }
    //     }
    //   }

    //   // send when finished end_id
    //   if (j == end_id && id != lastworker) {
    //     vec_buffs[end_id + 1] = vec_flags[end_id + 1];
    //     cmm.isend(id + 1, 0, vec_buffs[end_id + 1]);
    //   }
    // }  // loop over level

    // update bucket
    for (int j = start_id; j <= end_id; ++j) {
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
        if (j != start_id && vec_flags[j].find(str_a) == vec_flags[j].end())
          prime.insert(str_a);
      }  // loop over all items on buket layer i
      vec_flags[j].clear();
      buckets[j] = std::move(next[j]);
    }  // loop over all layers

    // communicates
    if (id != firstworker) cmm.send(id - 1, 0, buckets[start_id]);
    if (id != lastworker) cmm.recv(id + 1, 0, buckets[end_id + 1]);

    if (id != firstworker) {
      cmm.recv(id - 1, 0, vec_buffs[start_id]);
    }

    if (id != lastworker) {
      vec_buffs[end_id + 1] = vec_flags[end_id + 1];
      cmm.send(id + 1, 0, vec_buffs[end_id + 1]);
    }

    // insert start
    for (auto str_a : buckets[start_id]) {
      if (vec_flags[start_id].find(str_a) == vec_flags[start_id].end() &&
          vec_buffs[start_id].find(str_a) == vec_buffs[start_id].end())
        prime.insert(str_a);
    }
  }

  vector<string> vec_primes_local;

  std::copy(prime.begin(), prime.end(), std::back_inserter(vec_primes_local));

  // std::transform(prime.begin(), prime.end(),
  //                std::back_inserter(vec_primes_local), convertStrToNum<3>);

  int local_prime_size = vec_primes_local.size();
  int total_prime_size{0}, send_prime_size{0};

  // cout << "I am " << id << " start " << start_id << " end " << end_id << "
  // len "
  //      << local_len << " bucketsize " << bucketsize << " remains is " <<
  //      remains
  //      << " vec_primes " << vec_primes_local.size() << " send size "
  //      << send_prime_size << endl;

  all_reduce(cmm, local_prime_size, send_prime_size, mpi::maximum<int>());

  vec_primes_local.resize(send_prime_size);

  vector<string> each_prime_sizes(worker_num);
  vector<int> vec_primes_all;

  vec_primes_all.reserve(send_prime_size * worker_num);

  gather(cmm, vec_primes_local.data(), send_prime_size, vec_primes_all.data(),
         ROOT);
  gather(cmm, local_prime_size, each_prime_sizes, ROOT);

  if (cmm.rank() == ROOT) cout << myclock.elapsed() << " ";

  if (id == ROOT) {
    // convert gathered primes<int> to the vec_prime<string>
    for (int i = 0; i < worker_num; ++i) {
      int start = i * send_prime_size;
      int end = i * send_prime_size + each_prime_sizes[i];
      vector<int> size_param(send_prime_size, in_bit_num);

      // assign vec_primes
      std::transform(vec_primes_all.begin() + start,
                     vec_primes_all.begin() + end, size_param.begin(),
                     std::back_inserter(vec_primes), convertNumToStr<3>);
      std::sort(vec_primes.begin(), vec_primes.end(), compareprime);
    }
    cout << vec_primes.size() << endl;
  }

  // pass vec_primes to everyone
  broadcast(cmm, vec_primes, ROOT);

  // if (cmm.rank() == 2)
  // copy(vec_primes.begin(), vec_primes.end(),
  //      std::ostream_iterator<string>(cout, "\n"));
}

int main() {
  mpi::environment env;
  mpi::communicator cmm;

  int in_bit_num{0};
  vector<string> in_labels, out_labels;
  vector<string> input, output;
  vector<string> v;  // vector of strings that correponds to 1

  if (cmm.rank() == ROOT) {
    in_bit_num = readtruetable("input.pla", input, output);
    prepinput(v, input, output);  // parse inputs that respond to output is 1
  }

  broadcast(cmm, in_bit_num, ROOT);
  broadcast(cmm, v, ROOT);

  vector<string> relative(v);
  vector<string> vec_primes;  // primes in string format
  vector<string> result;      // vector<char*> result;

  // step 1
  find_primes(cmm, v, vec_primes, in_bit_num);

  // step 2
  find_results(cmm, vec_primes, relative, result, in_bit_num);

  // sort(result.begin(), result.end());
  // for (auto item : result) cout << item << endl;
  return 0;
}

/* hahahahahahha
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