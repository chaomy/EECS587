#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <stdio.h> 
#include <stdlib.h> 

using std::cout;
using std::endl;
using std::ifstream;
using std::string;
using std::unordered_set;
using std::vector;

/*
        attention !!!
        write struct Lock{
                ...
        }
*/
struct Lock {
  int* mutex;
  Lock() {
    int state = 0;
    cudaMalloc((void**)&mutex, sizeof(int));
    cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
  }
  ~Lock() { cudaFree(mutex); }

  __device__ void lock() {
    while (atomicCAS(mutex, 0, 1) != 0);
  }
  __device__ void unlock() { atomicExch(mutex, 0); }
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

int in_bit_num, out_bit_num;
vector<string> in_labels, out_labels;
vector<string> input, output;

/*
  A[num * 3], existed 
  A[num * 3 + 1], if find next 
  A[num * 3 + 2], if self is found by previous 
*/
__global__ void update(bool* A, int T, int NumThread, int numof2) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < NumThread) {
    for (int num = idx; num < T; num = num + NumThread) {
      if (A[3 * num] == 0) continue; 
      int cnt_2 = 0;
      // convert 2 base to 3 base, count 2
      for (int tmp = num; tmp; tmp /= 3) {
        cnt_2 += (tmp % 3 == 2);
      }

      if (cnt_2 != numof2) continue;

      for (int tmp = num, exp = 1; tmp; tmp /= 3, exp *= 3) {
        // only look for pairs when the bit is 0 
        if (tmp % 3 == 0) {
          int next = num + exp;
          if (A[3 * next]) {
            A[3 * (next + exp)] = true;
            A[3 * num + 1] = true;
            A[3 * next + 2] = true;
          }
        }
      }
    }
  }
}

__global__ void takePrime(bool* A, int T, int NumThread, int* size,
                          int* primes, Lock mylock) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < NumThread) {
    for (int num = idx; num < T; num = num + NumThread) {
      if (A[3 * num] && !A[3 * num + 1] && !A[3 * num + 2]) {
        printf("find prime %x\n", num);
        mylock.lock();
        primes[(*size)++] = num;
        mylock.unlock();
      }
    }
  }
}

// __global__ void assignEachRoundJob(){
// 	extern __shared__ int
// }

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

  // read head
  while (getline(s, line) && (line != ".e")) {
    vector<string> buff;
    split(line, " ", buff);
    input.push_back(buff[0]);
    output.push_back(buff[1]);
  }
}

inline int convertStr2Num(string s) {
  int num{0}, base{1};
  for (int i = s.size() - 1; i >= 0; --i, base *= 3) num += (s[i] - '0') * base;
  return num;
}

int main() {
  int BLOCK_X = 256;
  readTrueTable("input.pla");

  vector<string> v;
  vector<string> prime;   // vector<char*> prime;
  vector<string> result;  // vector<char*> result;

  prepInput(v);
  vector<string> relative(v);

  cout << "Input " << endl;
  // std::copy(v.begin(), v.end(), std::ostream_iterator<string>(cout, "\n"));

  int T{static_cast<int>(pow(3, in_bit_num))};
  int T3(T * 3);
  size_t nBytes = T3 * sizeof(bool);

  bool* A = (bool*)malloc(nBytes);
  int* primes = (int*)malloc(1000 * sizeof(int));
  int prime_size = 0;

  Lock mylock;

  // initialize
  memset(A, false, nBytes);

  for (int i = 0; i < input.size(); ++i) {
    int in_num = convertStr2Num(input[i]);
    if (output[i][0] == '1') {
      A[in_num * 3] = true;
      cout << input[i] <<" " << in_num << endl; 
    }
  }

  bool* d_A;
  int* d_primes;
  int* d_prime_size;

  cudaMalloc((bool**)&d_A, nBytes);
  cudaMalloc((int**)&d_primes, 1000 * sizeof(int));
  cudaMalloc((int**)&d_prime_size, sizeof(int));

  cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_prime_size, &prime_size, sizeof(int), cudaMemcpyHostToDevice);

  // block
  dim3 block(BLOCK_X, 1);
  dim3 grid(((1 << in_bit_num) + BLOCK_X - 1) / BLOCK_X, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start the timer
  cudaEventRecord(start);

  // __global__ void update(bool* A, int T, int NumThread, int numof2){
  for (int round = 0; round < in_bit_num; ++round) {
    update<<<grid.x, block.x>>>(d_A, T, 1 << in_bit_num, round);
  }

  // takePrime<<<grid.x, block.x>>>(d_A, T, 1 << in_bit_num, d_prime_size,
  //                                d_primes, mylock);

  cudaMemcpy(&prime_size, d_prime_size, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(primes, d_primes, 1000 * sizeof(int), cudaMemcpyDeviceToHost);

  // stop the timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  for (int i = 0; i < prime_size; ++i) cout << primes[i] << endl;

  // to be parallelet
  // for (int i = 0; i < 16; i++) {
  //   auto it = std::find_if(buckets.begin(), buckets.end(),
  //                          [](const vector<int>& a) { return a.size(); });
  //   if (it == buckets.end()) break;

  //   vector<vector<int>> next(17);
  //   vector<bool> flag(v.size());

  //   // update bucket
  //   for (int j = 0; j < 16; ++j) {
  //     for (auto a : buckets[j]) {
  //       for (auto b : buckets[j + 1]) {
  //         int res = checkBITs(16, v[a], v[b]);
  //         if (res != -1) {  // can merge
  //           flag[a] = 1, flag[b] = 1;
  //           v[a][res] = '2';
  //           next[j].push_back(a);
  //         }
  //       }
  //       if (flag[a] == 0) prime.push_back(v[a]);
  //     }
  //   }
  //   buckets = std::move(next);
  // }

  // int count;
  // string temp;

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
