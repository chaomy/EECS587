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
// struct Lock {
//   int* mutex;
//   Lock() {
//     int state = 0;
//     cudaMalloc((void**)&mutex, sizeof(int));
//     cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
//   }
//   ~Lock() { cudaFree(mutex); }

//   __device__ void lock() {
//     while (atomicCAS(mutex, 0, 1) != 0)
//       ;
//   }
//   __device__ void unlock() { atomicExch(mutex, 0); }
// };

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

/*
  A[num * 3], existed
  A[num * 3 + 1], if find next
  A[num * 3 + 2], if self is found by previous
*/
__global__ void update(bool* A, uint64_t T, int numBit, uint64_t NumThread,
                       int numof2) {
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < NumThread) {
    for (uint64_t num = idx; num < T; num += NumThread) {
      if (A[3 * num] == 0) continue;
      int cnt_2 = 0;
      // convert 2 base to 3 base, count 2
      for (uint64_t tmp = num; tmp; tmp /= 3) {
        cnt_2 += (tmp % 3 == 2);
      }

      if (cnt_2 != numof2) continue;

      for (uint64_t tmp = num, cnt = 0, exp = 1; cnt < numBit;
           tmp /= 3, exp *= 3, ++cnt) {
        // only look  for pairs when the bit is 0
        if (tmp % 3 == 0) {
          uint64_t next = num + exp;
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

inline __device__ bool comp(int n, uint64_t num_base2, uint64_t num_base3) {
  for (; num_base2 || num_base3; num_base2 /= 2, num_base3 /= 3) {
    int ai = num_base2 % 2;
    int bi = num_base3 % 3;
    if (ai != bi && bi != 2) return false;
  }
  return true;
}

/*
  1. each thread looks for all primes if it only has corelation with one prime,
  that prime is essetial prime
  2. mask
*/
// __global__ void findEssentialPrimes(bool* A, bool* B, bool* C, int T,
//                                     int numBit, int NumThread) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx < NumThread && B[idx]) {
//     int cnt = 0;
//     for (int num = T - 1; num >= 0; --num) {
//       if (A[3 * num] && !A[3 * num + 1] && !A[3 * num + 2]) {
//         if (comp(numBit, idx, num) && ++cnt > 1) break;
//       }
//     }
//     if (cnt == 1) {
//       for (int num = T - 1; num >= 0; --num) {
//         if (A[3 * num] && !A[3 * num + 1] && !A[3 * num + 2]) C[num] = true;
//       }
//     }
//   }
// }

__global__ void findEssentialPrimes(bool* B, bool* C, uint64_t* primes,
                                    int prime_size, int numBit,
                                    uint64_t NumThread) {
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  int first_meet = -1;
  if (idx < NumThread && B[idx]) {
    for (int i = prime_size - 1; i >= 0; --i) {
      if (comp(numBit, idx, primes[i])) {
        if (first_meet != -1) {
          first_meet = -2;
          break;
        }
        first_meet = primes[i];
      }
    }
    if (first_meet >= 0) {
      C[first_meet] = true;
    }
  }
}

// mask relatives that is related to essential primes
__global__ void maskRelatives(bool* B, bool* C, uint64_t* primes,
                              int prime_size, int numBit, uint64_t NumThread) {
  // C is essential primes, C[num] = '1' means num is an essential prime
  // B is relatives
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < NumThread && B[idx]) {
    for (int i = prime_size - 1; i >= 0; --i) {
      if (C[primes[i]] && comp(numBit, idx, primes[i])) {
        B[idx] = 0;
      }
    }
  }
}

__global__ void findResults(bool* B, bool* C, uint64_t* primes, int prime_size,
                            int numBit, uint64_t NumThread) {
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < NumThread && B[idx]) {  // is a relative
    for (int i = prime_size - 1; i >= 0; --i) {
      if (comp(numBit, idx, primes[i])) {
        C[primes[i]] = true;
        break;
      }
    }
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

void readTrueTable(string fname) {
  ifstream s(fname, std::iostream::in);
  string line;

  getline(s, line, ' ');
  getline(s, line);
  in_bit_num = stoi(line);

  getline(s, line, ' ');
  getline(s, line);
  out_bit_num = stoi(line);

  string buff1, buff2;
  while (getline(s, buff1, ' ') && getline(s, buff2)) {
    input.push_back(buff1);
    output.push_back(buff2);
  }
}

template <const int Base>
inline uint64_t convertStr2Num(string s) {
  uint64_t num{0}, base{1};
  for (int i = s.size() - 1; i >= 0; --i, base *= Base)
    num += (s[i] - '0') * base;
  return num;
}

inline string convertTo3baseStr(uint64_t num) {
  string res(in_bit_num, '0');
  for (uint64_t p = in_bit_num - 1; num; num /= 3) res[p--] = (num % 3) + '0';
  return res;
}

template <typename T>
struct comparePrime {
  bool operator()(T a, T b) {
    int cnta{0}, cntb{0};
    for (; a; a /= 3) cnta += (a % 3 == 2);
    for (; b; b /= 3) cntb += (b % 3 == 2);
    return cnta == cntb ? false : cnta < cntb;
  }
};

void runQMgpu(int jobid, int blocksize) {
  int BLOCK_X = blocksize;
  readTrueTable("input.pla" + std::to_string(jobid));

  vector<string> v;
  vector<string> prime;   // vector<char*> prime;
  vector<string> result;  // vector<char*> result;

  prepInput(v);
  vector<string> relative(v);

  // cout << "Input " << endl;
  // std::copy(v.begin(), v.end(), std::ostream_iterator<string>(cout, "\n"));

  uint64_t T{static_cast<uint64_t>(pow(3, in_bit_num))};
  uint64_t T3(T * 3);
  int prime_size_limit{100000000};

  size_t nBytesA = T3 * sizeof(bool);
  size_t nBytesB = (1 << in_bit_num) * sizeof(bool);
  size_t nBytesC = T * sizeof(bool);

  bool* A = (bool*)malloc(nBytesA);
  bool* B = (bool*)malloc(nBytesB);
  bool* C = (bool*)malloc(nBytesC);

  uint64_t* primes = (uint64_t*)malloc(prime_size_limit * sizeof(uint64_t));

  // initialize
  memset(A, false, nBytesA);
  memset(B, false, nBytesB);
  memset(C, false, nBytesC);

  for (int i = 0; i < input.size(); ++i) {
    if (output[i][0] == '1' || output[i][0] == '2') {
      uint64_t in_num_base3 = convertStr2Num<3>(input[i]);
      uint64_t in_num_base2 = convertStr2Num<2>(input[i]);
      A[in_num_base3 * 3] = true;
      B[in_num_base2] = output[i][0] == '1';
    }
  }

  bool* d_A;  // whole space
  bool* d_B;  // mark relative
  bool* d_C;  // mark final results

  uint64_t* d_primes;  // vector of primes implicates

  cudaMalloc((bool**)&d_A, nBytesA);
  cudaMalloc((bool**)&d_B, nBytesB);
  cudaMalloc((bool**)&d_C, nBytesC);

  cudaMemcpy(d_A, A, nBytesA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, nBytesB, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, nBytesC, cudaMemcpyHostToDevice);

  // block
  dim3 block(BLOCK_X, 1);
  dim3 grid(((1 << in_bit_num) + BLOCK_X - 1) / BLOCK_X, 1);

  float time1{0}, time2{0};
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start the timer
  cudaEventRecord(start);

  for (int round = 0; round < in_bit_num; ++round) {
    update<<<grid.x, block.x>>>(d_A, T, in_bit_num, 1 << in_bit_num, round);
  }

  // stop the timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time1, start, stop);

  cudaMemcpy(A, d_A, nBytesA, cudaMemcpyDeviceToHost);

  uint64_t avail = 0;
  for (uint64_t num = 0; num < T; ++num) {
    if (A[3 * num] && !A[3 * num + 1] && !A[3 * num + 2]) {
      primes[avail++] = num;
      if (avail == prime_size_limit - 10) {
        cout << avail << endl;
        free(A);
        free(B);
        free(C);
        cudaFree(d_A);
        cudaFree(d_C);
        cudaFree(d_B);
        return;
      }
    }
  }

  // sort based on num of '2' in the prime
  std::sort(primes, primes + avail, comparePrime<uint64_t>());

  cudaMalloc((uint64_t**)&d_primes, prime_size_limit * sizeof(uint64_t));
  cudaMemcpy(d_primes, primes, avail * sizeof(uint64_t),
             cudaMemcpyHostToDevice);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start the timer
  cudaEventRecord(start);

  // first find essential prime implicate first,
  findEssentialPrimes<<<grid.x, block.x>>>(d_B, d_C, d_primes, avail,
                                           in_bit_num, 1 << in_bit_num);

  // delete those relatives related to essential prime
  maskRelatives<<<grid.x, block.x>>>(d_B, d_C, d_primes, avail, in_bit_num,
                                     1 << in_bit_num);

  // CPU find prime
  findResults<<<grid.x, block.x>>>(d_B, d_C, d_primes, avail, in_bit_num,
                                   1 << in_bit_num);

  // stop the timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time2, start, stop);

  cout << BLOCK_X << " " << in_bit_num << " " << time1 << " " << time2 << endl;

  cudaMemcpy(C, d_C, nBytesC, cudaMemcpyDeviceToHost);

  for (uint64_t num = 0; num < T; ++num)
    if (C[num]) result.push_back(convertTo3baseStr(num));

  free(A);
  free(B);
  free(C);
  cudaFree(d_A);
  cudaFree(d_C);
  cudaFree(d_B);
}

int main(int argc, char** argv) {
  runQMgpu(atoi(argv[1]), atoi(argv[2]));
  return 0;
}

// __global__ void takePrime(bool* A, int T, int NumThread, int* size, int*
// primes,
//                           Lock mylock) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx < NumThread) {
//     for (int num = idx; num < T; num = num + NumThread) {
//       if (A[3 * num] && !A[3 * num + 1] && !A[3 * num + 2]) {
//         mylock.lock();
//         primes[(*size)++] = num;
//         mylock.unlock();
//       }
//     }
//   }
// }

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
