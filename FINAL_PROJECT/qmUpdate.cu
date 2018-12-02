/*
 * @Author: chaomy
 * @Date:   2018-12-02 00:25:05
 * @Last Modified by:   chaomy
 * @Last Modified time: 2018-12-02 00:25:52
 */

#include "qmHome.h"

void QMParallel::runQM() {
  int T{static_cast<int>(std::pow(3, in_bit_num))};
  size_t nBytes = T * sizeof(int);
  cout << "T = " << T << "nBytes = " << nBytes << endl;

  int *A = (int *)malloc(nBytes);

  // parse each string to num
  for (auto s : input) {
    int num{0}, base{1};
    for (int i = s.size() - 1; i >= 0; --i, base *= 3)
      num += (s[i] - '0') * base;
    cout << "num is " << num << endl;
  }

  int *d_A;
  cudaMalloc((int **)&d_A, nBytes);
  cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);
}