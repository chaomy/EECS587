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

  for (auto s : input) {
    bitset<32> bits(s);
    std::cout << "foo: " << bits << '\n';
  }

  int *d_A;

  cudaMalloc((int **)&d_A, nBytes);
  cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);
}