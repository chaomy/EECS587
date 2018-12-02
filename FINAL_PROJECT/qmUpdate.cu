/*
 * @Author: chaomy
 * @Date:   2018-12-02 00:25:05
 * @Last Modified by:   chaomy
 * @Last Modified time: 2018-12-02 00:25:52
 */

#include "qmHome.h"

static const int in_bit_num;
static const int out_bit_num;

void QMParallel::runQM() {
  int T{static_cast<int>(std::pow(3, in_bit_num))};
  size_t nBytes = T * sizeof(int);
  cout << "T = " << T << "nBytes = " << nBytes << endl;

  // int *A = (int *)malloc(nBytes);
  // int *d_A;

  // cudaMalloc((int **)&d_A, nBytes);
  // cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);
}