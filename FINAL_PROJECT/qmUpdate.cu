/*
 * @Author: chaomy
 * @Date:   2018-12-02 00:25:05
 * @Last Modified by:   chaomy
 * @Last Modified time: 2018-12-02 00:25:52
 */

#include "qmHome.h"

__global__ void upate(int *A, int T) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < T) {
    printf("my idx %d num is %d \n", idx, A[idx]);
  }
}

inline int convertStr2Num(string s) {
  int num{0}, base{0};
  for (int i = s.size() - 1; i >= 0; --i, base *= 3) num += (s[i] - '0') * base;
  return num;
}

void QMParallel::runQM(int BLOCK_X = 128) {
  int T{static_cast<int>(std::pow(3, in_bit_num))};
  size_t nBytes = T * sizeof(int);
  cout << "T = " << T << "nBytes = " << nBytes << endl;

  int *A = (int *)malloc(nBytes);

  // parse each string to num
  for (int i = 0; i < input.size(); ++i) {
    int in_num = convertStr2Num(input[i]);
    int out_num = convertStr2Num(output[i]);
    A[in_num] = out_num;
  }

  int *d_A;
  cudaMalloc((int **)&d_A, nBytes);
  cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);

  // schedule block
  dim3 block(BLOCK_X, 1);
  dim3 grid((T + BLOCK_X - 1) / BLOCK_X, 1);

  update<<<grid.x, block.x>>>(d_A, T);

  cudaMemcpy(A, d_A, nBytes, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  free(A);
}