#include <cmath>
#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

// void mat_initialize(double *vec, int N) {
//   for (int i = N - 1; i >= 0; --i)
//     mat[i][j] = (1 + cos(2 * i) + sin(j)), mat[i][j] *= mat[i][j];
// }

// inline void find_small2(const double &a, const double &b, const double &c,
//                         const double &d, double &res) {
//   double slot[4];
//   if (a < b)
//     slot[0] = a, slot[1] = b;
//   else
//     slot[0] = b, slot[1] = a;

//   if (c < d)
//     slot[2] = c, slot[3] = d;
//   else
//     slot[2] = d, slot[3] = c;

//   res = slot[0] < slot[2] ? fmin(slot[1], slot[2]) : fmin(slot[0], slot[3]);
// }

__device__ void swap(double *a, double *b) {
  double tmp = *a;
  *a = *b;
  *b = tmp;
};

__global__ void update(double *A, double *B, int N) {
  double slot[4];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = idx / N, iy = idx % N;
  if (ix > 0 && ix < N && iy > 0 && iy < N - 1) {
    slot[0] = A[idx - N - 1], slot[1] = A[idx - N + 1];
    slot[2] = A[idx + N - 1], slot[3] = A[idx + N + 1];
    if (slot[1] < slot[0]) swap(&slot[0], &slot[1]);
    if (slot[3] < slot[2]) swap(&slot[2], &slot[3]);
    B[idx] = A[idx] + slot[0] < slot[2] ? fmin(slot[1], slot[2])
                                        : fmin(slot[0], slot[3]);
  }
}

void matrix_update(int N) {
  int NN{N * N};
  size_t nBytes = NN * sizeof(double);
  double *A = (double *)malloc(nBytes);
  double *B = (double *)malloc(nBytes);

  // initialize
  for (int k = NN - 1; k >= 0; --k) {
    int i{k / N}, j{k % N};
    A[k] = (1 + cos(2 * i) + sin(j)), A[k] *= A[k];
  }

  double *d_A, *d_B;
  cudaMalloc((double **)&d_A, nBytes);
  cudaMalloc((double **)&d_B, nBytes);
  cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);

  // block size BLOCK_X x 1, grid size
  int BLOCK_X = 32;

  dim3 block(BLOCK_X, 1);
  dim3 grid((NN + BLOCK_X - 1) / BLOCK_X, 1);

  cout << "grid " << grid.x << " block " << block.x << endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start the timer
  cudaEventRecord(start);

  for (int i = 0; i < 10; ++i) {
    update<<<grid, block>>>(d_A, d_B, N);
    cudaMemcpy(d_A, d_B, nBytes, cudaMemcpyDeviceToDevice);
  }

  // stop the timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millisecond = 0;
  cudaEventElapsedTime(&millisecond, start, stop);

  // double sum{0};
  // for (int i = N - 1; i >= 0; --i)
  //   for (int j = N - 1; j >= 0; --j) sum += A[i][j];

  /* end timing */
  cout << " calculation time " << millisecond << endl;

  // cout << "sum = " << sum << " A[m][m] " << A[N / 2][N / 2] << " A[37][47] "
  //      << A[37][47] << " running time: " << duration << endl;

  cudaFree(d_A);
  cudaFree(d_B);
  free(A);
  free(B);
}

int main() { matrix_update(500); }
