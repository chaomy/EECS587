#include <cmath>
#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

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

__inline__ __device__ void swap(double *a, double *b) {
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

__global__ void reduceSmemUnrollDyn(double *g_idata, double *g_odata, int n) {
  extern __shared__ double smem[];

  // set thread ID
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

  // unrolling 4
  double tmpSum = 0;

  if (idx < n) {
    double a1, a2, a3, a4;
    a1 = a2 = a3 = a4 = 0;
    a1 = g_idata[idx];
    if (idx + blockDim.x < n) a2 = g_idata[idx + blockDim.x];
    if (idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
    if (idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];
    tmpSum = a1 + a2 + a3 + a4;
  }

  smem[tid] = tmpSum;
  __syncthreads();

  // in-place reduction in global memory
  if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];

  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];

  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];

  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];

  __syncthreads();

  // unrolling warp
  if (tid < 32) {
    volatile double *vsmem = smem;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = smem[0];
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

  double sum{0};
  reduceSmemUnrollDyn<<<grid.x / 4, block>>>(d_A, d_B, NN);
  for (int i = 0; i < grid.x / 4; i++) sum += d_B[i];

  /* end timing */
  cout << " calculation time " << millisecond << " sum = " sum << endl;

  // cout << "sum = " << sum << " A[m][m] " << A[N / 2][N / 2] << " A[37][47] "
  //      << A[37][47] << " running time: " << duration << endl;

  cudaFree(d_A);
  cudaFree(d_B);
  free(A);
  free(B);
}

int main() { matrix_update(500); }
