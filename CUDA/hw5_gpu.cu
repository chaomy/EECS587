#include <cmath>
#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

__inline__ __device__ void swap(float &a, float &b) {
  float tmp = a;
  a = b;
  b = tmp;
};

__global__ void update(float *A, float *B, int N) {
  float slot[4];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N * N) {
    int ix = idx / N, iy = idx % N;
    if (ix > 0 && ix < N - 1 && iy > 0 && iy < N - 1) {
      slot[0] = A[idx - N - 1], slot[1] = A[idx - N + 1];
      slot[2] = A[idx + N - 1], slot[3] = A[idx + N + 1];
      if (slot[1] < slot[0]) swap(slot[0], slot[1]);
      if (slot[3] < slot[2]) swap(slot[2], slot[3]);
      B[idx] = A[idx] + (slot[0] < slot[2] ? fmin(slot[1], slot[2])
                                           : fmin(slot[0], slot[3]));
    }
  }
}

// template <unsigned int blockSize>
// __global__ void reduceSmemDyn(float *g_idata, float *g_odata, unsigned int n)
// {
//   extern __shared__ float smem[];

//   // set thread ID
//   // unsigned int tid = threadIdx.x;
//   // float *idata = g_idata + blockIdx.x * blockDim.x;

//   // new version
//   unsigned int tid = threadIdx.x;
//   unsigned int i = blockIdx.x * (blockSize * 2) + tid;
//   unsigned int gridSize = blockSize * 2 * gridDim.x;

//   smem[tid] = 0;
//   while (i < n) {
//     smem[tid] += g_idata[i] + g_idata[i + blockSize];
//     i += gridSize;
//   }
//   __syncthreads();

//   // set to smem by each threads
//   // smem[tid] = idata[tid];
//   // __syncthreads();

//   // in-place reduction in global memory
//   if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];

//   __syncthreads();

//   if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];

//   __syncthreads();

//   if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];

//   __syncthreads();

//   if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];

//   __syncthreads();

//   // unrolling warp
//   if (tid < 32) {
//     volatile float *vsmem = smem;
//     vsmem[tid] += vsmem[tid + 32];
//     vsmem[tid] += vsmem[tid + 16];
//     vsmem[tid] += vsmem[tid + 8];
//     vsmem[tid] += vsmem[tid + 4];
//     vsmem[tid] += vsmem[tid + 2];
//     vsmem[tid] += vsmem[tid + 1];
//   }

//   // write result for this block to global mem
//   if (tid == 0) g_odata[blockIdx.x] = smem[0];
// }

__global__ void reduceSmemDyn(float *g_idata, float *g_odata, unsigned int n) {
  extern __shared__ float smem[];

  // set thread ID
  unsigned int tid = threadIdx.x;
  float *idata = g_idata + blockIdx.x * blockDim.x;

  // set to smem by each threads
  smem[tid] = idata[tid];
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
    volatile float *vsmem = smem;
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
  size_t nBytes = NN * sizeof(float);
  float *A = (float *)malloc(nBytes);
  float *B = (float *)malloc(nBytes);
  float res[3] = {0, 0, 0};
  int p1{N / 2 * N + N / 2}, p2{37 * N + 47};

  // initialize
  for (int k = NN - 1; k >= 0; --k) {
    int i{k / N}, j{k % N};
    A[k] = (1 + cos(2 * i) + sin(j)), A[k] *= A[k];
  }

  float *d_A, *d_B;
  cudaMalloc((float **)&d_A, nBytes);
  cudaMalloc((float **)&d_B, nBytes);

  cudaMemcpy(d_A, A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, A, nBytes, cudaMemcpyHostToDevice);

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

  int num_iter = 5;
  for (int i = 0; i < num_iter; ++i) {
    update<<<grid.x, block.x>>>(d_A, d_B, N);
    update<<<grid.x, block.x>>>(d_B, d_A, N);
  }

  cudaMemcpy(&res[1], &d_A[p1], sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&res[2], &d_A[p2], sizeof(float), cudaMemcpyDeviceToHost);

  const int BLOCK_SIZE = 512;
  for (int total = NN, blockTotal; total > 1; total = blockTotal) {
    blockTotal = total / BLOCK_SIZE + (total % BLOCK_SIZE == 0 ? 0 : 1);
    reduceSmemDyn<<<blockTotal, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_A, d_A, total);
  }

  // stop the timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millisecond = 0;
  cudaEventElapsedTime(&millisecond, start, stop);

  cudaMemcpy(&res[0], &d_A[0], sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(&res[1], &d_B[p1], sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(&res[2], &d_B[p2], sizeof(float), cudaMemcpyDeviceToHost);

  /* end timing */
  cout << " calculation time " << millisecond << " sum = " << res[0]
       << " A[N / 2][N / 2] " << res[1] << " A[37][47] " << res[2] << endl;

  cudaFree(d_A);
  cudaFree(d_B);
  free(A);
  free(B);
}

int main() { matrix_update(2000); }
