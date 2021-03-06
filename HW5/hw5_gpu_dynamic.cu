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

__global__ void parent(float *A, float *B, int N, int GRID_X, int BLOCK_X) {
  for (int i = 0; i < 5; ++i) {
    update<<<GRID_X, BLOCK_X>>>(A, B, N);
    __syncthreads();
    update<<<GRID_X, BLOCK_X>>>(B, A, N);
    __syncthreads();
  }
}

__global__ void reduceSum(float *A, float *S, int size) {
  extern __shared__ float shr_data[];
  size_t tid = threadIdx.x;
  size_t mid = threadIdx.x + blockIdx.x * blockDim.x;

  // initialize dynamic shared memory
  if (mid < size)
    shr_data[tid] = A[mid];
  else
    shr_data[tid] = 0;
  __syncthreads();

  for (size_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) shr_data[tid] += shr_data[tid + stride];
    __syncthreads();
  }

  if (tid < 32) {  // unrolling warp
    volatile float *v_shr_data = shr_data;
    v_shr_data[tid] += v_shr_data[tid + 32];
    v_shr_data[tid] += v_shr_data[tid + 16];
    v_shr_data[tid] += v_shr_data[tid + 8];
    v_shr_data[tid] += v_shr_data[tid + 4];
    v_shr_data[tid] += v_shr_data[tid + 2];
    v_shr_data[tid] += v_shr_data[tid + 1];
  }

  if (tid == 0) S[blockIdx.x] = shr_data[0];
};

void matrix_update(int N, int BLOCK_X = 128) {
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
  dim3 block(BLOCK_X, 1);
  dim3 grid((NN + BLOCK_X - 1) / BLOCK_X, 1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start the timer
  cudaEventRecord(start);

  parent<<<1, 1>>>(d_A, d_B, N, grid.x, block.x);

  cudaMemcpy(&res[1], &d_A[p1], sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&res[2], &d_A[p2], sizeof(float), cudaMemcpyDeviceToHost);

  for (int total = NN, blockTotal; total > 1; total = blockTotal) {
    blockTotal = (total + BLOCK_X - 1) / BLOCK_X;
    reduceSum<<<blockTotal, BLOCK_X, BLOCK_X * sizeof(float)>>>(d_A, d_A,
                                                                total);
  }

  // stop the timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millisecond = 0;
  cudaEventElapsedTime(&millisecond, start, stop);

  cudaMemcpy(&res[0], &d_A[0], sizeof(float), cudaMemcpyDeviceToHost);

  /* end timing */
  cout << "N " << N << " grid " << grid.x << " block " << block.x << " time "
       << millisecond << " sum " << res[0] << " chk1 " << res[1] << " chk2 "
       << res[2] << endl;

  cudaFree(d_A);
  cudaFree(d_B);
  free(A);
  free(B);
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);        // problem size
  int BLOCK_X = atoi(argv[2]);  // block size
  matrix_update(N, BLOCK_X);
}
