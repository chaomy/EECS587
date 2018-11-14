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

__global__ void reduceSmemDyn(float *A, float *S, int size) {
  extern __shared__ float sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  // initialize dynamic shared memory
  if (i < size)
    sdata[tid] = A[i];
  else
    sdata[tid] = 0;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid < 32) {  // unrolling warp
    volatile float *vsmem = sdata;
    vsmem[tid] += vsmem[tid + 32];
    vsmem[tid] += vsmem[tid + 16];
    vsmem[tid] += vsmem[tid + 8];
    vsmem[tid] += vsmem[tid + 4];
    vsmem[tid] += vsmem[tid + 2];
    vsmem[tid] += vsmem[tid + 1];
  }

  if (tid == 0)
    S[blockIdx.x] = sdata[0];  // each block has its sum of threads within
};

// template <unsigned int GRID_X, unsigned int BLOCK_X>
__global__ void parent(float *A, float *B, int N, int GRID_X, int BLOCK_X) {
  int p1 = N / 2 * N + N / 2, p2 = 37 * N + 47;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    for (int i = 0; i < 5; ++i) {
      update<<<GRID_X, BLOCK_X>>>(A, B, N);
      // __syncthreads();
      update<<<GRID_X, BLOCK_X>>>(B, A, N);
      // __syncthreads();
    }

    // store results to B
    B[p1] = A[p1];
    B[p2] = A[p2];

    for (int numToSum = N * N, numBlock; numToSum > 1; numToSum = numBlock) {
      numBlock = (numToSum + BLOCK_X - 1) / BLOCK_X;
      reduceSmemDyn<<<numBlock, BLOCK_X, BLOCK_X * sizeof(float)>>>(A, A,
                                                                    numToSum);
      // __syncthreads();
    }
  }
}

void matrix_update(int N, int BLOCK_X = 128) {
  int NN{N * N};
  size_t nBytes = NN * sizeof(float);
  float *A = (float *)malloc(nBytes);
  float *B = (float *)malloc(nBytes);
  float res[3] = {0, 0, 0};

  int p1 = N / 2 * N + N / 2, p2 = 37 * N + 47;
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

  parent<<<1, block.x>>>(d_A, d_B, N, grid.x, block.x);

  // stop the timer
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float millisecond = 0;
  cudaEventElapsedTime(&millisecond, start, stop);

  cudaMemcpy(&res[0], &d_A[0], sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&res[1], &d_B[p1], sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&res[2], &d_B[p2], sizeof(float), cudaMemcpyDeviceToHost);

  /* end timing */
  cout << "grid " << grid.x << " block " << block.x << " calculation time "
       << millisecond << " sum = " << res[0] << " A[N / 2][N / 2] " << res[1]
       << " A[37][47] " << res[2] << endl;

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
