#include <cmath>
#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

__global__ void nestedHelloWorld(int const iSize, int iDepth) {
  int tid = threadIdx.x;
  printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid,
         blockIdx.x);
  // condition to stop recursive execution
  if (iSize == 1) return;
  // reduce block size to half
  int nthreads = iSize >> 1;
  // thread 0 launches child grid recursively
  if (tid == 0 && nthreads > 0) {
    nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
    printf("-------> nested execution depth: %d\n", iDepth);
  }
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);        // problem size
  int BLOCK_X = atoi(argv[2]);  // block size
  nestedHelloWorld<<<1, 8>>>(8, 1);
}