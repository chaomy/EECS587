#include <iostream>

__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

int main(void) {
	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	// Allocate space for device
	cudaMalloc((void **) &d_a, size);
	cudaMalloc((void **) &d_b, size);
	cudaMalloc((void **) &d_c, size);

	// setup input values
	a = 2, b = 7;

	// copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add<<<1, 1>>> (d_a, d_b, d_c);
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	cout<< "result is "<<c;
	// Cleanup
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}
