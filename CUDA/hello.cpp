#include <stdio.h>
#include <iostream>

using std::cout;
using std::endl;

__global__ void mykernel(void) {}

int main() {
  mykernel<<<1, 1>>>();
  cout << "hello world" << endl;
  return 0;
}