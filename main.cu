#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h>
#include "support.h"


// Each thread can compute the sum of multiple components of vectors. Since
// there are 10 components, but only a maximum of 4 total threads, each thread
// computer the sum of its respective component, and shifts by a stride of the
// total number of vectors. This is done as long as it does not exceed the
// length of the vectors.
// 
// 1. Compute sum at respective index, while within bounds.
// 2. Shift to the next component, by a stride of total no. of threads (4).
// 
// threadIdx.x: thread index, within block (0 ... 1)
// blockIdx.x:  block index, within grid (0 ... 1)
// blockDim.x:  number of threads in a block (2)
// i: index into the vectors
__global__ void kernel(int *c, int *a, int *b, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // 1
  while (i < N) {                                // 1
    c[i] = a[i] + b[i];                          // 1
    i += gridDim.x * blockDim.x; // 2
  }
}


// 1. Allocate space for 3 vectors A, B, and C (of length 10).
// 2. Define vectors A and B (C = A + B will be computed by GPU).
// 3. Allocate space for A, B, C on GPU.
// 4. Copy A, B from host memory to device memory (GPU).
// 5. Execute kernel with 2 threads per block, and max. 2 blocks (2*2 = 4).
// 6. Wait for kernel to complete, and copy C from device to host memory.
// 7. Validate if the vector sum is correct (on CPU).
int main() {
  int N = 10;                  // 1
  size_t NB = N * sizeof(int); // 1
  int *a = (int*) malloc(NB);  // 1
  int *b = (int*) malloc(NB);  // 1
  int *c = (int*) malloc(NB);  // 1
  for (int i=0; i<N; i++) { // 2
    a[i] = 2*i;             // 2
    b[i] = -i;              // 2
  }                         // 2

  int *aD, *bD, *cD;          // 3
  TRY( cudaMalloc(&aD, NB) ); // 3
  TRY( cudaMalloc(&bD, NB) ); // 3
  TRY( cudaMalloc(&cD, NB) ); // 3
  TRY( cudaMemcpy(aD, a, NB, cudaMemcpyHostToDevice) ); // 4
  TRY( cudaMemcpy(bD, b, NB, cudaMemcpyHostToDevice) ); // 4

  int threads = 2;                            // 5
  int blocks  = MAX(CEILDIV(N, threads), 2);  // 5
  kernel<<<blocks, threads>>>(cD, aD, bD, N); // 5

  TRY( cudaMemcpy(c, cD, NB, cudaMemcpyDeviceToHost) ); // 6
  printf("a = "); PRINTVEC(a, N); printf("\n");
  printf("b = "); PRINTVEC(b, N); printf("\n");
  printf("c = "); PRINTVEC(c, N); printf("\n");

  for (int i=0; i<N; i++) {  // 7
    if (c[i] == i) continue; // 7
    fprintf(stderr, "%d + %d != %d (at component %d)\n", a[i], b[i], c[i], i);
  }
  return 0;
}
