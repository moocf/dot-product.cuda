#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h>
#include "support.h"


const int threads = 4;


// Each thread computes pairwise product of multiple components of vector.
// Since there are 10 components, but only a maximum of 4 total threads,
// each thread pairwise product of its component, and shifts by a stride
// of the total number of threads. This is done as long as it does not
// exceed the length of the vector. Each thread maintains the sum of the
// pairwise products it calculates.
// 
// Once pairiwise product calculation completes, the per-thread sum is
// stored in a cache, and then all threads in a block sync up to calculate
// the sum for the entire block in a binary tree fashion (in log N steps).
// The overall sum of each block is then stored in an array, which holds
// this partial sum. This partial sum is completed on the CPU. Hence, our
// dot product is complete.
// 
// 1. Compute sum of pairwise product at respective index, while within bounds.
// 2. Shift to the next component, by a stride of total no. of threads (4).
// 3. Store per-thread sum in shared cache (for further reduction).
// 4. Wait for all threads within the block to finish.
// 5. Reduce the sum in the cache to a single value in binary tree fashion.
// 6. Store this per-block sum into a partial sum array.
__global__ void kernel(float *c, float *a, float *b, int N) {
  __shared__ float cache[threads];
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int t = threadIdx.x, sum = 0;
  
  while (i < N) {                // 1
    sum += a[i] * b[i];          // 1
    i += blockDim.x * gridDim.x; // 2
  }
  cache[t] = sum; // 3

  __syncthreads(); // 4
  int T = blockDim.x / 2;                // 5
  while (T != 0) {                       // 5
    if (t < T) cache[t] += cache[t + T]; // 5
    __syncthreads();                     // 5
    T /= 2;                              // 5
  }

  if (t == 0) c[blockIdx.x] = cache[0]; // 6
}


// 1. Allocate space for 2 vectors A, B (of length 10).
// 2. Define vectors A and B.
// 3. Allocate space for partial sum C (of length "blocks").
// 4. Copy A, B from host memory to device memory (GPU).
// 5. Execute kernel with 2 threads per block, and max. 2 blocks (2*2 = 4).
// 6. Wait for kernel to complete, and copy partial sum C from device to host memory.
// 7. Reduce the partial sum C to a single value, the dot product (on CPU).
// 8. Validate if the dot product is correct (on CPU).
int main() {
  int N = 10;                     // 1
  size_t NB = N * sizeof(float);  // 1
  float *a = (float*) malloc(NB); // 1
  float *b = (float*) malloc(NB); // 1
  for (int i=0; i<N; i++) { // 2
    a[i] = (float) 2*i;     // 2
    b[i] = (float) i;       // 2
  }                         // 2

  int threads = 2;                             // 3
  int blocks = MAX(CEILDIV(N, threads), 2);    // 3
  size_t NC = blocks * sizeof(float);          // 3
  float* cpartial = (float*) malloc(NC);       // 3

  float *aD, *bD, *cpartialD;        // 4
  TRY( cudaMalloc(&aD, NB) );        // 4
  TRY( cudaMalloc(&bD, NB) );        // 4
  TRY( cudaMalloc(&cpartialD, NC) ); // 4
  TRY( cudaMemcpy(aD, a, NB, cudaMemcpyHostToDevice) ); // 4
  TRY( cudaMemcpy(bD, b, NB, cudaMemcpyHostToDevice) ); // 4

  kernel<<<blocks, threads>>>(cpartialD, aD, bD, N); // 5

  TRY( cudaMemcpy(cpartial, cpartialD, NC, cudaMemcpyDeviceToHost) ); // 6
  float c = SUM_ARRAY(cpartial, blocks); // 7

  printf("a = "); PRINTVEC(a, N); printf("\n");
  printf("b = "); PRINTVEC(b, N); printf("\n");
  printf("a .* b = %.1f\n", c);

  float cexpected = (float) 2 * SUM_SQUARES(N-1); // 8
  if (c != cexpected) {                           // 8
    fprintf(stderr, "ERROR: a .* b != %.1f\n", cexpected);
  }
  return 0;
}
