#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#ifndef TRY_CUDA
inline void try_cuda(cudaError err, const char* exp, const char* func, int line, const char* file) {
  if (err == cudaSuccess) return;
  fprintf(stderr,
    "%s: %s\n"
    "  in expression %s\n"
    "  at %s:%d in %s\n",
    cudaGetErrorName(err), cudaGetErrorString(err),
    exp,
    func, line, file);
  exit(err);
}

// Prints an error message and exits, if CUDA expression fails.
// TRY_CUDA( cudaDeviceSynchronize() );
#define TRY_CUDA(exp) try_cuda(exp, #exp, __func__, __LINE__, __FILE__)
#endif

#ifndef TRY
// Prints an error message and exits, if CUDA expression fails.
// TRY( cudaDeviceSynchronize() );
#define TRY(exp) TRY_CUDA(exp)
#endif


#ifndef CEILDIV
// Computes rounded-up integer division.
// ceildiv(6, 3) = 2
// ceildiv(7, 3) = 3
int ceildiv(int x, int y) {
  return (x + y-1) / y;
}

// Computes rounded-up integer division.
// CEILDIV(6, 3) = 2
// CEILDIV(7, 3) = 3
#define CEILDIV(x, y) ceildiv(x, y)
#endif


#ifndef PRINTVEC
// Prints a vector.
// printvec(x, 3) = {1, 2, 3}
inline void printvec(int *x, int N) {
  printf("{");
  for (int i=0; i<N-1; i++)
    printf("%d, ", x[i]);
  if (N>0) printf("%d", x[N-1]);
  printf("}");
}

// Prints a vector.
// PRINTVEC(x, 3) = {1, 2, 3}
#define PRINTVEC(x, N) printvec(x, N)
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y)? (x) : (y))
#endif

#ifndef MIN
#define MIN(x, y) ((x) < (y)? (x) : (y))
#endif
