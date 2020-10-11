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


#ifndef __SYNCTHREADS
void __syncthreads();
#define __SYNCTHREADS() __syncthreads()
#endif


#ifndef SUM_ARRAY
float sum_array(float* x, int N) {
    float a = 0;
    for (int i = 0; i < N; i++)
        a += x[i];
    return a;
}

// Finds sum of array elements.
// SUM_ARRAY({1, 2, 3}, 2) = 6
#define SUM_ARRAY(x, N) sum_array(x, N)
#endif


#ifndef PRINTVEC
inline void printvec(float* x, int N) {
    printf("{");
    for (int i = 0; i < N - 1; i++)
        printf("%.1f, ", x[i]);
    if (N > 0) printf("%.1f", x[N - 1]);
    printf("}");
}

// Prints a vector.
// PRINTVEC(x, 3) = {1, 2, 3}
#define PRINTVEC(x, N) printvec(x, N)
#endif


#ifndef SUM_SQUARES
inline int sum_squares(int x) {
    return x * (x + 1) * (2 * x + 1) / 6;
}

// Computes sum of squares of natural numbers.
// SUM_SQUARES(3) = 1^2 + 2^2 + 3^2 = 14
#define SUM_SQUARES(x) sum_squares(x)
#endif


#ifndef CEILDIV
inline int ceildiv(int x, int y) {
    return (x + y - 1) / y;
}

// Computes rounded-up integer division.
// CEILDIV(6, 3) = 2
// CEILDIV(7, 3) = 3
#define CEILDIV(x, y) ceildiv(x, y)
#endif


#ifndef MAX
// Finds maximum value.
// MAX(2, 3) = 3
#define MAX(x, y) ((x) > (y)? (x) : (y))
#endif

#ifndef MIN
// Finds minimum value.
// MIN(2, 3) = 2
#define MIN(x, y) ((x) < (y)? (x) : (y))
#endif


#ifndef UINT
typedef unsigned int uint;
#define UINT uint
#endif

#ifndef UINT8
typedef unsigned char uint8;
#define UINT8 uint8
#endif
