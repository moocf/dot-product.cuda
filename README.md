The scalar product of two vectors is called dot product.

```c
Each thread can compute the sum of multiple components of vectors. Since
there are 10 components, but only a maximum of 4 total threads, each thread
computer the sum of its respective component, and shifts by a stride of the
total number of vectors. This is done as long as it does not exceed the
length of the vectors.

1. Compute sum at respective index, while within bounds.
2. Shift to the next component, by a stride of total no. of threads (4).

threadIdx.x: thread index, within block (0 ... 1)
blockIdx.x:  block index, within grid (0 ... 1)
blockDim.x:  number of threads in a block (2)
i: index into the vectors
```

```c
1. Allocate space for 3 vectors A, B, and C (of length 10).
2. Define vectors A and B (C = A + B will be computed by GPU).
3. Allocate space for A, B, C on GPU.
4. Copy A, B from host memory to device memory (GPU).
5. Execute kernel with 2 threads per block, and max. 2 blocks (2*2 = 4).
6. Wait for kernel to complete, and copy C from device to host memory.
7. Validate if the vector sum is correct (on CPU).
```

```bash
# OUTPUT
a = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
b = {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}
c = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

See [main.cu] for code.

[main.cu]: main.cu


### references

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
