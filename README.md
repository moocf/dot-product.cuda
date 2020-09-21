The scalar product of two vectors is called dot product.

```c
Each thread computes pairwise product of multiple components of vector.
Since there are 10 components, but only a maximum of 4 total threads,
each thread pairwise product of its component, and shifts by a stride
of the total number of threads. This is done as long as it does not
exceed the length of the vector. Each thread maintains the sum of the
pairwise products it calculates.

Once pairiwise product calculation completes, the per-thread sum is
stored in a cache, and then all threads in a block sync up to calculate
the sum for the entire block in a binary tree fashion (in log N steps).
The overall sum of each block is then stored in an array, which holds
this partial sum. This partial sum is completed on the CPU. Hence, our
dot product is complete.
```

```
kernel():
1. Compute sum of pairwise product at respective index, while within bounds.
2. Shift to the next component, by a stride of total no. of threads (4).
3. Store per-thread sum in shared cache (for further reduction).
4. Wait for all threads within the block to finish.
5. Reduce the sum in the cache to a single value in binary tree fashion.
6. Store this per-block sum into a partial sum array.
```

```c
main():
1. Allocate space for 2 vectors A, B (of length 10).
2. Define vectors A and B.
3. Allocate space for partial sum C (of length "blocks").
4. Copy A, B from host memory to device memory (GPU).
5. Execute kernel with 2 threads per block, and max. 2 blocks (2*2 = 4).
6. Wait for kernel to complete, and copy partial sum C from device to host memory.
7. Reduce the partial sum C to a single value, the dot product (on CPU).
8. Validate if the dot product is correct (on CPU).
```

```bash
# OUTPUT
a = {0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0}
b = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
a .* b = 570.0
```

See [main.cu] for code.

[main.cu]: main.cu


### references

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
