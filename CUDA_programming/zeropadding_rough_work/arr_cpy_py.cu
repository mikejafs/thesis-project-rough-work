/*
Array copy file prepared for running and calling
within python
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C"{

        __global__ void ArrayCopy(int* a, int* b, int arr_size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < arr_size) {
            b[tid] = a[tid];
        }
    }

    void run_array_copy(int* a, int* b, int arr_size) {
        int *d_a, *d_b;
        size_t bytes = arr_size * sizeof(int);

        // Allocate memory on the device
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);

        // Copy the data to the device
        cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

        // Launch the kernel
        int NUM_THREADS = 256;
        int NUM_BLOCKS = (arr_size + NUM_THREADS - 1) / NUM_THREADS;
        ArrayCopy<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, arr_size);

        // Copy the result back to host
        cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_a);
        cudaFree(d_b);
    }
}
