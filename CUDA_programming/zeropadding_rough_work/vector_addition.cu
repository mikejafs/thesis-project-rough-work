#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// Cuda kernel for vector addition
// __global__means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(int* a, int* b, int* c, int n){

    //calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    //vector boundary grid
    if (tid < n){
        //Each thread adds a single element
        c[tid] = a[tid] + b[tid];
    }
}


//initialize matrices with random entries
void matrix_init(int* a, int n){
    for (int i = 0; i < n; i ++){
        a[i] = rand() % 100;
    }
}


int main()
{
    //Array of size 2^16
    int n = 1 << 16;
    // Host vector pointers
    int *h_a, *h_b, *h_c;
    //Device vector pointers
    int *d_a, *d_b, *d_c;
    // Allocation size for all vectors
    size_t bytes = sizeof(int) * n;

    //Allocate host memory
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    //Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //Initialize vectors a and b with random values between 0 and 99
    matrix_init(h_a, n);
    matrix_init(h_b, n);

    // Copy data from
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Threadblock size (2^8 = 256)
    int NUM_THREADS = 256;

    // Grid size
    int NUM_BLOCKS = (int)ceil(n / NUM_THREADS);

    //launch kernel on default stream w/o shmem
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, n);

    // Copy sum vector from decive to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    //Check result for errors
    // error_check(h_a, h_b, h_c, n);

    printf("COMPLETED SUCCESFULLY");

    // printf("%d", h_c);

    return 0;
}