#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
Same as array_copy1.0, but cleaned up to contain the better
versions of the practices mentioned throughout that file.

For a better description of what each line is doing, see the file
array_copy1.0.cu
*/

__global__ void ArrayCopy(int* a, int* b, int arr_size)
{
    //a -> input array
    //b -> output array
    
    //establish thread id in terms of 1D thread blocks
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < arr_size){
        b[tid] = a[tid];
    }
}

void VecInit(int* a, int n){
    //Random array initialization function. Returns random entries 
    //btwn 0 and 100 
    for (int i = 0; i < n; i ++){
        a[i] = rand() % 100;
    }
}

//Initialize the kernel function
int main(){
    int arr_size = 10;
    int* h_a;
    int* h_b;
    int* d_a; 
    int* d_b;

    size_t bytes = sizeof(int) * arr_size;

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    VecInit(h_a, arr_size);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (arr_size + NUM_THREADS - 1) / NUM_THREADS;

    ArrayCopy<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, arr_size);
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    //print out the copied data
    for (int i = 0; i < arr_size; i++){
        printf("%d", h_b[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;    
}



