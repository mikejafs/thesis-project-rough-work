#include <stdio.h>
#include <stdlib.h> 
#include <math.h>


//outline of c function (cuda kernel)
//this is the place to assign a thread id
//we leave all variables as pointers to the variable address
//also need to use the __global__ call to define a cuda kernel
__global__ void array_copy(int* a, int* b, int arr_size)
{
    //a array: input array
    //b array: output array
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    //check boundary condition and cpy intries from a to b
    if (tid < arr_size){
        b[tid] = a[tid];
    }
}

//initialize arrays with random entries
//I'm still not too sure how defining a as a pointer here
//can lead to the code actually assigning values to a in the main()
//function....
void matrix_init(int *a, int n){
    for (int i = 0; i < n; i ++){
        a[i] = rand() % 100;
        // a[i] = i;
    }
}

//kernel function implimentation
int main(){
    int arr_size = 20.0;
    int h_a[arr_size], h_b[arr_size];
    int *d_a, *d_b;

    //define bytes as the size of an int data type \times n
    size_t bytes = sizeof(int) * arr_size;

    //initialize the input array with random numbers between 0, 99
    matrix_init(h_a, arr_size);

    //print input array
    printf("The input array is:\n");
    for (int i = 0; i < arr_size; i++){
        printf("%d ", h_a[i]);
    }

    printf("\n");

    //allocate memory on the device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    //Transfer data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    //thread and grid block sizes
    int NUM_THREADS = 1 << 8;
    // int NUM_THREADS = arr_size + 255;
    int NUM_BLOCKS = (arr_size + NUM_THREADS - 1) / NUM_THREADS;

    //launch kernel
    array_copy<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, arr_size);

    //Transfer data from device to host
    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    //print output array
    printf("The output array is: \n");
    for (int i = 0; i < arr_size; i++){
        printf("%d ", h_b[i]);
    }
    printf("\n");

    //free the memory on the device
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
