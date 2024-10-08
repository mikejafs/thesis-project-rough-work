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
void matrix_init(int* a, int n){
    for (int i = 0; i < n; i ++){
        a[i] = rand() % 100;
        // a[i] = i;
    }
}

//kernel function implimentation
int main(){
    int arr_size = 10;
    int* h_a; 
    int* h_b;
    int* d_a;
    int* d_b;

    //define bytes as the size of an int data type \times n
    size_t bytes = sizeof(int) * arr_size;
    
    //allocate memory on the host
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);

    //allocate memory on the device
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    
    //initialize the input array with random numbers between 0, 99
    matrix_init(h_a, arr_size);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    //thread and grid block sizes
    // int NUM_THREADS = 1 << 8;

    //better to keep the num_threads as a whole number like this for better readability
    //also best practice to keep NUM_THREADS as a whole number and let the 
    //NUM_BLOCKS increase/decrease to fit the problem
    //Finally note that this is really the number of threads per block
    int NUM_THREADS = 256;

    //Note that doing things this way apeals to the notion of letting the # blocks
    //increase to fit the problem size, with the added charactaristic of 
    //not being 0 if arr_size < NUM_THREADS, or having an extra thread block if 
    //arr_size == n * NUM_THREADS, where n is an integer
    int NUM_BLOCKS = (arr_size + NUM_THREADS - 1) / NUM_THREADS;

    //launch kernel
    array_copy<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, arr_size);

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
