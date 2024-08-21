#include <iostream>

__global__ void helloWorld() {
    printf("Hello, World from GPU!\n");
}

int main() {
    // Launch kernel
    helloWorld<<<1, 1>>>();
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    return 0;
}

