#include <iostream>

// A simple CUDA kernel that prints a message
__global__ void helloWorldKernel() {
    printf("Hello, World from GPU!\n");
}

int main() {
    // Launch the kernel with one block and one thread
    helloWorldKernel<<<1, 1>>>();
    
    // Wait for the GPU to finish executing the kernel
    cudaDeviceSynchronize();
    
    std::cout << "Hello, World from CPU!" << std::endl;

    return 0;
}
