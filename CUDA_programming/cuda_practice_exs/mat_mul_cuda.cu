
// This program computes a simple version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>


using std::cout;
using std::generate;
using std::vector;

__global__ void mat_mul(const int* a, const int *b, int *c, int N){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[row * N + col] = 0;
    for (int k = 0; k < N; k++){
        c[row * N + col] += a[row * N + k] * b[k * N + col];
    }
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N){

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            int temp = 0;
            for (int k = 0; k < N; k++){
                temp += a[i*N + k] * b[k*N + j];
            }

            assert(temp == c[i * N + j]);
        }
    }
}

int main(){

    int N = 1 << 10;  //bit shift operater (1<<n is equiv to 2^n)
    size_t bytes = N * N * sizeof(int);

    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    //Initialize matrices
    generate(h_a.begin(), h_a.end(), []() {return rand() % 100;});
    generate(h_b.begin(), h_b.end(), []() {return rand() % 100;});

    //Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    //Copy data from host to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int THREADS = 32;
    int BLOCKS = N / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    mat_mul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, N);

    cout << "COMPLETED SUCCESSFULLY \n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}