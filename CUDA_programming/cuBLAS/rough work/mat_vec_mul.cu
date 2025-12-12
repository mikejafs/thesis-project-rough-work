// nvcc -shared -o mat_vec_mul.so -Xcompiler -fPIC mat_vec_mul.cu -lcublas
// This code is a CUDA C++ implementation of a matrix-vector multiplication using cuBLAS.

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{
    void mat_vec_mul(
        float *A,
        float *x,
        float *y,
        int n
    ){
        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;

        // Perform the matrix-vector multiplication
        cublasSgemv(handle, 
            CUBLAS_OP_T, n, n, 
            &alpha, 
            A, n, 
            x, 1, 
            &beta, 
            y, 1
        );

        cublasDestroy(handle);
    }
}

