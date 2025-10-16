//nvcc -Xcompiler -fPIC -shared -o libgemm.so gemm.cu -lcublas

#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{
    void matmul_gemm(
        float *A,
        float *B,
        float *C,
        int m, int n, int k
    ){
        cublasHandle_t handle;
        cublasCreate(&handle);

        // cublasStatus_t status;

        // Create the cuBLAS handle
        // status = cublasCreate(&handle);
        // if (status != CUBLAS_STATUS_SUCCESS) {
        //     return 1;
        // }


        const float alpha = 1.0f;
        const float beta = 0.0f;

        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            B, n,
            A, k,
            &beta,
            C, n
        );

        cublasDestroy(handle);

        // return (status == CUBLAS_STATUS_SUCCESS) ? 0 : 2;
    }
} 