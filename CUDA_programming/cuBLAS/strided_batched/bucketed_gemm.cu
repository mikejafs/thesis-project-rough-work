// bucketed_gemm.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CUDA_OK(stmt) do {                             \
    cudaError_t _e = (stmt);                           \
    if (_e != cudaSuccess) {                           \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
                   cudaGetErrorString(_e), __FILE__, __LINE__); \
      return int(_e);                                  \
    }                                                  \
  } while(0)

#define CUBLAS_OK(stmt) do {                           \
    cublasStatus_t _s = (stmt);                        \
    if (_s != CUBLAS_STATUS_SUCCESS) {                 \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",\
                   int(_s), __FILE__, __LINE__);       \
      return int(_s);                                  \
    }                                                  \
  } while(0)

// Column-major, FP32
// Computes: for i in [0..batchCount-1],
//   C_i(m x n) = alpha * A_i(m x k) * B_i(k x n) + beta * C_i(m x n)
// A, B, C are provided in "strided batch" layout: base pointer + element stride.
// If B is shared across the whole batch, pass strideB = 0 and point B to the shared matrix.
// Returns 0 on success, otherwise CUDA/cuBLAS error code.
extern "C" int sgemm_strided_batched_same_shape(
    const float* A, const float* B, float* C,
    int m, int n, int k,
    long long strideA, long long strideB, long long strideC,
    int batchCount,
    float alpha, float beta,
    int use_tf32)  // 0 = off, 1 = enable TF32 (Ampere+)
{
  // Quick parameter sanity
  if (m <= 0 || n <= 0 || k <= 0 || batchCount <= 0) return -1;

  cublasHandle_t handle;
  CUBLAS_OK(cublasCreate(&handle));

  // Optional: run on the per-thread default stream (0); or expose a stream arg.
  // cublasSetStream(handle, 0);

  if (use_tf32) {
    // Best-effort: ignore if not supported (pre-Ampere still returns SUCCESS).
    CUBLAS_OK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  }

  const int lda = m;  // column-major
  const int ldb = k;
  const int ldc = m;

  // cuBLAS expects pointers and strides in ELEMENTS (not bytes).
  CUBLAS_OK(cublasSgemmStridedBatched(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      m, n, k,
      &alpha,
      A, lda, strideA,
      B, ldb, strideB,
      &beta,
      C, ldc, strideC,
      batchCount));

  CUBLAS_OK(cublasDestroy(handle));
  return 0;
}
