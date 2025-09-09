// grouped_gemm.cu  (requires CUDA Toolkit >= 12.5 for cublasSgemmGroupedBatched)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstdio>

#define CHECK_CUDA(x)  do { auto e=(x); if(e!=cudaSuccess){ \
  std::fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  return int(e);} } while(0)
#define CHECK_CUBLAS(x) do { auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  std::fprintf(stderr,"cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, int(s)); \
  return int(s);} } while(0)

// One call → many groups of different (m,n,k).
// Aarray_dev_flat/Barray_dev_flat/Carray_dev_flat are device arrays of device pointers,
// each of length problem_count = sum(group_size[i]) across groups.
//
// All per-group arrays (sizes, leading dims, alphas/betas, group_size) are HOST arrays.
extern "C" int sgemm_grouped_batched(
    cublasHandle_t handle,
    int group_count,
    const int* m_array, const int* n_array, const int* k_array,       // [group_count]
    const int* lda_array, const int* ldb_array, const int* ldc_array,  // [group_count]
    const float* alpha_array, const float* beta_array,                 // [group_count]
    const int* group_size,                                             // [group_count]
    const float* const* Aarray_dev_flat,                               // [problem_count] (device)
    const float* const* Barray_dev_flat,                               // [problem_count] (device)
    float* const*       Carray_dev_flat)                               // [problem_count] (device)
{
  // Use host pointer mode for alpha/beta arrays.
  CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

  // All groups here use 'N','N'; make them inputs if you need per-group transposes.
  std::vector<cublasOperation_t> transA(group_count, CUBLAS_OP_N);
  std::vector<cublasOperation_t> transB(group_count, CUBLAS_OP_N);

  CHECK_CUBLAS(
    cublasSgemmGroupedBatched(
      handle,
      transA.data(), transB.data(),
      m_array, n_array, k_array,
      alpha_array,
      Aarray_dev_flat, lda_array,
      Barray_dev_flat, ldb_array,
      beta_array,
      Carray_dev_flat, ldc_array,
      group_count,
      group_size
    )
  );
  return 0;
}
