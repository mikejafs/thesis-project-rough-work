// block_diag_gemm.cu
// nvcc -std=c++14 -lcublasLt -shared -fPIC -o libblockdiaggemm.so block_diag_gemm.cu

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <vector>
#include <stdexcept>

// Helper to throw on error
inline void _chk(cublasStatus_t s, const char* m) {
    if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(m);
}
inline void _chk(cudaError_t e,    const char* m) {
    if(e != cudaSuccess)        throw std::runtime_error(m);
}

extern "C" void block_diag_gemm_cublaslt(
    const void* const* A_ptrs,  // [batchCount]: ptr to each M_i block (m_i×m_i, col‑major)
    const int*          starts,  // [batchCount]: inclusive row start of each block in [0…N)
    const int*          stops,   // [batchCount]: exclusive row end
    const void*         X,       // full X: N×P, col‑major
    void*               Y,       // full Y: N×P, col‑major
    int                 N,       // X,Y rows
    int                 P,       // X,Y cols
    float               alpha,
    float               beta,
    int                 batchCount)
{
    // 1) create cuBLASLt handle
    cublasLtHandle_t lt;
    _chk(cublasLtCreate(&lt), "cublasLtCreate failed");

    // 2) descriptors & layouts vectors
    std::vector<cublasLtMatmulDesc_t>   descs(batchCount);
    std::vector<cublasLtMatrixLayout_t> layoutsA(batchCount),
                                        layoutsB(batchCount),
                                        layoutsC(batchCount);

    for(int i = 0; i < batchCount; ++i) {
        int m = stops[i] - starts[i];

        // 2.1) descriptor: FP32 compute, FP32 data
        _chk(cublasLtMatmulDescCreate(&descs[i],
               CUBLAS_COMPUTE_32F, CUDA_R_32F),
             "MatmulDescCreate");  // :contentReference[oaicite:0]{index=0}

        // 2.2) no-transpose attributes
        cublasOperation_t opN = CUBLAS_OP_N;
        _chk(cublasLtMatmulDescSetAttribute(
               descs[i],
               CUBLASLT_MATMUL_DESC_TRANSA,
               &opN, sizeof(opN)),
             "SetAttr TRANSA");
        _chk(cublasLtMatmulDescSetAttribute(
               descs[i],
               CUBLASLT_MATMUL_DESC_TRANSB,
               &opN, sizeof(opN)),
             "SetAttr TRANSB");

        // 2.3) layouts: A_i is m×m (lda=m), B_i is m×P (ldb=N), C_i is m×P (ldc=N)
        _chk(cublasLtMatrixLayoutCreate(
               &layoutsA[i], CUDA_R_32F, m,   m, m),
             "Layout A");
        _chk(cublasLtMatrixLayoutCreate(
               &layoutsB[i], CUDA_R_32F, m,   P, N),
             "Layout B");
        _chk(cublasLtMatrixLayoutCreate(
               &layoutsC[i], CUDA_R_32F, m,   P, N),
             "Layout C");
    }

    // 3) create preference and set max workspace bytes
    cublasLtMatmulPreference_t pref;
    _chk(cublasLtMatmulPreferenceCreate(&pref),
         "PrefCreate");  
    // first, figure out needed workspace per-block
    size_t maxWorkspace = 0;
    {
      cublasLtMatmulHeuristicResult_t h;
      int                             cnt;
      for(int i = 0; i < batchCount; ++i) {
        _chk(cublasLtMatmulAlgoGetHeuristic(
               lt,
               descs[i],
               layoutsA[i],
               layoutsB[i],
               layoutsC[i],
               layoutsC[i],
               pref,
               /*requested=*/1,
               &h,
               &cnt),
             "AlgoGetHeuristic");  // :contentReference[oaicite:1]{index=1}
        maxWorkspace = std::max(maxWorkspace, h.workspaceSize);
      }
    }
    // then set preference to allow up to that much workspace:
    _chk(cublasLtMatmulPreferenceSetAttribute(
           pref,
           CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
           &maxWorkspace, sizeof(maxWorkspace)),
         "PrefSetAttr");  // :contentReference[oaicite:2]{index=2}

    // 4) allocate one workspace buffer, aligned by cudaMalloc
    void* workspace = nullptr;
    _chk(cudaMalloc(&workspace, maxWorkspace),
         "cudaMalloc(workspace)");

    // 5) for each block: pick algo + launch matmul
    for(int i = 0; i < batchCount; ++i) {
        int m = stops[i] - starts[i];
        // slice pointers into X,Y (col‑major, so just advance by starts[i] rows)
        const void* Bs = (const char*)X + starts[i] * sizeof(float);
        void*       Cs = (      char*)Y + starts[i] * sizeof(float);

        // 5.1) heuristic again (now with full pref)
        cublasLtMatmulHeuristicResult_t heuristic;
        int                             cnt;
        _chk(cublasLtMatmulAlgoGetHeuristic(
               lt,
               descs[i],
               layoutsA[i],
               layoutsB[i],
               layoutsC[i],
               layoutsC[i],
               pref,
               1, &heuristic, &cnt),
             "AlgoGetHeuristic(2)");
        if(cnt == 0) throw std::runtime_error("no algo found");

        // 5.2) actual matmul launch
        _chk(cublasLtMatmul(
               lt,
               descs[i],
               &alpha,
               A_ptrs[i], layoutsA[i],
               Bs,         layoutsB[i],
               &beta,
               Cs,         layoutsC[i],
               Cs,         layoutsC[i],
               &heuristic.algo,
               &heuristic, 1,
               0 /*stream*/  // use default stream
             ), "cublasLtMatmul");  // :contentReference[oaicite:3]{index=3}
    }

    // 6) clean up
    cudaFree(workspace);
    for(int i = 0; i < batchCount; ++i) {
        cublasLtMatmulDescDestroy(   descs[i]);
        cublasLtMatrixLayoutDestroy(layoutsA[i]);
        cublasLtMatrixLayoutDestroy(layoutsB[i]);
        cublasLtMatrixLayoutDestroy(layoutsC[i]);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtDestroy(lt);
}
