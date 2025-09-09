# cupy_buckets_demo.py
import os, ctypes, numpy as np, cupy as cp

# --- load the shared library ---
lib = ctypes.CDLL(os.path.abspath("./libbucketed_gemm.so"))
lib.sgemm_strided_batched_same_shape.argtypes = [
    ctypes.c_void_p,  # A base
    ctypes.c_void_p,  # B base
    ctypes.c_void_p,  # C base
    ctypes.c_int, ctypes.c_int, ctypes.c_int,        # m, n, k
    ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,  # strideA, strideB, strideC (elements)
    ctypes.c_int,     # batchCount
    ctypes.c_float, ctypes.c_float,                  # alpha, beta
    ctypes.c_int      # use_tf32
]
lib.sgemm_strided_batched_same_shape.restype = ctypes.c_int

def sgemm_strided_batched(A_base_ptr, B_base_ptr, C_base_ptr,
                          m, n, k, strideA, strideB, strideC,
                          batch, alpha=1.0, beta=0.0, use_tf32=0):
    err = lib.sgemm_strided_batched_same_shape(
        ctypes.c_void_p(A_base_ptr),
        ctypes.c_void_p(B_base_ptr),
        ctypes.c_void_p(C_base_ptr),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
        ctypes.c_longlong(strideA),
        ctypes.c_longlong(strideB),
        ctypes.c_longlong(strideC),
        ctypes.c_int(batch),
        ctypes.c_float(alpha), ctypes.c_float(beta),
        ctypes.c_int(use_tf32))
    if err != 0:
        raise RuntimeError(f"sgemm_strided_batched failed with error code {err}")

# --- problem generator ---
rng = np.random.default_rng(0)

# Fixed inner dimension (your width) and RHS columns
k = 4          # e.g., 3–5 in your case
n = 1          # set >1 if you multiply by multiple RHS columns at once

# Choose some heights (buckets) and counts per bucket
heights = [96, 128, 192, 256]       # variable m_i
counts  = [700, 500, 400, 300]      # blocks per height

# Shared right-hand matrix B (k x n) for ALL blocks:
B = cp.asfortranarray(cp.random.standard_normal((k, n), dtype=cp.float32))

# Create per-bucket lists of A_i and output placeholders C_i
buckets = {}
for m, cnt in zip(heights, counts):
    A_list = [cp.asfortranarray(cp.random.standard_normal((m, k), dtype=cp.float32))
              for _ in range(cnt)]
    C_list = [cp.empty((m, n), dtype=cp.float32, order='F') for _ in range(cnt)]
    buckets[m] = (A_list, C_list)

# --- run: pack -> call cuBLAS (one call per bucket) -> unpack (optional) ---
use_tf32 = 1   # set 0 for strict FP32; 1 to enable TF32 on Ampere+ (usually faster)

for m, (A_list, C_list) in buckets.items():
    batch = len(A_list)
    # Strides in ELEMENTS for column-major blocks
    strideA = m * k
    strideB = 0          # shared B optimization: reuse same B for all problems
    strideC = m * n

    # Allocate strided packs (column-major 3D tensors: (m, k, batch), (m, n, batch))
    A_pack = cp.empty((m, k, batch), dtype=cp.float32, order='F')
    C_pack = cp.empty((m, n, batch), dtype=cp.float32, order='F')  # uninitialized unless beta!=0

    # Fill A_pack (device-to-device copies through CuPy assignment)
    # If beta != 0, prefill C_pack with existing C_i
    for i in range(batch):
        A_pack[:, :, i] = A_list[i]
        # If you want beta != 0, do: C_pack[:, :, i] = C_list[i]
        # For beta==0, we can leave C_pack uninitialized.

    # Call cuBLAS (one call does ALL GEMMs for this m)
    sgemm_strided_batched(
        A_pack.data.ptr,
        B.data.ptr,
        C_pack.data.ptr,
        m, n, k,
        strideA, strideB, strideC,
        batch,
        alpha=1.0, beta=0.0,
        use_tf32=use_tf32
    )

    # Unpack outputs if you need them back in individual arrays
    for i in range(batch):
        C_list[i][...] = C_pack[:, :, i]

# --- (optional) correctness spot-check vs CuPy matmul ---
max_abs_err = 0.0
for m, (A_list, C_list) in buckets.items():
    for i in range(3):  # spot-check a few per bucket
        Ai = A_list[i]
        Ci_ref = Ai @ B  # CuPy reference (column-major aware)
        max_abs_err = max(max_abs_err, float(cp.max(cp.abs(C_list[i] - Ci_ref)).get()))
print(f"max abs error (spot-checks): {max_abs_err:.3e}")

