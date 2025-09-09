# edges_grouped_demo.py
import os, ctypes as ct, numpy as np, cupy as cp
from cupy.cuda import cublas

# -------------------- load library + create cuBLAS handle --------------------
lib = ct.cdll.LoadLibrary(os.path.abspath("./libgrouped_gemm.so"))
handle = cublas.create()

lib.sgemm_grouped_batched.restype = ct.c_int
c_int_p   = ct.POINTER(ct.c_int)
c_float_p = ct.POINTER(ct.c_float)
lib.sgemm_grouped_batched.argtypes = [
    ct.c_void_p,                 # cublasHandle_t
    ct.c_int,                    # group_count
    c_int_p, c_int_p, c_int_p,   # m[], n[], k[]
    c_int_p, c_int_p, c_int_p,   # lda[], ldb[], ldc[]
    c_float_p, c_float_p,        # alpha[], beta[]
    c_int_p,                     # group_size[]
    ct.c_void_p, ct.c_void_p,    # Aarray_dev_flat, Barray_dev_flat (device pointers)
    ct.c_void_p                  # Carray_dev_flat (device pointer)
]

# -------------------- problem spec (you provide edges) -----------------------
rng = np.random.default_rng(0)

# Dimensions:
k = 32    # fixed inner dimension (your "width")
n = 8     # number of RHS columns (set 1 if vector RHS)

# Build a parent A (column-major / Fortran) and a shared B (also Fortran).
# IMPORTANT: column-major so lda = number of rows of A_parent.
M_parent = 4096
A_parent = cp.asfortranarray(cp.asarray(rng.standard_normal((M_parent, k), dtype=np.float32)))
B_shared = cp.asfortranarray(cp.asarray(rng.standard_normal((k, n), dtype=np.float32)))

# Your edges array: start/stop rows for each block (length B+1; strictly increasing; last <= M_parent).
# --- Build demo edges robustly (skip this whole block if you already have edges) ---
num_blocks = 1200
heights_host = rng.integers(16, 64, size=num_blocks, dtype=np.int32)   # B lengths
starts_host  = np.r_[0, np.cumsum(heights_host[:-1])].astype(np.int32)  # length B
ends_host    = (starts_host + heights_host).astype(np.int32)            # length B

# keep only blocks that fit within M_parent
mask = ends_host <= M_parent
heights_host = heights_host[mask]
starts_host  = starts_host[mask]
ends_host    = ends_host[mask]
B = int(mask.sum())

# construct edges of length B+1
edges_host = np.empty(B + 1, dtype=np.int32)
edges_host[:-1] = starts_host
edges_host[-1]  = ends_host[-1]

# If you already have edges_host, just use it:
# edges_host = your_edges  # np.ndarray, shape (B+1,), dtype=int

# Derive per-block heights m_i from edges
m_i = (edges_host[1:] - edges_host[:-1]).astype(np.int32)   # shape (B,)
B = int(m_i.size)

print(f"Total blocks: {B}, unique heights: {np.unique(m_i).size}, M_parent={M_parent}, k={k}, n={n}")

# -------------------- group by height and build device pointer arrays --------
# Map blocks to groups of identical height (no packing/copies — zero-copy views)
unique_m, inverse = np.unique(m_i, return_inverse=True)     # unique heights + group id per block
G = int(unique_m.size)                                      # number of groups
group_size = np.bincount(inverse, minlength=G).astype(np.int32)

# Build flat arrays of device pointers for A, B, C in group order.
A_ptr_chunks = []
B_ptr_chunks = []
C_ptr_chunks = []
C_objs_groups = []     # keep references so device memory stays alive

for g in range(G):
    m = int(unique_m[g])
    idxs = np.where(inverse == g)[0]     # block indices in this group
    # Pointers to the FIRST element of each submatrix A_i = A_parent[edges[i]:edges[i+1], :]
    # Column-major: offset = edges[i] * sizeof(float) from base; lda = M_parent for ALL groups
    A_ptrs_host = [int(A_parent.data.ptr) + int(edges_host[i]) * A_parent.itemsize for i in idxs]

    # Outputs per block (column-major m x n)
    C_blocks = [cp.empty((m, n), dtype=cp.float32, order='F') for _ in idxs]
    C_ptrs_host = [int(C.data.ptr) for C in C_blocks]

    # RHS: use the SAME B for all problems (opt). Repeat its pointer.
    B_ptrs_host = [int(B_shared.data.ptr)] * len(idxs)

    # Push to device as arrays-of-pointers (cuBLAS expects these pointer arrays on device)
    A_ptr_chunks.append(cp.asarray(A_ptrs_host, dtype=cp.uintp))
    B_ptr_chunks.append(cp.asarray(B_ptrs_host, dtype=cp.uintp))
    C_ptr_chunks.append(cp.asarray(C_ptrs_host, dtype=cp.uintp))

    C_objs_groups.append(C_blocks)  # keep references for checking

# Concatenate per-group pointer arrays into ONE flat array per A/B/C (device)
A_flat = cp.concatenate(A_ptr_chunks)      # dtype=uintp, length = sum(group_size)
B_flat = cp.concatenate(B_ptr_chunks)
C_flat = cp.concatenate(C_ptr_chunks)
problem_count = int(A_flat.size)
assert problem_count == int(group_size.sum())

print("Flat pointer arrays built:",
      f"A_flat={A_flat.size}, B_flat={B_flat.size}, C_flat={C_flat.size},",
      f"sum(group_size)={group_size.sum()}")

# -------------------- per-group descriptors (host) ---------------------------
m_arr    = unique_m.astype(np.int32)
n_arr    = np.full(G, n, dtype=np.int32)
k_arr    = np.full(G, k, dtype=np.int32)
lda_arr  = np.full(G, A_parent.shape[0], dtype=np.int32)   # lda = rows of A_parent (M_parent)
ldb_arr  = np.full(G, k, dtype=np.int32)                   # B is k x n (col-major)
ldc_arr  = unique_m.astype(np.int32)                       # ldc = rows of C_i = m_i
alpha_arr= np.ones(G, dtype=np.float32)
beta_arr = np.zeros(G, dtype=np.float32)

# -------------------- ONE grouped GEMM call ---------------------------------
status = lib.sgemm_grouped_batched(
    ct.c_void_p(handle),
    ct.c_int(G),
    m_arr.ctypes.data_as(c_int_p),
    n_arr.ctypes.data_as(c_int_p),
    k_arr.ctypes.data_as(c_int_p),
    lda_arr.ctypes.data_as(c_int_p),
    ldb_arr.ctypes.data_as(c_int_p),
    ldc_arr.ctypes.data_as(c_int_p),
    alpha_arr.ctypes.data_as(c_float_p),
    beta_arr.ctypes.data_as(c_float_p),
    group_size.ctypes.data_as(c_int_p),
    ct.c_void_p(int(A_flat.data.ptr)),   # device pointer to flat A pointer array
    ct.c_void_p(int(B_flat.data.ptr)),   # device pointer to flat B pointer array
    ct.c_void_p(int(C_flat.data.ptr))    # device pointer to flat C pointer array
)
if status != 0:
    raise RuntimeError(f"grouped GEMM failed with status {status}")

# -------------------- sanity prints & correctness spot checks ----------------
print("\nShapes (first few blocks per first few groups):")
for g in range(min(G, 3)):
    m = int(unique_m[g])
    print(f"  Group {g}: m={m}, count={int(group_size[g])}, A_sub lda={A_parent.shape[0]}, C ldc={m}")

# Spot-check a couple results per group
max_err = 0.0
for g in range(min(G, 3)):
    m = int(unique_m[g])
    # sample up to two blocks from this group
    for j in range(min(2, int(group_size[g]))):
        C = C_objs_groups[g][j]
        # reconstruct a CuPy view of the corresponding A submatrix from pointer (no copy)
        # Safer (and simpler) correctness check: just slice from A_parent using edges
        # (CuPy will handle strides; we didn't modify A_parent.)
        # Find the global block index among all blocks with height m:
        idxs = np.where(inverse == g)[0]
        i_global = int(idxs[j])
        A_view = A_parent[edges_host[i_global]:edges_host[i_global+1], :]  # (m,k) view
        ref = A_view @ B_shared
        err = float(cp.max(cp.abs(C - ref)).get())
        print(f"  Group {g} sample {j}: C.shape={C.shape}, ref.shape={ref.shape}, err={err:.3e}")
        max_err = max(max_err, err)
print(f"\nMax abs error (spot checks): {max_err:.3e}")
