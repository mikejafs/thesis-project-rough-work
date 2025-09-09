# cupy_grouped_demo.py
import ctypes as ct, cupy as cp, numpy as np, os
from cupy.cuda import cublas

lib = ct.cdll.LoadLibrary(os.path.abspath("./libgrouped_gemm.so"))
handle = cublas.create()

# ---- build groups (varying m_i), fixed k,n ----
rng = np.random.default_rng(0)
G = 8
k, n = 64, 16
m_list  = [int(rng.integers(32, 512)) for _ in range(G)]
size_li = [int(rng.integers(50, 120)) for _ in range(G)]  # group_size[i]

B_shared = cp.asarray(rng.standard_normal((k, n), dtype=np.float32))
A_ptr_groups, B_ptr_groups, C_ptr_groups = [], [], []
C_objs_groups = []  # keep refs so memory stays alive

for gi, (m, gi_size) in enumerate(zip(m_list, size_li)):
    A_ptrs = []
    C_ptrs = []
    C_objs = []
    for _ in range(gi_size):
        Aij = cp.asarray(rng.standard_normal((m, k), dtype=np.float32), order='F')
        Cij = cp.zeros((m, n), dtype=cp.float32, order='F')

        # print(f"Group {gi}: Aij shape={Aij.shape}, Cij shape={Cij.shape}, B shape={B_shared.shape}")

        A_ptrs.append(Aij.data.ptr)
        C_ptrs.append(Cij.data.ptr)
        C_objs.append(Cij)
    B_ptrs = [B_shared.data.ptr] * gi_size

    # device arrays-of-pointers (uintp) for this group
    A_ptr_groups.append(cp.asarray(A_ptrs, dtype=cp.uintp))
    B_ptr_groups.append(cp.asarray(B_ptrs, dtype=cp.uintp))
    C_ptr_groups.append(cp.asarray(C_ptrs, dtype=cp.uintp))
    C_objs_groups.append(C_objs)

# ---- concatenate into one flat device array per A/B/C ----
A_flat = cp.concatenate(A_ptr_groups)  # dtype=uintp, length = problem_count
B_flat = cp.concatenate(B_ptr_groups)
C_flat = cp.concatenate(C_ptr_groups)
problem_count = int(A_flat.size)

# ---- per-group parameters ----
m_arr    = np.asarray(m_list, dtype=np.int32)
n_arr    = np.asarray([n]*G, dtype=np.int32)
k_arr    = np.asarray([k]*G, dtype=np.int32)
lda_arr  = np.asarray(m_list, dtype=np.int32)  # column-major
ldb_arr  = np.asarray([k]*G, dtype=np.int32)
ldc_arr  = np.asarray(m_list, dtype=np.int32)
alpha_arr= np.asarray([1.0]*G, dtype=np.float32)
beta_arr = np.asarray([0.0]*G, dtype=np.float32)
gsize_arr= np.asarray(size_li, dtype=np.int32)

# ---- ctypes bindings ----
c_int_p   = ct.POINTER(ct.c_int)
c_float_p = ct.POINTER(ct.c_float)

lib.sgemm_grouped_batched.restype  = ct.c_int
lib.sgemm_grouped_batched.argtypes = [
    ct.c_void_p,                 # cublasHandle_t
    ct.c_int,                    # group_count
    c_int_p, c_int_p, c_int_p,   # m[], n[], k[]
    c_int_p, c_int_p, c_int_p,   # lda[], ldb[], ldc[]
    c_float_p, c_float_p,        # alpha[], beta[]
    c_int_p,                     # group_size[]
    ct.c_void_p, ct.c_void_p,    # Aarray_dev_flat, Barray_dev_flat  (device pointers)
    ct.c_void_p                  # Carray_dev_flat
]

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
    gsize_arr.ctypes.data_as(c_int_p),
    ct.c_void_p(int(A_flat.data.ptr)),  # <-- flat device arrays-of-pointers
    ct.c_void_p(int(B_flat.data.ptr)),
    ct.c_void_p(int(C_flat.data.ptr))
)
if status != 0:
    raise RuntimeError(f"sgemm_grouped_batched failed with status {status}")

# for gi, (m, gi_size) in enumerate(zip(m_list, size_li)):
#     for j in range(min(2, gi_size)):
#         Cij = C_objs_groups[gi][j]
#         print(f"Group {gi} instance {j}: Cij shape after GEMM = {Cij.shape}")


# quick check a couple results
# (You will likely keep the original A objects around; here we only kept C’s.)
idx0 = 0
for gi, (m, gi_size) in enumerate(zip(m_list, size_li)):
    for j in range(min(2, gi_size)):
        # reconstruct A pointer view just for checking
        A_ptr = int(A_ptr_groups[gi][j].get())
        Aij = cp.ndarray((m, k), dtype=cp.float32,
                         memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(A_ptr, m*k*4, owner=None), 0),
                         order='F')
        ref = Aij @ B_shared
        err = float(cp.max(cp.abs(C_objs_groups[gi][j] - ref)).get())
        print(f"group {gi} sample {j} err={err:.3e}")
    idx0 += gi_size
