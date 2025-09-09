import cupy as cp
import numpy as np
import ctypes

# 1) Define your block‑diagonal data
#    e.g. three blocks of sizes 2, 3, and 1
x = 3
# block_sizes = [x, int(4*x/3), int(x/2)]

block_sizes = [2]

# Generate random FP32 blocks in Fortran (col‑major) order
# blocks = [
#     cp.asfortranarray(cp.random.rand(sz, 2, dtype=cp.float32))
#     for sz in block_sizes
# ]

blocks = [
    cp.asfortranarray(cp.random.rand(sz, 1, dtype=cp.float32))
    for sz in block_sizes
]

#print the blocks
print("Blocks:")
for i, block in enumerate(blocks):
    print(f"Block {i} (size {block.shape}):")
    print(block)

# Compute the start/stop row indices for each block in the big N×N matrix
starts = [int(sum(block_sizes[:i])) for i in range(len(block_sizes))]
stops  = [starts[i] + block_sizes[i] for i in range(len(block_sizes))]

print(starts)
print(stops)

# 2) Create a random dense X of shape (N, P)
N = sum(block_sizes)
P = 1  # e.g. 4 columns
X = cp.asfortranarray(cp.random.rand(N, P, dtype=cp.float32))

#print the X matrix
print("\nX matrix:")
print(X)
print()

# 3) (rest of your code, unchanged)
batchCount = len(blocks)

# Prepare pointers to each diagonal block
A_ptrs = (ctypes.c_void_p * batchCount)(
    *[blocks[i].data.ptr for i in range(batchCount)]
)

# Ensure X, Y are Fortran (col‑major)
Xf = cp.asfortranarray(X)
Yf = cp.zeros((N, P), dtype=cp.float32, order='F')

# Load the shared library and bind the function
lib = ctypes.cdll.LoadLibrary('./libblockdiaggemm.so')
fn  = lib.block_diag_gemm_cublaslt
fn.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),  # A_ptrs
    ctypes.POINTER(ctypes.c_int),     # starts
    ctypes.POINTER(ctypes.c_int),     # stops
    ctypes.c_void_p,                  # X
    ctypes.c_void_p,                  # Y
    ctypes.c_int,                     # N
    ctypes.c_int,                     # P
    ctypes.c_float,                   # alpha
    ctypes.c_float,                   # beta
    ctypes.c_int                      # batchCount
]
fn.restype = None

# 4) Call into cuBLASLt
fn(
    A_ptrs,
    np.array(starts, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    np.array(stops,  dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    ctypes.c_void_p(Xf.data.ptr),
    ctypes.c_void_p(Yf.data.ptr),
    N, P,
    1.0, 0.0,
    batchCount
)

# Now Yf contains α·(block‑diag M)·X + β·Y  for each block in one shot.
print(Yf)


# print(blocks[0].shape)
# print(Xf.T.shape)

#compare answer with cupy
Y_cupy = cp.zeros((N, P), dtype=cp.float32)
for i, block in enumerate(blocks):
    start, stop = starts[i], stops[i]
    print(start, stop)
    print(f"Processing block {i} with shape {block.shape}")
    print(block.shape, Xf[start:stop].T.shape)
    print(Y_cupy[start:stop].shape)
    print((block @ Xf[start:stop].T).shape)
    Y_cupy[start:stop] = block @ Xf[start:stop].T

# print("\nY matrix from cupy:")
# print(Y_cupy)

# # Check if the results match    
# if cp.allclose(Yf, Y_cupy):
#     print("\nThe results match!")
# else:
#     print("\nThe results do not match!")
#     print("Yf:", Yf)
#     print("Y_cupy:", Y_cupy)    

