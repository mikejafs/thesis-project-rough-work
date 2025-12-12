"""
Hard coded grouped batched gemm using the gemmGroupedBatched cublas function. 

Used for testing correctness of the batched grouped gemm and for intuition 
building before jumping to the harder problem of using the edges array to 
define the different groups of matrices.
"""


"""
TODO:

1. Go through the code again in detail and understand why all the lda etc... have the shapes 
and dimensions they do for the problem provided. c=> Add sufficient documentation to be able
to understand where things come from.

2. turn the lib.groupedbatched... into a wrapper function anticipating the type of function
we will eventually like to have.

3. Start a new file that uses the simulated edges array to define the blocks.

4. Compare the calculation Del^t @ N^-1 @ Del to both CuPy zeropadded & the one C function from bobby's code  

"""



import cupy as cp
import numpy as np
import ctypes

print('\n')



#load the grouped batched shared library file
lib = ctypes.cdll.LoadLibrary("/home/mike/Thesis_rough_work/CUDA_programming/cuBLAS/gemm_grouped_batched/gemmGroupedBatched.so")

#define the python-side argtypes for the grouped batched cublas gemm
lib.gemmGroupedBatched.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]


#small wrapper used to define pointers from python arrays
def ptr(array):
    return ctypes.c_void_p(array.data.ptr)


#----------------------------------------------------------
# Small example with three groups, each containing a set of 
# matrices with different sizes than the other group. Note
# the groupings below match what we expect from corrcal 
# in that the cols are always the same (n_eig or n_src) and 
# the rows have different sizes.
#----------------------------------------------------------

group_count = 3

#Group 0: four 2x3*3x2 gemms
g0_Aarray = cp.random.rand(4, 2, 3, dtype = cp.float32)
g0_Barray = cp.random.rand(4, 3, 2, dtype = cp.float32)
g0_Carray = cp.zeros((4, 2, 2), dtype=cp.float32)

# print(g0_Aarray @ g0_Barray)

#Group 1: one 4x3*3x4 gemms
g1_Aarray = cp.random.rand(1, 4, 3, dtype = cp.float32)
g1_Barray = cp.random.rand(1, 3, 4, dtype = cp.float32)
g1_Carray = cp.zeros((1, 4, 4), dtype=cp.float32)


#Group 2: two 1x3*3x1 gemms 
g2_Aarray = cp.random.rand(2, 1, 3, dtype = cp.float32)
g2_Barray = cp.random.rand(2, 3, 1, dtype = cp.float32)
g2_Carray = cp.zeros((2, 1, 1), dtype=cp.float32)



#Create lists of pointers pointing to each of the matrices
# Combine all A, B, C blocks
all_A = list(g0_Aarray) + list(g1_Aarray) + list(g2_Aarray)
all_B = list(g0_Barray) + list(g1_Barray) + list(g2_Barray)
all_C = list(g0_Carray) + list(g1_Carray) + list(g2_Carray)

# Host arrays of device pointers
A_ptrs = np.array([ar.data.ptr for ar in all_A], dtype=np.uintp)
B_ptrs = np.array([br.data.ptr for br in all_B], dtype=np.uintp)
C_ptrs = np.array([cr.data.ptr for cr in all_C], dtype=np.uintp)

A_ptrs_dev = cp.asarray(A_ptrs)  # dtype=uint64 on device
B_ptrs_dev = cp.asarray(B_ptrs)
C_ptrs_dev = cp.asarray(C_ptrs)

# # Convert to ctypes
# A_ptrs_ct = A_ptrs_dev.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
# B_ptrs_ct = B_ptrs_dev.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
# C_ptrs_ct = C_ptrs_dev.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
# print(A_ptrs_ct)

A_ptrs_ct = ctypes.c_void_p(A_ptrs_dev.data.ptr)
B_ptrs_ct = ctypes.c_void_p(B_ptrs_dev.data.ptr)
C_ptrs_ct = ctypes.c_void_p(C_ptrs_dev.data.ptr)


#Arrays of the matrix shapes
m_row = np.array([2, 4, 1], dtype=np.int32)
n_row = np.array([2, 4, 1], dtype=np.int32)
k_row = np.array([3, 3, 3], dtype=np.int32)

#arrays of leading dimensions
# lda = np.array([2, 4, 1], dtype=np.int32)
# ldb = np.array([3, 3, 3], dtype=np.int32)
# ldc = np.array([2, 4, 1], dtype=np.int32)
lda = k_row.copy()
ldb = n_row.copy()
ldc = n_row.copy()

#arrays of transpose specification
transA = np.array([0]*group_count, dtype=np.int32)
transB = np.array([0]*group_count, dtype=np.int32)


#alpha & beta arrays 
alpha_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
beta_arr = np.array([0.0, 0.0, 0.0], dtype=np.float32)

#array of group sizes
group_sizes = np.array([4, 1, 2], dtype=np.int32)


lib.gemmGroupedBatched(
    group_count,
    transA.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    transB.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    m_row.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    n_row.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    k_row.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    alpha_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    # ctypes.c_void_p(A_ptrs_dev.data.ptr),
    A_ptrs_ct,  
    lda.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    # ctypes.c_void_p(A_ptrs_dev.data.ptr),
    B_ptrs_ct,
    ldb.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    beta_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    # ctypes.c_void_p(A_ptrs_dev.data.ptr),
    C_ptrs_ct,
    ldc.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    group_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
)


print(cp.allclose(g0_Carray, g0_Aarray @ g0_Barray))
print(cp.allclose(g1_Carray, g1_Aarray @ g1_Barray))
print(cp.allclose(g2_Carray, g2_Aarray @ g2_Barray))




