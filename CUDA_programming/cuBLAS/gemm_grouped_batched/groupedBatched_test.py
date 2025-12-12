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
 to understand where things come from. DONE

2. turn the lib.groupedbatched... into a wrapper function anticipating the type of function
 we will eventually like to have. WILL DO THIS IN THE NEW 'groupedBatched_func.py' FILE

3. Start a new file that uses the simulated edges array to define the blocks. COMING

4. Compare the calculation Del^t @ N^-1 @ Del to both CuPy zeropadded & the one C function 
 ("make_small_blocks") from bobby's code. Note, however, that bobby's code accepts only double 
 precision  COMING

"""

import cupy as cp
import numpy as np
import ctypes

print('\n')



#load the grouped batched shared library file
lib = ctypes.cdll.LoadLibrary("/home/mike/Thesis_rough_work/CUDA_programming/cuBLAS/gemm_grouped_batched/gemmGroupedBatched.so")

#define the python-side argtypes for the grouped batched cublas gemm

#~~~ Note that numpy and cupy work differently when creating pointer types and accessing
#~~~ raw pointer values. From Numpy, the pointer type is created using the .POINTER(ctypes.c_etc)
#~~~ syntax, whereas with CuPy, .c_etc.. eg ctypes.c_int is all it takes. Therefore, the differe-
#~~~ nces in the way pointer objects are defined throughout is related to whether the code needs
#~~~ to live on the host or device -> Can check which is necessary through the cuBLAS docs.

lib.gemmGroupedBatched.argtypes = [
    ctypes.c_int,                   #just a number (device/host irrelevant)
    ctypes.POINTER(ctypes.c_int),   #on host
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,                #first array of pointers to sub matrices (on device)
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]




#small wrapper used to define pointers from (CuPy) python arrays
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

#First combine all A, B, C blocks
all_A = list(g0_Aarray) + list(g1_Aarray) + list(g2_Aarray)
all_B = list(g0_Barray) + list(g1_Barray) + list(g2_Barray)
all_C = list(g0_Carray) + list(g1_Carray) + list(g2_Carray)

#Arrays of raw device pointers
A_ptrs = cp.array([ar.data.ptr for ar in all_A], dtype=cp.uintp)
B_ptrs = cp.array([br.data.ptr for br in all_B], dtype=cp.uintp)
C_ptrs = cp.array([cr.data.ptr for cr in all_C], dtype=cp.uintp)

#Convert to ctypes (Using CuPy syntax for creating a pointer to a raw device memory address)
A_ptrs_ct = ctypes.c_void_p(A_ptrs.data.ptr)
B_ptrs_ct = ctypes.c_void_p(B_ptrs.data.ptr)
C_ptrs_ct = ctypes.c_void_p(C_ptrs.data.ptr)


#Now construct the auxilliary arrays

# Keep in mind, we are workig in row-major ordering, using the col-major ->
# row major 'trick' where we say that C = A @ B <=> C^T = B^T @ A^T. cuBLAS
# expects things in col-major, so we can set things up in row-major and
# provide matrices to the 'flipped' ordering of A and B in the matrix
# multiply. Because cuBLAS assumes things are in col-major, this appears to 
# cuBLAS like everything is transposed so from our end, we are doing C = A @ B,
# but cuBLAS actually sees C^T = A^T @ B^T. 

#Arrays of the matrix shapes
m_row = np.array([2, 4, 1], dtype=np.int32)  #rows of A matrices within each group
n_row = np.array([2, 4, 1], dtype=np.int32)  #cols of B matrices within each group
k_row = np.array([3, 3, 3], dtype=np.int32)  #rows of B matrices within each group (i.e., #eigenmodes for diffuse mat)

#arrays of leading dimensions (in row-major, leading dimensions are the # cols)
#the quickest way to define leading dimensions in this case is then:

lda = k_row.copy()
ldb = n_row.copy()
ldc = n_row.copy()

#arrays of transpose specification (in Python, 0 -> CUBLAS_OP_N, 1 -> CUBLAS_OP_T)
transA = np.array([0]*group_count, dtype=np.int32)
transB = np.array([0]*group_count, dtype=np.int32)

#alpha & beta arrays 
alpha_arr = np.array([1.0, 1.0, 1.0], dtype=np.float32)
beta_arr = np.array([0.0, 0.0, 0.0], dtype=np.float32)

#array of group sizes
group_sizes = np.array([4, 1, 2], dtype=np.int32)
print(transA.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))


#Call the grouped batched gemm from the .cu file. note the difference in ctypes pointer
#object depending on whether array is defined on the host or the device. For gemmGroupedBatched,
#only the array of pointers to the matrices that are being multiplied are stored on the device,
#hence the ctypes.c_void_p(A_ptrs.data.ptr) syntax at those arguments.
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


#check against Numpy
print(cp.allclose(g0_Carray, g0_Aarray @ g0_Barray))
print(cp.allclose(g1_Carray, g1_Aarray @ g1_Barray))
print(cp.allclose(g2_Carray, g2_Aarray @ g2_Barray))




