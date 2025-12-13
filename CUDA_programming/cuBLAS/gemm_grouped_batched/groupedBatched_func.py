import numpy as np
import cupy as cp
import ctypes
from simulate_params import *

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#add a single print line for better terminal readability
print()

lib = ctypes.cdll.LoadLibrary("/home/mike/Thesis_rough_work/CUDA_programming/cuBLAS/gemm_grouped_batched/gemmGroupedBatched.so")

#define the python-side argtypes for the grouped batched cublas gemm
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


def group_blocks_by_size(edges):
    edges = cp.asnumpy(edges)
    block_sizes = np.diff(edges)
 
    groups = {}
    for i, height in enumerate(block_sizes):
        groups.setdefault(height, []).append(i)

    return block_sizes, groups


#thin wrapper around the grouped batched matmul cublas function
def groupedBatchedMatmul(A_array, B_array, edges):
    # print('The edges array is:', edges)

    bl_sizes, grps = group_blocks_by_size(edges)
    group_sizes = np.array([len(grp_size) for grp_size in grps.values()], dtype=np.int32)
    # print('the group size are:', group_sizes)
    # print('The block sizes are:', bl_sizes)
    # print('PRINTING GRPS:', grps)

    groupCount = len(grps)  #total number of groups
    # print('The total number of groups is:', groupCount)

    transA = np.array([0]*groupCount, dtype=np.int32)
    transB = np.array([0]*groupCount, dtype=np.int32)
    
    alpha_arr = np.ones(groupCount, dtype=np.float32)
    beta_arr = np.zeros(groupCount, dtype=np.float32)

    m_array = np.zeros(groupCount, dtype=np.int32)
    n_array = np.zeros(groupCount, dtype=np.int32)
    k_array = np.zeros(groupCount, dtype=np.int32)

    #now build pointer arrays for each of the input and output matrices
    A_ptrs = []
    B_ptrs = []
    C_ptrs = []

    n_eig = A_array.shape[1]

    for g_i, (h, block_id) in enumerate(grps.items()):
        m = n_eig
        n = n_eig #above and this are equal only if we are performing A^T @ A
        k = h

        m_array[g_i] = m
        n_array[g_i] = n
        k_array[g_i] = k

        A_list = []
        B_list = []
        C_list = []
        
        for b_id in block_id:

            start = edges[b_id]
            stop = edges[b_id+1]

            # A_array_i = A_array[start:stop, :].T
            # B_array_i = B_array[start:stop, :]

            A_array_i = A_array[start:stop, :].astype(cp.float32, copy=False).T  # (n_eig, h)
            B_array_i = B_array[start:stop, :].astype(cp.float32, copy=False)    # (h, n_eig)


            C_array_i = cp.zeros((n_eig, n_eig), dtype=cp.float32)

            # print("Block", b_id)
            # print("A_block:\n", A_array_i)
            # print("B_block:\n", B_array_i)
            # print("Aᵀ @ B (CPU):\n", (A_array_i @ B_array_i).get())
     

            A_list.append(A_array_i)
            B_list.append(B_array_i)
            C_list.append(C_array_i)

        A_ptrs.append(A_list)
        B_ptrs.append(B_list)
        C_ptrs.append(C_list)

    # === Flatten pointer lists into ONE contiguous list (required by cuBLAS) ===
    flat_A = [arr for group in A_ptrs for arr in group]
    flat_B = [arr for group in B_ptrs for arr in group]
    flat_C = [arr for group in C_ptrs for arr in group]

    # Create device arrays of raw pointers
    A_ptrs_raw = cp.asarray([a.data.ptr for a in flat_A], dtype=cp.uintp)
    B_ptrs_raw = cp.asarray([b.data.ptr for b in flat_B], dtype=cp.uintp)
    C_ptrs_raw = cp.asarray([c.data.ptr for c in flat_C], dtype=cp.uintp)

    # ctypes pointer to the device arrays
    A_ptrs_ct = ctypes.c_void_p(A_ptrs_raw.data.ptr)
    B_ptrs_ct = ctypes.c_void_p(B_ptrs_raw.data.ptr)
    C_ptrs_ct = ctypes.c_void_p(C_ptrs_raw.data.ptr)

    lda_array = k_array.copy()
    ldb_array = n_array.copy()
    ldc_array = n_array.copy()

    lib.gemmGroupedBatched(
        groupCount,
        transA.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        transB.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        m_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        k_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        alpha_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        # ctypes.c_void_p(A_ptrs_dev.data.ptr),
        A_ptrs_ct,  
        lda_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        # ctypes.c_void_p(A_ptrs_dev.data.ptr),
        B_ptrs_ct,
        ldb_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        beta_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        # ctypes.c_void_p(A_ptrs_dev.data.ptr),
        C_ptrs_ct,
        ldc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        group_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    )

    flat_C = [arr for group in C_ptrs for arr in group]
    C_out = [None] * len(bl_sizes)
    idx = 0
    for (h, block_ids), size in zip(grps.items(), group_sizes):
        for j, b in enumerate(block_ids):
            C_out[b] = flat_C[idx + j]
        idx += size

    # print("edges:", edges)
    # print("block_sizes:", bl_sizes)
    # print("groups:", grps)
    # print("group_sizes:", group_sizes)
    # print("N_blocks:", len(bl_sizes), "sum(group_sizes):", group_sizes.sum())

    C_stacked = cp.stack(C_out, axis=0)
    return C_stacked  # ordered by block index



if __name__ == "__main__":
    
    n_ant = 500
    n_eig = 3
    n_src = 1

    cp.random.seed(10)
    spms = SimCorrcalParams(n_ant, n_eig, n_src, precision='float32', xp=cp)
    edges = spms.edges()

    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]
    # src = sim_data[2]


    # bl_sizes, grps = group_blocks_by_size(edges)
    # print(bl_sizes)
    
    # for g in grps.items():
    #     print(g)


    # a = np.ones(10).astype(np.float32)
    # for i in a[:1]:
    #     print(type(i))

    
    #test for gemmgroupedBatched as we go

    #we will eventually like to use this e-wise temp term:
    temp = noise[..., None] * diff

    # print('temp:', temp)

    #running the batched grouped matmul
    C_array = groupedBatchedMatmul(diff, temp, edges)
    print(C_array)

    # print(diff)






    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #also conclude with a single print line 
    print()