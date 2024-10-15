# gcc -shared -o out.so -fPIC zp_c_ctypes_1d_v2.c

import ctypes
import numpy as np

full_path = "/home/mike/Thesis_rough_work/CUDA_programming/zeropadding_rough_work/zeropadding_C_1.1/out.so"

zp_lib = ctypes.cdll.LoadLibrary(full_path)

# try:
#     zp_lib = ctypes.cdll.LoadLibrary(full_path)
# except OSError as e:
#     print(f"Error loading shared library: {e}")


zp_lib.zeroPad.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_long,
    ctypes.c_long
]

def zeroPad(array, edges):
    array = np.array(array, dtype=np.double)
    edges = np.array(edges, dtype=np.int64)
    largest_block = np.array(np.diff(edges).max(), dtype = np.int64)
    n_blocks = np.array(edges.size - 1, dtype = np.int64)

    #can add some logic here in the future to
    #decide which (1d or 2d) ZP routine to use.
    #For now things are just 1D

    out_array = np.zeros((n_blocks*largest_block), dtype = np.double)

    zp_lib.zeroPad(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        n_blocks,
        largest_block
    )
    return out_array, largest_block, n_blocks


if __name__ == "__main__":
    n_bl = 50
    array = np.random.rand(n_bl)
    edges = np.unique(np.random.randint(1, n_bl - 1, size = 5))
    edges = np.concatenate((np.array([0]), edges, np.array([n_bl], dtype = np.int64)))
    zp_array, largest_block, n_blocks = zeroPad(array, edges)
    zp_array = zp_array.reshape(n_blocks, largest_block)
    print(zp_array)


