import ctypes
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cupyx.profiler import benchmark

full_path = "/home/mike/Thesis_rough_work/CUDA_programming/zeropadding_rough_work/zeropadding_CUDA/out.so"

zp_cuda_lib = ctypes.cdll.LoadLibrary(full_path)

zp_cuda_lib.zeroPad.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

def zeroPad(array, edges):
    array = np.array(array, dtype=np.double)
    edges = np.array(edges, dtype=np.int64)
    array_size = array.shape[0] 
    largest_block = np.array(np.diff(edges).max(), dtype = np.int32)
    n_blocks = np.array(edges.size - 1, dtype = np.int32)
    
    #can add some logic here in the future to
    #decide which (1d or 2d) ZP routine to use.
    #For now things are just 1D

    out_array = np.zeros((n_blocks*largest_block), dtype = np.double)

    zp_cuda_lib.zeroPad(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
        array_size,
        n_blocks,
        largest_block
    )
    return out_array, largest_block, n_blocks

if __name__ == "__main__":
    n_bl = 120000
    #for more easy to verify case:
    # array = np.array([1,2,3,4,5,6,7,8,9, 10, 11, 12, 13, 14])
    # edges = np.array([0, 2, 9, 14])

    array = np.random.rand(n_bl)
    # array = np.random.randint(1, 9, size=n_bl)

    #use this array along with the seaborn heatmap if want to check things are working
    # array = np.full(n_bl, 1, dtype=np.double)

    edges = np.unique(np.random.randint(1, n_bl - 1, size = 500))
    edges = np.concatenate((np.array([0]), edges, np.array([n_bl], dtype = np.int64)))
    zp_array, largest_block, n_blocks = zeroPad(array, edges)

    # #uncomment if want to time things
    # test_results = str(benchmark(zeroPad, (array, edges), n_repeat=100))
    # test_results = test_results.split()
    # cpu_t = float(test_results[3])/1e6
    # gpu_t = float(test_results[14])/1e6
    # print(f"Time on cpu: {cpu_t:.6f}s")
    # print(f"Time on gpu: {gpu_t:.6f}s")

    zp_array = zp_array.reshape(n_blocks, largest_block)
    print(zp_array)
    
    # pull up a clear map of the output array in 2D
    sns.heatmap(zp_array)
    plt.show()