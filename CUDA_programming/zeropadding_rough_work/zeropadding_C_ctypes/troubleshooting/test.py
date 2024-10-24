import ctypes
import numpy as np

# Path to the compiled C shared library
full_path = "/home/mike/Thesis_rough_work/CUDA_programming/zeropadding_rough_work/zeropadding_C_1.1/troubleshooting/out.so"

# Load the shared library
zp_lib = ctypes.cdll.LoadLibrary(full_path)

# Define argument types for the C function
zp_lib.zeroPad.argtypes = [
    ctypes.POINTER(ctypes.c_double),         # in_array
    ctypes.POINTER(ctypes.c_double),         # out_array
    ctypes.POINTER(ctypes.c_ulong),          # edges
    ctypes.c_ulong,                          # n_blocks
    ctypes.c_ulong                           # largest_block
]

# Define the zeroPad function in Python
def zeroPad(array, edges):
    array = np.array(array, dtype=np.double)
    edges = np.array(edges, dtype=np.uint64)  # Ensure edges is unsigned long

    largest_block = np.array(np.diff(edges).max(), dtype=np.uint64)
    n_blocks = np.array(edges.size - 1, dtype=np.uint64)

    out_array = np.zeros((n_blocks * largest_block), dtype=np.double)

    # Call the C function
    zp_lib.zeroPad(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        out_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong)),  # Pass edges as unsigned long
        n_blocks,
        largest_block
    )
    return out_array

if __name__ == "__main__":
    # Example usage
    n_bl = 10
    array = np.random.rand(n_bl)
    edges = np.unique(np.random.randint(1, n_bl - 1, size=2))
    edges = np.concatenate((np.array([0]), edges, np.array([n_bl], dtype=np.uint64)))
    
    print(f"Input array: {array}")
    print(f"Edges: {edges}")

    zp_array = zeroPad(array, edges)
    print("Zero-padded array:", zp_array)
