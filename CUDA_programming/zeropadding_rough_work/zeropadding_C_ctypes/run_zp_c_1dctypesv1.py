# gcc -shared -o out.so -fPIC zp_c_1dctypes.c

import ctypes
import numpy as np

full_path = "/home/mike/Thesis_rough_work/CUDA_programming/zeropadding_rough_work/zeropadding_C_1.1/out.so"

zp_algo = ctypes.cdll.LoadLibrary(full_path)

zp_algo.zeroPad.argtypes = [
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int)
]
#effectively define the return variable (in this case is a pointer)
zp_algo.zeroPad.restype = ctypes.POINTER(ctypes.c_int)


zp_algo.free_memory.argtypes = [ctypes.POINTER(ctypes.c_int)]
zp_algo.free_memory.restype = None

#zeropad function
def zero_pad(array, edges):
    array = np.array(array, dtype="int32")
    edges = np.array(edges, dtype="int32")
    edges_size = len(edges)

    out_rows = ctypes.c_int(0)
    out_cols = ctypes.c_int(0)

    result_ptr = zp_algo.zeroPad(
        array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        edges_size,
        ctypes.byref(out_rows),
        ctypes.byref(out_cols)
    )

    result = np.ctypeslib.as_array(result_ptr, shape=(out_rows.value*out_cols.value,))
    # result = result.reshape((out_rows.value, out_cols.value))

    return result

if __name__ == "__main__":
    array = np.random.randint(1, 100, 20)
    edges = [0, 5, 12, 18, len(array)]

    print(array)

    padded_result = zero_pad(array, edges)

    print(padded_result)




