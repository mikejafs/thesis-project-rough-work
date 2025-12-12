import numpy as np
import cupy as cp
import ctypes

arr = np.random.rand(10)
arr_cp = cp.random.rand(10)
print(arr)

# arr_p = arr.data
arr_p = arr.data
arr_cp_p = arr_cp.data

print(type(arr_cp_p))
print(type(arr_p))


print(ctypes.c_void_p)