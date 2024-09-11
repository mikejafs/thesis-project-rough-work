import ctypes
import numpy as np

full_path = "/home/mike/Thesis_rough_work/CUDA_programming/zeropadding_rough_work/out.so"
lib = ctypes.cdll.LoadLibrary(full_path)

ArrayCopy = lib.run_array_copy
ArrayCopy.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int)
# ArrayCopy.argtypes = ()
ArrayCopy.restype = None

arr_size = 10
a = np.random.randint(0, 100, arr_size).astype(np.int32)
b = np.zeros(arr_size, dtype=np.int32)

# Step 4: Convert numpy arrays to ctypes pointers
a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

# Step 5: Call the function from the shared library
ArrayCopy(a_ptr, b_ptr, arr_size)

print(a)
print(b)


