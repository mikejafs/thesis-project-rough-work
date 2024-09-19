import ctypes
import numpy as np

#define path to shared library
def get_arr_cpy():
    full_path = "/home/mike/Thesis_rough_work/CUDA_programming/zeropadding_rough_work/out.so"

    #load the shared library using the the dynamic linked library method, now can access functions in the .so file
    lib = ctypes.cdll.LoadLibrary(full_path)

    #assign function in lib.(..stuff..) to ArrayCopy, and specify what type of arguments this function expects
    ArrayCopy = lib.run_array_copy
    ArrayCopy.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int)
    ArrayCopy.restype = None

    return ArrayCopy

__arr_cpy = get_arr_cpy()


def Arr_cpy(a, b, arr_size):
    #Convert numpy arrays to ctypes pointers
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    __arr_cpy(a_ptr, b_ptr, arr_size)


if __name__ == '__main__':
    arr_size = 10
    a = np.random.randint(0, 100, arr_size).astype(np.float128)
    b = np.zeros(arr_size, dtype=np.float128)
    Arr_cpy(a, b, arr_size=arr_size)    
    print("The input array was", a)
    print("The output array is", b)

# # Step 5: Call the function from the shared library
# ArrayCopy(a_ptr, b_ptr, arr_size)

# print(a)
# print(b)


