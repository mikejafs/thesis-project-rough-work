import cupy as cp
import ctypes

lib = ctypes.cdll.LoadLibrary("./libgemm.so")

# lib.gemm.restype = ctypes.c_int
lib.matmul_gemm.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

#Dimensions for this test case
m, n, k = 2, 3, 4

#Wrapper function for cublas matmul
def gemm(A, B):
    m = A.shape[0]
    k = A.shape[1]
    n = B.shape[1]
    C = cp.zeros((m, n), dtype = cp.float32)

    lib.matmul_gemm(
        A.data.ptr,
        B.data.ptr,
        C.data.ptr,
        m, n, k
    )
    return C

#example useage
if __name__ == "__main__":

    #Allocate CuPy arrays (column major or Fortran order)
    # A = cp.asfortranarray(cp.random.rand(m, k, dtype = cp.float32))
    # B = cp.asfortranarray(cp.random.rand(k, n, dtype = cp.float32))
    
    A = (cp.random.rand(m, k, dtype = cp.float32))
    B = (cp.random.rand(k, n, dtype = cp.float32))


    #call cublas wrapper function
    result = gemm(A, B)

    #test against numpy:
    #....
    numpy_result = A@B
    # numpy_result = cp.asfortranarray(numpy_result) #convert back to fortran order for comparison
    

    cp.cuda.Stream.null.synchronize()
    print("A:\n", A)
    print("B:\n", B)
    print("C = A@B:\n", result)
    print("with numpy:\n", numpy_result)
    if cp.allclose(result, numpy_result):
        print("The gemm was a success")
    else:
        print("The result don't match")
