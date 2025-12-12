import ctypes
import cupy as cp

full_path = "/home/mike/Thesis_rough_work/CUDA_programming/cuBLAS/mat_vec_mul.so"

mat_vec_mul_lib = ctypes.cdll.LoadLibrary(full_path)

mat_vec_mul_lib.mat_vec_mul.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # Pointer to the matrix (A)
    ctypes.POINTER(ctypes.c_float),  # Pointer to the vector (x)
    ctypes.POINTER(ctypes.c_float),  # Pointer to the vector (y)
    ctypes.c_int                     # number of rows in the matrix (A)
]


def mat_vec_mul(A, x):
    """
    Perform matrix-vector multiplication using a shared library.

    Parameters:
    A (cupy.ndarray): The input matrix (2D array).
    x (cupy.ndarray): The input vector (1D array).

    Returns:
    cupy.ndarray: The result of the matrix-vector multiplication.
    """
    assert A.ndim == 2, "A must be a 2D array"
    assert x.ndim == 1, "x must be a 1D array"
    # assert A.shape[1] == x.shape[0], "Matrix and vector dimensions do not match"

    y = cp.zeros(A.shape[0], dtype=A.dtype)

    mat_vec_mul_lib.mat_vec_mul(
        ctypes.cast(A.data.ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(x.data.ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(y.data.ptr, ctypes.POINTER(ctypes.c_float)),
        A.shape[0]  # number of rows in the matrix A
    )

    return y    


def explicit_matmul(A, x):
    out = cp.zeros((A.shape[0], 1))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            out[i] += A[i, j] * x[j]
    return out


# Example usage:
if __name__ == "__main__":
    A = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=cp.float32)
    x = cp.array([1, 2, 3], dtype=cp.float32)

    y_cublas = mat_vec_mul(A, x)
    y_cupy = A@x
    print(f"result of cupy operation: {y_cupy}")
    print("Result of matrix-vector multiplication:", y_cublas)


    print(explicit_matmul(A, x))
    # Expected output: [14. 32.]
