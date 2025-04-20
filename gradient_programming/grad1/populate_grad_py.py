"""
Module to house the importing of the shared library for populate gradient cuda function
"""

import ctypes
import cupy as cp

full_path = "/home/mike/Thesis_rough_work/gradient_programming/populate_grad.so"

pop_grad_lib = ctypes.cdll.LoadLibrary(full_path)

pop_grad_lib.populate_gradient.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_long),
    ctypes.POINTER(ctypes.c_long),
    ctypes.c_int,
    ctypes.c_int
]