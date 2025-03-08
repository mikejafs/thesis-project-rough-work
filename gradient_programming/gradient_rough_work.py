"""
MODULE FOR CONSTRUCTING AND TESTING THE FULL GRADIENT OF THE LIKELIHOOD FUNCTION
"""
import ctypes
import os
import sys

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
target_dir = os.path.join(parent_dir, 'corrcal_gpu_pipeline', 'pipeline')
sys.path.insert(0, target_dir)

import numpy as np
import cupy as cp
import corrcal
from corrcal import SparseCov
from corrcal.optimize import *
from zp_puregpu_funcs_py import *
from utils import *
from invcov import *
from accumulate_grad_py import *
from cupyx.profiler import benchmark

#pseudo-ish code to follow...

"""
Reminder to make smaller unit tests allong the way here. For example, before testing everything including the accumulate 
gradient function, construct all the p, q, s, and t matrices and compare to Bobby's code. Then proceed to testing new 
accumulate grad function against the full routine...
"""

#full grad function
def gpu_grad_nll(gains, data, scale, phs_norm_fac, noise, diff_mat, src_mat, edges, ant_1_array, ant_2_array, xp):
    """

    """

    #0.5 zeropad EVERYTHING....
    #zeropad noise, diffuse, source matrices, and gain matrices
    zp_noise_inv, lb, nb = zeroPad(noise, edges, return_inv=True)
    zp_noise, _, _ = zeroPad(noise, edges, return_inv=False)  #need the non-inverse for constructing regular sparce cov
    zp_diff_mat, _, _ = zeroPad(diff_mat, edges, return_inv=False)
    zp_src_mat, _, _ = zeroPad(src_mat, edges, return_inv=False)
    zp_data, _, _ = zeroPad(data, edges, return_inv=False)
    zp_cplex_gain_mat = zeropad_gains(gains, edges, ant_1_array, ant_2_array, xp = cp, return_inv=False)

    #1. apply gains to the source and diffuse matrices (ie. constructing the 'true' convariance)
    gain_diff_mat = apply_gains(zp_cplex_gain_mat, zp_diff_mat, xp=cp)
    gain_src_mat = apply_gains(zp_cplex_gain_mat, zp_src_mat, xp=cp)

    # print("gpu gain diff mat is")
    # print(gain_diff_mat)

    #2. compute inverse cov components
    # print(gain_diff_mat)
    # print(gain_src_mat)
    # print(zp_noise[...,None].shape)
    # print(gain_diff_mat.shape)
    inv_noise, inv_diff, inv_src = inverse_covariance(zp_noise_inv, gain_diff_mat, gain_src_mat, edges, cp, ret_det=False, N_is_inv=True)
    # print("gpu")
    # print(inv_diff)
    # print(cp.allclose(inv_noise, zp_noise_inv))

    #(I think) can roughly follow current python/C implimentation for the next few steps without much change

    #3. Now compute p = C^-1 @ data => Might want to construct my own __matmul__ function for this
    p = sparse_cov_times_vec(zp_noise, zp_diff_mat, zp_src_mat, inv_noise, inv_diff, inv_src, zp_data, isinv=True, xp=cp)
    # print(p.shape)
    # print(p)

    #4. compute q = (C - N) @ G.T @ p
    q = p.copy()
    q[:, ::2] = zp_cplex_gain_mat.real*p[:, ::2] + zp_cplex_gain_mat.imag*p[:, 1::2]
    q[:, 1::2] = -zp_cplex_gain_mat.imag*p[:, ::2] + zp_cplex_gain_mat.real*p[:, 1::2]
    # print(zp_noise.shape)
    # print(q.shape)
    zp_noise = zp_noise.reshape(nb, lb, 1) #1D mats are left as 2D and not 2D + 1 col so that invcov runs so need to reshape here
    # print(q)

    #trying to follow Bobby's code, but not sure why it is cov@q since the formalism is (C-N)
    zp_noise = cp.zeros_like(zp_noise) #=> This is why

    #CURRENTLY THE SOURCE OF THE PROBLEM -> NEED TO FIGURE OUT IF THIS ROUTINE IS RUNNING PROPERLY BEFORE PROCEEDING
    q = sparse_cov_times_vec(zp_noise, zp_diff_mat, zp_src_mat, inv_noise, inv_diff, inv_src, q, isinv=False, xp=cp)
    # print(q)

    #5. compute s and t
    zp_s = p[:, ::2]*q[: ,::2] + p[:, 1::2]*q[:, 1::2]
    zp_t = p[:, 1::2]*q[:, ::2] - p[:, ::2]*q[:, 1::2]

    # print(zp_diff_mat.shape)
    # print((zp_diff_mat[:, ::2]**2).shape)
    # print((zp_diff_mat[:, 1::2]**2).shape)
    # print(np.sum(zp_diff_mat[:, ::2]**2 + zp_diff_mat[:, 1::2]**2, axis=2))
    # a = np.sum(zp_diff_mat[:, ::2]**2, axis=2)
    # a = a.reshape(nb*int(lb/2))
    # print(a)

    #6. compute the inverse power
    inv_power = cp.sum(
        inv_diff[:, ::2]**2 + inv_diff[:, 1::2]**2, axis=2
    ) + cp.sum(
        inv_src[:, ::2]**2 + inv_src[:, 1::2]**2, axis=2
    )

    # print("gpu", zp_s)
    # print("gpu", zp_t)
    # print("gpu", inv_power)

    #WILL NEED TO UNDO ZEROPAD THIS AND EVERYTHING ELSE BEFORE PASSING TO THE ACCUMUILATE GRADIENT
    #FUNCTION
    inv_power = inv_power.reshape(nb, int(lb/2), 1)
    # print(inv_power)
    # print(inv_power.shape)
    # print(lb)
    # print(nb)

    #7. accumulate gradient
    #Need to first undo the zeropadding of s, t, and P
    s = undo_zeroPad(zp_s, edges, ReImsplit=False)
    t = undo_zeroPad(zp_t, edges, ReImsplit=False)
    P = undo_zeroPad(inv_power, edges, ReImsplit=False)

    # print(zp_s)
    # print(zp_s.shape)
    #
    # print(s)
    # print(t)
    # print(P)

    gradient = accumulate_gradient(
        gains, s, t, P, noise, ant_1_array, ant_2_array
    )

    # print(gradient)

    # print(gradient)
        #nominally, k1, k2 indices are constructed from looping over ant_inds. This is problematic since
        # ant_inds now do not follow the zeropadding scheme.

        #SOLN: return a new ant_1_ind and ant_2_ind that correspond to the indices of the zeropadded matrices

        #7.1.

    amps = cp.sqrt(gains[::2]**2 + gains[1::2]**2)
    phases = cp.arctan2(gains[1::2], gains[::2])
    n_ants = gains.size//2
    grad_phs_prefac = 2 * cp.sum(phases) / (amps * n_ants**2 * phs_norm_fac**2)
    gradient[::2] -= grad_phs_prefac * cp.sin(phases)
    gradient[1::2] += grad_phs_prefac * cp.cos(phases)

    return gradient/scale


def accumulate_gradient(gains, s, t, P, noise, ant_1_inds, ant_2_inds):
    """Loop over baselines and accumulate the per-antenna gradient contribs.

    Thin wrapper around the accumulate_gradient CUDA function.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    gradient = cp.zeros_like(gains)
    n_bls = ant_1_inds.size
    acc_grad_lib.accumulate_gradient(
        ctypes.cast(gains.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(s.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(t.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(P.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(noise.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(gradient.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(ant_1_inds.data.ptr, ctypes.POINTER(ctypes.c_long)),
        ctypes.cast(ant_2_inds.data.ptr, ctypes.POINTER(ctypes.c_long)),
        n_bls,
    )
    return gradient

#
# def cpu_grad_nll(gains, cov, data, ant_1_inds, ant_2_inds, scale=1, phs_norm_fac=np.inf):
#     """Calculate the gradient of the negative log-likelihood.
#
#     This is the gradient with respect to the real/imaginary per-antenna gains.
#     See Eq. ?? of Pascua+ 25 for details of what is being calculated.
#
#     Parameters
#     ----------
#     same as nll. fill this out later.
#     """
#     # Prepare the gain matrix.
#     gains = gains / scale
#     complex_gains = gains[::2] + 1j * gains[1::2]
#     gain_mat = complex_gains[ant_1_inds] * complex_gains[ant_2_inds].conj()
#
#     # Prepare some auxiliary matrices/vectors.
#     cinv = cov.copy()
#     cinv.apply_gains(gains / scale, ant_1_inds, ant_2_inds)
#     # print(cinv.diff_mat)
#     cinv = cinv.inv(return_det=False)
#     # print("cpu")
#     # print(cinv.diff_mat)
#     p = cinv @ data
#     # print("cpu", p)
#     noise = cov.noise
#     cov = cov.copy()
#     cov.noise = np.zeros_like(cov.noise)
#     # print(cinv.diff_mat)
#
#     # Compute q = (C-N) @ G.T @ p.
#     q = p.copy()
#     q[::2] = gain_mat.real * p[::2] + gain_mat.imag * p[1::2]
#     q[1::2] = -gain_mat.imag * p[::2] + gain_mat.real * p[1::2]
#     # print(q)
#     q = cov @ q
#
#     # print(q)
#
#
#     # Now compute s = Re(q.conj() * p), t = Im(q.conj() * p).
#     s = p[::2] * q[::2] + p[1::2] * q[1::2]
#     t = p[1::2] * q[::2] - p[::2] * q[1::2]
#
#     # Compute the "inverse power" for use in the trace calculation.
#     inv_power = np.sum(
#         cinv.diff_mat[::2] ** 2 + cinv.diff_mat[1::2] ** 2, axis=1
#     ) + np.sum(
#         cinv.src_mat[::2] ** 2 + cinv.src_mat[1::2] ** 2, axis=1
#     )
#     # print("cpu", s)
#     # print("cpu", t)
#     # print("cpu", inv_power)
#
#
#     gradient = accumulate_gradient(
#         gains, s, t, inv_power, noise, ant_1_inds, ant_2_inds
#     )
#
#     print(gradient)
#
#     # # Accumulate the contributions from the phase normalization.
#     # amps = np.sqrt(gains[::2] ** 2 + gains[1::2] ** 2)
#     # phases = np.arctan2(gains[1::2], gains[::2])
#     # n_ants = complex_gains.size
#     # grad_phs_prefac = 2 * np.sum(phases) / (amps * n_ants ** 2 * phs_norm_fac ** 2)
#     # gradient[::2] -= grad_phs_prefac * np.sin(phases)
#     # gradient[1::2] += grad_phs_prefac * np.cos(phases)
#
#     # return gradient / scale

def simulate(debugging=False):
    #NOTE: Need to construct a class for simulating these parameters
    #   with detailed descriptions of the sizes with motivations

    #TODO: Do the above.....

    # simulation params relevant for testing application of gains to a matrix
    n_ant = 500
    n_bl = int((n_ant**2))
    n_gains = 2*n_bl
    n_eig = 3
    n_src = 5
    xp = cp  # run things on the gpu using cupy

    # this might be the easiest (and most general) way to devise an edges
    # array, though we hard code an ex. edges array to be sure it fits
    # the desired format of having no odd entries.
    edges = (xp.unique(xp.random.randint(1, int(n_bl / 2) - 1, size=n_ant) * 2))
    edges = xp.concatenate((xp.array([0]), edges, xp.array([n_bl])))
    # print(f"The edges of the redundant blocks have indices{edges}")

    # some random noise, diffuse, source covariance matrices, and gain mat
    xp = cp
    sim_noise_mat = xp.random.rand(n_bl, dtype='float64')
    sim_diff_mat = xp.random.rand(n_bl, n_eig, dtype='float64')
    sim_src_mat = xp.random.rand(n_bl, n_src, dtype='float64')
    sim_gains = cp.random.rand(n_gains, dtype='float64')  # Re/Im split + ant1 & ant 2 = 4*n_ant
    sim_data = xp.random.rand(n_bl, dtype='float64')
    ant_1_array = cp.arange(n_bl//2)
    ant_2_array = cp.arange(n_bl//2, 2*n_bl//2)

    #testing grad_nll on the gpu
    gpu_grad = gpu_grad_nll(sim_gains, sim_data, 1, np.inf, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, ant_1_array, ant_2_array, xp=cp)
    # print(benchmark(gpu_grad_nll, (sim_gains, sim_data, 1, np.inf, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, ant_1_array, ant_2_array, cp), n_repeat=10))

    #testing existing cpu version; need to convert everything to Numpy arrays first
    noise_mat = cp.asnumpy(sim_noise_mat)
    src_mat = cp.asnumpy(sim_src_mat)
    diff_mat = cp.asnumpy(sim_diff_mat)
    edges_mat = cp.asnumpy(edges)
    gains_mat = cp.asnumpy(sim_gains)
    data_vec = cp.asnumpy(sim_data)
    ant_1_data = cp.asnumpy(ant_1_array)
    ant_2_data = cp.asnumpy(ant_2_array)
    # print(type(noise_mat))
    # print(type(src_mat))
    # print(type(diff_mat))
    # print(type(edges))

    cov = SparseCov(noise_mat, src_mat, diff_mat, edges_mat, n_eig, isinv=False)
    cpu_grad = grad_nll(gains_mat, cov, data_vec, ant_1_data, ant_2_data, scale=1, phs_norm_fac=np.inf)
    # print(benchmark(grad_nll, (gains_mat, cov, data_vec, ant_1_data, ant_2_data, 1, np.inf), n_repeat=10))

    # cpu_grad_nll(gains_mat, cov, data_vec, ant_1_data, ant_2_data, scale=1, phs_norm_fac=np.inf)

    if debugging:
        #test whether the zp gains and apply gains still work when done separately
        zp_gains = zeropad_gains(sim_gains, edges, ant_1_array, ant_2_array, xp = cp, return_inv=False)
        print(zp_gains)

        #zeropad the diffuse gain mat
        zp_sim_diff_mat, largest_block, n_blocks = zeroPad(sim_diff_mat, edges, return_inv=False)

        out = apply_gains(zp_gains, zp_sim_diff_mat, xp=cp)
        print(out)

    print(np.allclose(gpu_grad, cpu_grad))


    # print(cpu_grad.shape)
    # print(gpu_grad.shape)

if __name__ == "__main__":
    #simulate test matrices to test intermitantly as we go along above
    simulate(debugging=False)












