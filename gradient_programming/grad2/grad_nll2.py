"""
NOTE: (UPDATES) Cleaned things up compared to the grad1 folder. Still figuring out
                the correct working relationship between gains and ant_1&2_arrays

MODULE CONTAINIG THE GRADIENT OF THE NEGATIVE LOG LIKELIHOOD FUNCTION

This version used the 'chain rule' version of the gradient
"""

import ctypes
import os
import sys

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
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
from populate_grad_py2 import *
from cupyx.profiler import benchmark



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
    # print(q.shape)

    #5. compute s and t => Note this bring the shape of s & t to 1/2len(p or q)
    zp_s = p[:, ::2]*q[: ,::2] + p[:, 1::2]*q[:, 1::2]
    zp_t = p[:, 1::2]*q[:, ::2] - p[:, ::2]*q[:, 1::2]
    # print(f"shape of s {zp_s.shape}")
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
    #s and t are impicitely p and q 
    s = undo_zeroPad(zp_s, edges, ReImsplit=False)
    t = undo_zeroPad(zp_t, edges, ReImsplit=False)
    P = undo_zeroPad(inv_power, edges, ReImsplit=False)

    gradr, gradi = populate_gradient(
        n_ant, gains, s, t, P, noise, ant_1_array, ant_2_array
    )

    # gradr = gradr.reshape(n_ant, n_ant)
    # gradi = gradi.reshape(n_ant, n_ant)
    # print(gradr)
    # print(gradi)
    # print(gradr.T + gradr)
    # print(gains[::2])

    A = gradr + gradr.T
    B = gradi.T - gradi

    # dLdgr = (gradr.T + gradr)@gains[::2] + (gradi.T - gradi)@gains[1::2]
    # dLdgi = (gradr.T + gradr)@gains[1::2] + (gradi - gradi.T)@gains[::2]

    dLdgr = A@gains[::2] + B@gains[1::2]
    dLdgi = A@gains[1::2] - B@gains[::2]


    gradient = cp.zeros((2*len(dLdgr)))
    # print(gradient.shape)
    # print(dLdgr.shape)
    # print(dLdgi.shape)

    gradient[::2] = dLdgr
    gradient[1::2] = dLdgi

    # print(gradient)

    amps = cp.sqrt(gains[::2]**2 + gains[1::2]**2)
    phases = cp.arctan2(gains[1::2], gains[::2])
    n_ants = gains.size/2
    grad_phs_prefac = 2 * cp.sum(phases) / (amps * n_ants**2 * phs_norm_fac**2)
    gradient[::2] -= grad_phs_prefac * cp.sin(phases)
    gradient[1::2] += grad_phs_prefac * cp.cos(phases)


    return gradient/scale


"""
---------------------------------
Bobby's code
---------------------------------
"""
def cpu_grad_nll(gains, cov, data, ant_1_inds, ant_2_inds, scale=1, phs_norm_fac=np.inf):
    """Calculate the gradient of the negative log-likelihood.

    This is the gradient with respect to the real/imaginary per-antenna gains.
    See Eq. ?? of Pascua+ 25 for details of what is being calculated.

    Parameters
    ----------
    same as nll. fill this out later.
    """
    # Prepare the gain matrix.
    gains = gains / scale
    complex_gains = gains[::2] + 1j * gains[1::2]
    gain_mat = complex_gains[ant_1_inds] * complex_gains[ant_2_inds].conj()

    # Prepare some auxiliary matrices/vectors.
    cinv = cov.copy()
    cinv.apply_gains(gains / scale, ant_1_inds, ant_2_inds)
    # print(cinv.diff_mat)
    cinv = cinv.inv(return_det=False)
    # print("cpu")
    # print(cinv.diff_mat)
    p = cinv @ data
    # print("cpu", p)
    noise = cov.noise
    cov = cov.copy()
    cov.noise = np.zeros_like(cov.noise)
    # print(cinv.diff_mat)

    # Compute q = (C-N) @ G.T @ p.
    q = p.copy()
    q[::2] = gain_mat.real * p[::2] + gain_mat.imag * p[1::2]
    q[1::2] = -gain_mat.imag * p[::2] + gain_mat.real * p[1::2]
    # print(q)
    q = cov @ q



def populate_gradient(n_ant, gains, s, t, P, noise, ant_1_inds, ant_2_inds):
    """Loop over baselines and accumulate the per-antenna gradient contribs.

    Thin wrapper around the accumulate_gradient CUDA function.

    Parameters
    ----------
    #TODO

    Returns
    -------
    #TODO
    """
    dLdGr = cp.zeros((n_ant, n_ant))
    dLdGi = cp.zeros((n_ant, n_ant))
    
    n_bls = ant_1_inds.size
    # print(n_bls)

    # n_ant = len(gains)
    pop_grad_lib.populate_gradient(
        ctypes.cast(gains.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(s.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(t.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(P.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(noise.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(dLdGr.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(dLdGi.data.ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(ant_1_inds.data.ptr, ctypes.POINTER(ctypes.c_long)),
        ctypes.cast(ant_2_inds.data.ptr, ctypes.POINTER(ctypes.c_long)),
        n_ant,
        n_bls,
    )
    return dLdGr, dLdGi


def simulate(n_ant, debugging=False):
    #TODO: construct a class for simulating these parameters
    #with detailed descriptions of the sizes with motivations

    
    # simulation params relevant for testing application of gains to a matrix
    # n_ant = (n_ant,)
    ant_1_array, ant_2_array = cp.tril_indices(n_ant, k=-1)
    n_bl = 2*len(ant_1_array)
    # n_bl = int((2*n_ant**2))
    n_gains = 2*n_ant
    n_eig = 3
    n_src = 5
    xp = cp  # run things on the gpu using cupy

    # this might be the easiest (and most general) way to devise an edges
    # array, though we hard code an ex. edges array to be sure it fits
    # the desired format of having no odd entries.
    edges = (xp.unique(xp.random.randint(1, int(n_bl / 2) - 1, size=n_ant)* 2))
    edges = xp.concatenate((xp.array([0]), edges, xp.array([n_bl])))
    # print(f"The edges of the redundant blocks have indices{edges}")


    """ RUNNING WITH GPU """
    # some random noise, diffuse, source covariance matrices, and gain mat
    xp = cp
    sim_noise_mat = xp.random.rand(n_bl, dtype='float64')
    sim_diff_mat = xp.random.rand(n_bl, n_eig, dtype='float64')
    sim_src_mat = xp.random.rand(n_bl, n_src, dtype='float64')
    sim_gains = cp.random.rand(n_gains, dtype='float64')  # Re/Im split + ant1 & ant 2 = 4*n_ant
    sim_data = xp.random.rand(n_bl, dtype='float64')
    # ant_1_array = cp.arange(n_bl//2)
    # # ant_2_array = cp.arange(n_bl)
    # ant_2_array = cp.arange(n_bl//2, 2*n_bl//2)

    # ant_1_upper, ant_2_upper = cp.triu_indices(n_ant, k=0)
    # ant_1_array, ant_2_array = cp.tril_indices(n_ant, k=-1)
    # ant_1_array = cp.concatenate((ant_1_upper, ant_1_lower))
    # ant_2_array = cp.concatenate((ant_2_upper, ant_2_lower))
    # print(ant_1_array)
    # print(ant_2_array)

    #testing grad_nll on the gpu
    gpu_grad = gpu_grad_nll(sim_gains, sim_data, 1, np.inf, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, ant_1_array, ant_2_array, xp=cp)
    print(benchmark(gpu_grad_nll, (sim_gains, sim_data, 1, np.inf, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, ant_1_array, ant_2_array, cp), n_repeat=10))


    """ RUNNING WITH CORRCAL """
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
    print(benchmark(grad_nll, (gains_mat, cov, data_vec, ant_1_data, ant_2_data, 1, np.inf), n_repeat=10))


    """ TESTING WITH EXPLICIT CORRCAL - still need cov for this """
    # cpu_grad = cpu_grad_nll(gains_mat, cov, data_vec, ant_1_data, ant_2_data, scale=1, phs_norm_fac=np.inf)


    """ COMPARING OUTPUTS BTWN CPU AND GPU """
    gpu_grad_np = cp.asnumpy(gpu_grad)
    # print(gpu_grad_np)
    # print()
    # print(cpu_grad)

    print(np.allclose(gpu_grad_np, cpu_grad))

    #if wanting to return total number of files that coincide with corrcal (outdated)
    truth_check =  np.allclose(gpu_grad_np, cpu_grad)

    return gpu_grad_np - cpu_grad, cpu_grad, truth_check

    # print(cpu_grad.shape)
    # print(gpu_grad.shape)


if __name__ == "__main__":

    n_ant = 100
    results_n_ant = []
    for i in range(1):
        result, cpu_grad, truth = simulate(n_ant = n_ant, debugging=False)
    print(f" n_ant = {n_ant}")
    #     result = cp.asnumpy(result)
    #     results_n_ant.append(truth)
    #     # print(type(result))
    #     print(f"on trial {i}")
    #     plt.plot(result, marker='.', lw=0, ms=1)
    #     plt.xlabel("Number of Antennas (Re Im Split)")
    #     plt.ylabel(r"$\nabla \mathcal{L}_{gpu} - \nabla \mathcal{L}_{cpu}$")

    # plt.savefig('grad2_difference_for_nant={}.png'.format(n_ant), dpi=300, format='png', bbox_inches='tight')
    # plt.show()

    # # print(results_n_ant)

    # plt.plot(cp.asnumpy(results_n_ant), 'o')
    # plt.title(f"Number of antennas = {n_ant}")
    # plt.xlabel('N Trials')
    # plt.ylabel(r'Agreement with CPU $\nabla (-\text{log}\mathcal{L})$')
    # plt.savefig('grad2_truth_for_nant={}.png'.format(n_ant), dpi=300, format='png', bbox_inches='tight')
    # plt.show()



