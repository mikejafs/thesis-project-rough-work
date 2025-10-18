import healpy as hp
import numpy as np
import os
import astropy.units as u
import matplotlib.pyplot as plt
from pygdsm import GlobalSkyModel16
# %matplotlib inline

def compute_cov_corr_mat(
        freqs,
        sky_maps,
):
    nside = 1024

    #prepare the corresponding alms
    lmax = 3*nside - 1
    alm_list = []
    for i in range(len(freqs)):
        alm = hp.map2alm(sky_maps[i], lmax = lmax)
        alm_list.append(alm)
    alm_list = np.array(alm_list)

    #compute the covariance matrix
    cl_cov_mat = np.zeros((len(freqs), len(freqs), lmax+1))
    for i in range(len(freqs)):
        for j in range(len(freqs)):
            cl_cov_mat[i, j, :] = hp.alm2cl(alm_list[i], alm_list[j])

    #compute the correlation matrix using the covariance matrix
    corr_mat = 0*cl_cov_mat[:, :, :]
    for i in range(len(freqs)):
        for j in range(len(freqs)):
            corr_mat[i, j, :] = cl_cov_mat[i, j, :]/np.sqrt(
                cl_cov_mat[i, i, :]*cl_cov_mat[j, j, :]
                )

    return cl_cov_mat, corr_mat

def plot_cov_corr(
        cov_mat,
        corr_mat,
        freqs,
        ell,
        return_eigs=False
):
    #plot the covariance and correlation matrices side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()

    #covariance subplot
    im_cov = axes[0].imshow(cov_mat[:,:,ell], origin='upper', 
                        extent=([
                            freqs[0], freqs[-1], freqs[0], freqs[-1]
                                 ]))
    axes[0].set_title(r"Covariance, $\ell=${}".format(ell))
    axes[0].set_xlabel("frequency (MHz)")
    axes[0].set_ylabel("frequency (MHz)")
    fig.colorbar(im_cov, ax=axes[0], fraction=0.046, pad=0.04)

    #covariance subplot
    im_corr = axes[1].imshow(corr_mat[:,:,ell], origin='upper', 
                        extent=([
                            freqs[0], freqs[-1], freqs[0], freqs[-1]
                                 ]))
    axes[1].set_title(r"Correlation Coefficient, $\ell=${}".format(ell))
    axes[1].set_xlabel("frequency (MHz)")
    axes[1].set_ylabel("frequency (MHz)")
    fig.colorbar(im_corr, ax=axes[1], fraction=0.046, pad=0.04)

    #also return the eigenvalues describing the correlation matrix
    if return_eigs:
        e_val, e_vec = np.linalg.eigh(corr_mat[:, :, ell]) 
        print("Eigenvals: \n", e_val/e_val[-1])