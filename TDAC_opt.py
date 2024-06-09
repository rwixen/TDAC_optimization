"""
Written by Ryan Wixen, June 2024
following the routine described in
    R. D. Koilpillai and P. P. Vaidyanathan. Cosine-modulated FIR filter banks
    satisfying perfect reconstruction. IEEE Transactions on Signal Processing,
    40(4):770–783, April 1992.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Optimization routine

def min_stopband_energy_TDAC_window(m, M, omega_s):
    """
    Optimize a protototype filter for an extended TDAC transform.
    M is the number channels of the transform.
        M must be greater than 2 and is typically less than or equal to about 2048.
    m is the extenstion factor such that 2 * M * m is the length of the filter.
        m must be at least 1 and is typically less than about 4.
    omega_s is the cutoff frequency beyond which the filter's energy is minimized.
    
    Returns:
        p0, the optimized prototype filter
        betas, the lattice coefficients for each pair of polyphase components
    """
    if m <= 3:
        betas = np.block([1 / np.sqrt(2) * np.ones((M // 2, 1)), np.zeros((M // 2, m - 1))])
    else:
        _, warmbetas = min_stopband_energy_TDAC_window(m - 1, M, omega_s)
        betas = np.block([warmbetas, np.zeros((M // 2, 1))])
    
    def obj_and_grad(betas, omega_s, P, m, M):
        betas = np.reshape(betas, (M // 2, m))
        gis = gis_from_betas(betas, m, M)
        p0 = p0_from_gis(gis, m, M)
        return stopband_energy(p0, omega_s), grad_phi_1_wrt_betas(betas, p0, P, m, M)
    
    P = calc_P(omega_s, m, M)
    args = (omega_s, P, m, M)
    optresult = sp.optimize.minimize(obj_and_grad, betas.flatten(), args, method="BFGS", jac=True)
    betas = np.reshape(optresult.x, (M // 2, m))
    gis = gis_from_betas(betas, m, M)
    p0 = p0_from_gis(gis, m, M)
    
    return p0, betas

# Objective function

def stopband_energy(p0, omega_s):   
    """
    Returns the integral of the magnitude squared of the frequency respose of p0 from omega_s to pi.
    """
    N = p0.size - 1
    r = np.convolve(p0, np.flip(p0), "full")[N:]
    k = np.arange(1, N + 1)
    phi1 = (np.pi - omega_s) * r[0] - 2 * np.sum(r[k] * np.sin(k * omega_s) / k)
    return phi1

# Computing the prototype filter, p0, from the lattice coefficients, betas

def p0_from_gis(gis, m, M):
    """
    Compute the prototype filter p0 from its 2M polyphase components gis.
    """
    p0 = np.zeros(2 * m * M)
    for i in range(2 * M):
        p0[i::2 * M] = gis[i]
    return p0

def gis_from_betas(betas, m, M):
    """
    Compute the set of 2M polyphase components from the set of M // 2 * m lattice coefficients.
    """
    gis = np.empty((2 * M, m))
    for i in range(M // 2):
        gis[i, :], gis[i + M, :] = gi_and_giplusM_from_beta(betas[i, :], m, normalize=True)
        gis[2 * M - 1 - i, ::-1], gis[M - 1 - i, ::-1] = gis[i, :], gis[i + M, :] # linear phase
    if M % 2 != 0:
        gis[M // 2, :], gis[M // 2 + M, :] = np.zeros(m), np.zeros(m)
        gis[M // 2, m // 2], gis[M // 2 + M, m - 1 - m // 2] = np.sqrt(.5), np.sqrt(.5)
    return gis

def gi_and_giplusM_from_beta(beta, m, normalize):
    """
    Compute the a pair of polyphase components of length m from a set of m lattice coefficients.
    normalize is a bool that determines whether the polyphase components are scaled to have unity gain.
    """
    gi = np.zeros(m)
    giplusM = np.zeros(m)
    gi[0] = beta[0]
    giplusM[0] = 1
    for i in range(1, m):
        # roll the non-empty indices
        giplusM[1:i + 1] = giplusM[:i]
        giplusM[0] = 0
        
        gi[:i + 1], giplusM[:i + 1] = beta[i] * gi[:i + 1] + giplusM[:i + 1], gi[:i + 1] + -beta[i] * giplusM[:i + 1]
    if normalize:
        alpha = calc_alpha(beta)
        gi *= alpha
        giplusM *= alpha
    return gi, giplusM

def calc_alpha(beta):
    """
    Compute the scale factor to apply to a pair of polyphase components calculated from
    a set of lattice coefficients such that the polyphase components have unity gain.
    """
    return np.prod(1 / np.sqrt(1 + beta ** 2))

# Computing the gradient of the objective, phi1, with respect to the lattice coefficients, betas

def grad_phi_1_wrt_betas(betas, p0, P, m, M):
    """
    Compute the gradient of the stopband energy with respect to the lattice coefficients.
    """
    j_phi1_wrt_p0 = jac_phi1_wrt_p0_linphase(p0, P, m, M)
    jacs = np.empty((M // 2, m))
    for i in range(M // 2):
        jacs[i, :] = jac_phi1_wrt_betai_linphase(i, betas, m, M, j_phi1_wrt_p0)
    return jacs.flatten()

def jac_phi1_wrt_betai_linphase(i, betas, m, M, j_phi1_wrt_p0):
    """
    Compute the gradient of the stopband energy with respect to the ith set of lattice coefficients, betas[i].
    """
    j_gi_wrt_betai, j_giplusM_wrt_betai = jac_normalized_gi_and_giplusM_wrt_betai(i, betas, m)
    # For linear phase
    j_gminusi_wrt_betai, j_gminusiplusM_wrt_betai = np.flip(j_gi_wrt_betai), np.flip(j_giplusM_wrt_betai)


    j_phi1_wrt_betai = 2 * (j_gi_wrt_betai @ jac_phi1_wrt_gi(j_phi1_wrt_p0, i, m, M) \
                                    + j_giplusM_wrt_betai @ jac_phi1_wrt_gi(j_phi1_wrt_p0, i + M, m, M))
    # Due to linear phase, the preceding statement is equivalent to the following.
    # j_phi1_wrt_betai = j_gi_wrt_betai @ jac_phi1_wrt_gi(j_phi1_wrt_p0, i, m, M) \
    #                     + j_giplusM_wrt_betai @ jac_phi1_wrt_gi(j_phi1_wrt_p0, i + M, m, M) \
    #                     + np.flip(j_gminusi_wrt_betai @ jac_phi1_wrt_gi(j_phi1_wrt_p0, 2 * M - 1 - i, m, M) \
    #                     + j_gminusiplusM_wrt_betai @ jac_phi1_wrt_gi(j_phi1_wrt_p0, M - 1 - i, m, M))
    
    return j_phi1_wrt_betai

def jac_phi1_wrt_p0_linphase(p0, P, m, M):
    """
    Compute the gradient of the stopband energy p0.T @ P @ p0 with respect to the prototype filter, p0.
    """
    section = 2 * P[:m * M, :] @ p0
    return np.concatenate((section, np.flip(section)))


def calc_P(omega_s, m, M):
    """
    Calculate the matrix P such that p0.T @ P @ p0 = stopband_energy(p0, omega_s)
    """
    L = 2 * m * M
    P = (np.pi - omega_s) * np.eye(L)
    for k in range(1, L):
        P[np.arange(k, L), np.arange(0, L - k)] -= np.sin(k * omega_s) / k
        P[np.arange(0, L - k), np.arange(k, L)] -= np.sin(k * omega_s) / k
    return P

def jac_phi1_wrt_gi(j_phi1_wrt_p0, i, m, M):
    """
    Compute the gradient of the stopband energy with respect to the ith polyphase component.
    """
    return jac_p0_wrt_gi_times(j_phi1_wrt_p0, i, m, M)

def jac_p0_wrt_gi_times(x, i, m, M):
    """
    Compute the product of the gradient of the prototype filter with respect to the ith polyphase component and x.
    """
    return x[i::2 * M]

def jac_normalized_gi_and_giplusM_wrt_betai(i, betas, m):
    """
    Compute the Jacobian the ith pair of polyphase components
    with respect to the ith set of lattice coefficients, betas[i].
    """
    beta = betas[i, :]
    alpha = calc_alpha(beta)
    gi, giplusM = gi_and_giplusM_from_beta(beta, m, normalize=False)
        
    dalpha_dbetai = -alpha * beta / (1 + np.square(beta))
                       
    j_gitilde, j_giplusMtilde = jac_unnormalized_gi_and_giplusM_wrt_betai(i, betas, m)
    j_gi = alpha * j_gitilde + np.outer(dalpha_dbetai, gi)
    j_giplusM = alpha * j_giplusMtilde + np.outer(dalpha_dbetai, giplusM)
    return j_gi, j_giplusM

def jac_unnormalized_gi_and_giplusM_wrt_betai(i, betas, m):
    """
    Compute the Jacobian the every pair of polyphase components
    with respect to the every set of lattice coefficients, betas.
    """
    beta = betas[i, :]
    jaci = np.zeros((m, m)) # (lattice index j, polyphase index k)
    jaciplusM = np.zeros((m, m))
    jaci[0, 0] = 1
    jaci[1:, 0] = beta[0]
    jaciplusM[1:, 0] = 1
    for section_ind in range(1, m):
        # roll the non-empty indices
        jaciplusM[:, 1:section_ind + 1] = jaciplusM[:, :section_ind]
        jaciplusM[:, 0] = 0
        jaciplusM[section_ind, :section_ind + 1] = -jaciplusM[section_ind, :section_ind + 1]
        jinds = np.arange(m) != section_ind
        jaci[jinds, :section_ind + 1], jaciplusM[jinds, :section_ind + 1] = \
                beta[section_ind] * jaci[jinds, :section_ind + 1] + jaciplusM[jinds, :section_ind + 1], \
                jaci[jinds, :section_ind + 1] + -beta[section_ind] * jaciplusM[jinds, :section_ind + 1]
    return jaci, jaciplusM

# Utiltity functions

def plot_filter_and_frequency_response(p0, m, M):
    """
    A utility function to plot a filter and its frequency response.
    """
    plt.plot(p0)
    plt.title("Optimized Protoype Filter ($m = {}$,  $M = {}$)".format(m, M))
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.show()

    N = np.maximum(2 ** 12, (2 * m * M) * 8)
    freq_response = np.abs(np.fft.rfft(p0, N))[:N // 2 + 1] / M / np.sqrt(2)
    mag_response = 20 * np.log10(freq_response, where=freq_response != 0)
    mag_response[freq_response == 0] = -np.inf
    plt.figure()
    plt.plot(np.linspace(0, .5, N // 2 + 1), mag_response)
    plt.xlabel("Normalized Frequency ($\omega/2\pi$)")
    plt.ylabel(r"Scaled Magnitude Response (dB)")
    plt.title("Optimized Protoype Filter Magnitude Response ($m = {}$,  $M = {}$)".format(m, M))
    plt.xlim(0, .5)
    plt.ylim((-60, 10))
    plt.show()