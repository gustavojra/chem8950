import numpy as np
import time
import sys
from tools import *
from det import Determinant

def H_dif0(det1, h, ERI):

    alphas = det1.alpha_list()
    betas = det1.beta_list()

    hcore = np.einsum('mm,m->', h, alphas) + np.einsum('mm,m->', h, betas)

    # Compute J for all combinations of m n being alpha or beta
    JK  = np.einsum('mmnn, m, n', ERI, alphas, alphas, optimize = 'optimal')\
        + np.einsum('mmnn, m, n', ERI, betas, betas, optimize = 'optimal')  \
        + np.einsum('mmnn, m, n', ERI, alphas, betas, optimize = 'optimal') \
        + np.einsum('mmnn, m, n', ERI, betas, alphas, optimize = 'optimal')
    # For K m and n have to have the same spin, thus only two cases are considered
    JK -= np.einsum('mnnm, m, n', ERI, alphas, alphas, optimize = 'optimal')
    JK -= np.einsum('mnnm, m, n', ERI, betas, betas, optimize = 'optimal')
    return 0.5 * JK + hcore

def H_dif4(det1, det2, h, ERI):
    phase = det1.phase(det2)
    alp1, bet1 = det1.set_subtraction_list(det2)
    alp2, bet2 = det2.set_subtraction_list(det1)
    if len(alp1) != len(alp2):
        return 0
    [o1,o2,o3,o4] = alp1 + bet1 + alp2 + bet2
    if len(alp1) == 1:
        return phase * (ERI[o1,o3,o2,o4])
    else:
        return phase * (ERI[o1,o3,o2,o4] - ERI[o1,o4,o2,o3])

    #[[o1, s1], [o2, s2]] = det1.exclusive(det2)
    #[[o3, s3], [o4, s4]] = det2.exclusive(det1)
    #if s1 == s3 and s2 == s4:
    #    J = ERI[o1, o3, o2, o4] 
    #else:
    #    J = 0
    #if s1 == s4 and s2 == s3:
    #    K = ERI[o1, o4, o2, o3]
    #else:
    #    K = 0
    #return phase * (J - K)

def H_dif2(det1, det2, h, ERI):
    # Use exclusive to return a list of alpha and beta orbitals present in the first det, but no in the second 
    [alp1, bet1] = det1.set_subtraction_list(det2)
    [alp2, bet2] = det2.set_subtraction_list(det1)
 #   if len(alp1) != len(alp2):  # Check if the different orbitals have same spin  # DONT THINK THIS IS NECESSARY
 #       return 0
    phase = det1.phase(det2)
    [o1, o2] = alp1 + bet1 + alp2 + bet2
    # For J, (mp|nn), n can have any spin. Two cases are considered then. Obs: det1.occ or det2.occ would yield the same result. When n = m or p J - K = 0
    J = np.einsum('nn, n->', ERI[o1,o2], det1.alpha_list()) + np.einsum('nn, n->', ERI[o1,o2], det1.beta_list()) 
    if len(alp1) > 0:
        K = np.einsum('nn, n->', ERI.swapaxes(1,3)[o1,o2], det1.alpha_list())
    else:
        K = np.einsum('nn, n->', ERI.swapaxes(1,3)[o1,o2], det1.beta_list())
    return phase * (h[o1,o2] + J - K)


# FUNCTION: Given a list of determinants, compute the Hamiltonian matrix

def get_Hamiltonian(dets, h, ERI):
        l = len(dets)
        H = np.zeros((l,l))
        for i,d1 in enumerate(dets):
            H[i,i] = H_dif0(d1, h, ERI)
        for i,d1 in enumerate(dets):
            for j,d2 in enumerate(dets):
                if j >= i:
                    break
                dif = d1 - d2
                if dif == 4:
                    H[i,j] = H_dif4(d1, d2, h, ERI)
                    H[j,i] = H[i,j]
                elif dif == 2:
                    H[i,j] = H_dif2(d1, d2, h, ERI)
                    H[j,i] = H[i,j]
        return H
