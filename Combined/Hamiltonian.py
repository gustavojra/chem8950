import numpy as np
from det import Determinant

def H_dif0(det1, h, ERI):
    
    # Compute H matrix element between two identical determinants

    alphas = det1.alpha_list()
    betas = det1.beta_list()

    # One electron term
    hcore = np.einsum('mm,m->', h, alphas) + np.einsum('mm,m->', h, betas)

    # Two electron term
    # Compute J for all spin cases: [MM|NN] + [mm|nn] + [MM|nn] + [mm|NN] 
    # Note that, we can merge the last two terms: [MM|NN] + [mm|nn] + 2*[MM|nn]
    JK  =   np.einsum('mmnn, m, n', ERI, alphas, alphas, optimize = 'optimal')
    JK +=  np.einsum('mmnn, m, n', ERI, betas, betas, optimize = 'optimal')  
    JK += 2*np.einsum('mmnn, m, n', ERI, alphas, betas, optimize = 'optimal') 
    # For K: [MN|NM] + [mn|nm]
    JK -= np.einsum('mnnm, m, n', ERI, alphas, alphas, optimize = 'optimal')
    JK -= np.einsum('mnnm, m, n', ERI, betas, betas, optimize = 'optimal')

    return hcore + 0.5 * JK

def H_dif2(det1, det2, h, ERI):

    # Compute H matrix element between two determinants that differ by two spin-orbitals (Single excitation)

    # Get phase factor
    phase = det1.phase(det2)

    # Get alpha and beta lists
    alphas = det1.alpha_list()
    betas = det1.beta_list()

    # Use Set subtraction to get orbitals present in Det1, but not in Det2 and vice versa
    a1, b1 = det1.set_subtraction_list(det2)
    a2, b2 = det2.set_subtraction_list(det1)

    # Note, this is a list concatenation expression
    # Since there are maximum of two orbitals different, this operation yields two orbital indexes (m, and p)
    # If the number of alphas and betas is conserved, which is the case in this code, one does not need to worry about checking if the spin of m and p match
    [m, p] = a1 + b1 + a2 + b2

    # For J we have [~m~p|NN] + [~m~p|nn]
    # Note that, alphas and betas were created using det1. It would not matter if you had chosen det2. When n = m or p, this term cancells out with K.
    J  = np.einsum('nn, n->', ERI[m,p], alphas) 
    J += np.einsum('nn, n->', ERI[m,p], betas) 

    # For K we need to know the spin of m and p. So we check the length of a1, if it is non-zero m and p are alphas.
    # then for K we have [MN|NP] or [mn|np]. Note that swapaxes that was used so we can select m and p easily. 
    if len(a1) > 0:
        K = np.einsum('nn, n->', ERI.swapaxes(1,3)[m,p], alphas)
    else:
        K = np.einsum('nn, n->', ERI.swapaxes(1,3)[m,p], betas)

    # Return the final value, remember to include phase!
    return phase * (h[m,p] + J - K)

def H_dif4(det1, det2, h, ERI):

    # Compute H matrix element between two determinants that differ by four spin-orbitals (Double excitation)

    # Get phase factor
    phase = det1.phase(det2)

    # Use Set subtraction to get orbitals present in Det1, but not in Det2 and vice versa
    a1, b1 = det1.set_subtraction_list(det2)
    a2, b2 = det2.set_subtraction_list(det1)

    # Note, this is a list concatenation expression
    # Since there are maximum of four different orbitals, this operation yields four orbital indexes (m, n, p, q)
    # If the number of alphas and betas is conserved, which is the case in this code, one does not need to worry about checking if the spin of m and p match
    [m,n,p,q] = a1 + b1 + a2 + b2

    # To verify if m and n have the same spin we check the length of a1.
    if len(a1) == 1:
        # Case 1: m,p have different spin than n,q. Return J = [mp,NQ] or [MP,nq]
        return phase * (ERI[m,p,n,q])
    else:
        # Case 2: All indexes have the same spin. Return J - K = [MP|NQ] - [MQ|NP] or [mp|nq] - [mq|np]
        return phase * (ERI[m,p,n,q] - ERI[m,q,n,p])

def get_Hamiltonian(dets, h, ERI):

    # Given a list of determinants, and integrals, return Hamiltonian matrix

    l = len(dets)
    H = np.zeros((l,l))
    for i,d1 in enumerate(dets):
        # Get diagonal elements
        H[i,i] = H_dif0(d1, h, ERI)
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
