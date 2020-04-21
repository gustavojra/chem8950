import psi4
import numpy as np
import sys 
sys.path.append('.')
from uhf import compute_uhf
from tools import *
import time
import copy

def show_progress(timings, printif):

    # Function to display the progress of computation setup

    if len(timings) > 0 and printif == print:
        sys.stdout.write("\033[F"*8)

    tasks = """
    Building:
      \U0001F426  Spin-Orbital arrays            {} 
      \U0001F986  MO integrals                   {}
      \U0001F983  J^(-1/2) (P|Q)                 {}
      \U0001F989  (uv|P)                         {}
      \U0001F427  b(ab|Q)                        {}
      \U0001F9A2   D[ijab]                        {}"""

    x = []
    for t in timings:
        x.append(emoji('check') + '   ' + '{:10.5f} seconds'.format(t))

    while len(x) < 6:
        x.append('')

    printif(tasks.format(*x))

def CEPA_Energy(T, V):

    return (1/4)*np.einsum('mncd, mncd -> ', V, T, optimize='optimal')

def CEPA_Amplitude(T, D, Voovv, Voooo, babQ, Vvoov):

    p = range(T.shape[3])

    newT = np.zeros(T.shape)

    newT += Voovv

    # DF-<vv||vv> part: ([cd|db] - [cbda])*t(ij,cd)
    for a in p:
        for b in p:
            bX = np.einsum('cQ, dQ -> cd', babQ[:,a,:], babQ[:,b,:], optimize='optimal')
            bY = np.einsum('cQ, dQ -> cd', babQ[:,b,:], babQ[:,a,:], optimize='optimal')
            newT[:,:,a,b] += 0.5*np.einsum('cd, ijcd-> ij', bX-bY, T, optimize='optimal')

    newT += 0.5*np.einsum('ijmn, mnab -> ijab', Voooo, T, optimize='optimal')

    X = np.einsum('cjmb, imac -> ijab', Vvoov, T, optimize='optimal')

    newT += X - X.transpose(1,0,2,3) - X.transpose(0,1,3,2) + X.transpose(1,0,3,2)

    newT *= D

    rms = np.sqrt(np.sum(np.square(newT - T)))/(T.size)

    return newT, rms

def compute_dfCEPA(Settings, silent=False, compare=False):

    t0 = time.time()

    printif = print if not silent else lambda *k, **w: None

    Escf, Ca, Cb, epsa, epsb, _, g, Vnuc = compute_uhf(Settings, return_C=True, return_integrals=True)
    
    printif("""
    =======================================================
                           CEPA0
      (For those who havent heard Coupled Cluster exists)
                             {}
    =======================================================
    """.format('\U0001F474'))

    tsave = []
    show_progress(tsave, printif)

    # Create Spin-Orbital arrays
    t = time.time()
    C = np.block([
        [Ca,                 np.zeros(Ca.shape)],
        [np.zeros(Cb.shape), Cb]
        ]) 

    eps = np.concatenate((epsa, epsb)) 

    g = np.kron(np.eye(2), g)
    g = np.kron(np.eye(2), g.T)

    # re-Sorting orbitals
    s = np.argsort(eps)
    eps = eps[s]
    C = C[:,s]

    tsave.append(time.time() - t)
    show_progress(tsave, printif)
    
    # Get the MO integral

    t = time.time()
    nelec = Settings['nalpha'] + Settings['nbeta']
    # Converto to Physicists notation
    g = g.transpose(0,2,1,3)
    # Antisymmetrize
    g = g - g.transpose(0,1,3,2)

    # Save Slices
    ERI = lambda ax,bx,cx,dx: np.einsum('ap,bq,cr,ds,abcd->pqrs', C[:,ax], C[:,bx], C[:,cx], C[:,dx], g, optimize='optimal')
    nmo = len(g)
    o = slice(0, nelec)
    v = slice(nelec, nmo)
    Voovv = ERI(o,o,v,v)
    Voooo = ERI(o,o,o,o)
    Vvoov = ERI(v,o,o,v)

    tsave.append(time.time() - t)
    show_progress(tsave, printif)

    # Use Density-Fitting for the Vvvvv case
    # Build J^(-1/2) using Psi4
    t = time.time()
    molecule = psi4.geometry(Settings['molecule'])
    basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'], puream=0)
    dfbasis = psi4.core.BasisSet.build(molecule, 'DF_BASIS_MP2', Settings['df_basis'], puream=0)
    mints = psi4.core.MintsHelper(basis)
    zero = psi4.core.BasisSet.zero_ao_basis_set()

    Jinvs = np.squeeze(mints.ao_eri(dfbasis, zero, dfbasis, zero).np)
    Jinvs = psi4.core.Matrix.from_array(Jinvs)
    Jinvs.power(-0.5, 1.e-14)
    Jinvs = Jinvs.np

    tsave.append(time.time()-t)
    show_progress(tsave, printif)

    # Get integrals uvP

    t = time.time()
    ao_uvP = np.squeeze(mints.ao_eri(basis, basis, dfbasis, zero).np)

    fh = slice(0, int(nmo/2))
    sh = slice(int(nmo/2), nmo)

    ## Build spin-blocks
    uvP = np.zeros((nmo, nmo, ao_uvP.shape[2]))
    uvP[fh, fh, :] = ao_uvP
    uvP[sh, sh, :] = ao_uvP

    tsave.append(time.time()-t)
    show_progress(tsave, printif)

    # Get b(abP)
    
    t = time.time()
    babQ = np.einsum('ua, vb, uvP, PQ -> abQ', C[:,v], C[:, v], uvP, Jinvs, optimize='optimal')
    tsave.append(time.time()-t)
    show_progress(tsave, printif)

    # Get eigenvalues Matrix D
    t = time.time()
    new = np.newaxis
    eo = eps[:nelec]
    ev = eps[nelec:]
    D = 1.0/(eo[:, new, new, new] + eo[new, :, new, new] - ev[new, new, :, new] - ev[new, new, new, :])

    tsave.append(time.time() - t)
    show_progress(tsave, printif)
    t = time.time()

    # Initial guess for amplitudes

    T = Voovv*D

    # Get MP2 energy

    E = CEPA_Energy(T, Voovv)

    print('\nMP2 Energy:   {:<15.10f}\n'.format(E + Escf))

    # Setup iteration options
    rms = 0.0
    dE = 1
    ite = 1
    rms_LIM = 10**(-8)
    E_LIM = 10**(-12)
    t0 = time.time()
    printif('\U00003030'*20)

    # Start CC iterations
    while abs(dE) > E_LIM or rms > rms_LIM:
        t = time.time()
        if ite > Settings["cc_max_iter"]:
            raise NameError('CEPA0 equations did not converge')
        T, rms = CEPA_Amplitude(T, D, Voovv, Voooo, babQ, Vvoov)
        dE = -E
        E = CEPA_Energy(T, Voovv)
        dE += E
        printif("Iteration  {}".format(numoji(ite)))
        printif("\U000027B0 Correlation energy:     {:< 15.10f}".format(E))
        printif("\U0001F53A Energy change:          {:< 15.10f}".format(dE))
        printif("\U00002622  Max RMS residue:        {:< 15.10f}".format(rms))
        printif("\U0001F570  Time required:          {:< 15.10f}".format(time.time() - t))
        printif('\U00003030'*20)
        ite += 1

    printif('\U0001F3C1 DF-CEPA0 Energy:    {:<15.10f}'.format(E + Escf))
    printif('\U000023F3 CEPA0 iterations took %.2f seconds.\n' % (time.time() - t0))

    if compare:

        from CEPA import compute_CEPA

        printif('\n   Running regular CEPA...\n')

        t0 = time.time()
        Efull = compute_CEPA(Settings, silent=True, compare=False)
        printif('\U0001F3C1 CEPA0 Energy:    {:<15.10f}'.format(Efull))
        printif('\U000023F3 CEPA0 iterations took %.2f seconds.\n' % (time.time() - t0))

        printif('\n\U0000274C Density Fitting Error:   {:<15.10f}'.format(E+Escf - Efull))


    return (E + Escf)


if __name__ == '__main__':

    from input import Settings
    compute_dfCEPA(Settings, compare=True)
