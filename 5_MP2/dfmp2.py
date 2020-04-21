import psi4
import numpy as np
import sys 
from scipy.linalg import fractional_matrix_power
sys.path.append('.')
from uhf import compute_uhf
from tools import *
import time
from somp2 import compute_SOMP2

def show_progress(timings, printif):

    if len(timings) > 0: 
        sys.stdout.write("\033[F"*6)

    tasks = """
    Building:
      \U0001F426  Spin-Orbital arrays            {} 
      \U0001F986  J^(-1/2) (P|Q)                 {}
      \U0001F989  (uv|P)                         {}
      \U0001F9A2   b(ia|Q)                        {}"""

    x = []
    for t in timings:
        x.append(emoji('check') + '   ' + '{:10.5f} seconds'.format(t))

    while len(x) < 4:
        x.append('')

    printif(tasks.format(*x))

def compute_DFMP2(Settings, silent=False):

    t0 = time.time()

    printif = print if not silent else lambda *k, **w: None

    Escf, Ca, Cb, epsa, epsb, = compute_uhf(Settings, return_C=True, return_integrals=False)
    
    printif("""
    ===================================================
               Density-Fitting Spin-Orbital
            Møller-Plesset Perturbation Theory
                           {}
    ===================================================
    """.format('\U0001F347' + '\U000027A1' + emoji('wine')))

    tasks = """
    Building:
      \U0001F426  Spin-Orbital arrays            {} 
      \U0001F986  J^(-1/2) (P|Q)                 {}
      \U0001F989  (uv|P)                         {}
      \U0001F9A2   b(ia|Q)                        {}"""

    printif(tasks.format('','','',''))

    # Create Spin-Orbital arrays
    tsave = []
    t = time.time()
    C = np.block([
        [Ca,                 np.zeros(Ca.shape)],
        [np.zeros(Cb.shape), Cb]
        ]) 

    eps = np.concatenate((epsa, epsb)) 
    nmo = len(eps)

    ## Sorting orbitals
    s = np.argsort(eps)
    eps = eps[s]
    C = C[:,s]

    tsave.append(time.time() -t)
    show_progress(tsave, printif)

    # Create slices
    nelec = Settings['nalpha'] + Settings['nbeta']
    o = slice(0, nelec)
    v = slice(nelec, nmo)
    
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

    # Get b(iaP)
    
    t = time.time()
    biaQ = np.einsum('ui, va, uvP, PQ -> iaQ', C[:,o], C[:, v], uvP, Jinvs, optimize='optimal')
    print(biaQ)
    tsave.append(time.time()-t)
    show_progress(tsave, printif)

    # Get MP2 energy

    printif('\n{}  Computing DF-MP2 energy'.format('\U0001F9ED'), end= ' ')
    t = time.time()
    Emp2 = 0
    new = np.newaxis
    eo = eps[o]
    ev = eps[v]
    for i in range(nelec):
        for j in range(nelec):
            D = -ev[:, new] - ev[new, :]
            D = D + eo[i] + eo[j] 
            D = 1.0/D
            bAB = np.einsum('aQ, bQ-> ab', biaQ[i,:,:], biaQ[j,:,:])
            bBA = np.einsum('bQ, aQ-> ab', biaQ[i,:,:], biaQ[j,:,:])
            Emp2 += np.einsum('ab, ab->', np.square(bAB-bBA), D, optimize='optimal')

    Emp2 = Emp2/4.0

    tsave.append(time.time()-t)
    printif('\n{} DF-MP2 Correlation Energy:    {:>16.10f}'.format(emoji('bolt'), Emp2))
    printif('\U0001F308 Final Electronic Energy:      {:>16.10f}'.format(Emp2+Escf))
    printif('\n{} Total Computation time: {:10.5f} seconds'.format('\U0000231B',time.time() - t0))



    # Compare results with SO-MP2

    printif('\n   \U0001F4C8 Comparing results with regular MP2:')

    t0 = time.time()
    E0mp2 = compute_SOMP2(Settings, psi4compare=False, silent=True)
    printif('\n{} MP2 Correlation Energy:    {:>16.10f}'.format(emoji('bolt'), E0mp2))
    printif('{} DF error:                  {:>16.10f}'.format('\U0000274C', E0mp2-Emp2))
    printif('\n{} Total Computation time: {:10.5f} seconds'.format('\U0000231B',time.time() - t0))


    # Compare with Psi4

    psi4.set_options({'basis': Settings['basis'],
                      'df_basis_mp2' : Settings['df_basis'],
                      'e_convergence' : 1e-12, 
                      'scf_type': 'pk',
                      'puream'   : False,
                      'reference': 'uhf'})

    psi4_mp2 = psi4.energy('mp2')
    printif('\n\n{} Psi4  DF-MP2 Energy:          {:>16.10f}'.format(emoji('eyes'), psi4_mp2))

if __name__ == '__main__':

    from input import Settings
    compute_DFMP2(Settings)
