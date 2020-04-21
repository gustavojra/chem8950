import psi4
import numpy as np
import sys 
from scipy.linalg import fractional_matrix_power
sys.path.append('.')
from uhf import compute_uhf
from tools import *
import time


def compute_DFMP2(Settings):
    Escf, Ca, Cb, epsa, epsb, = compute_uhf(Settings, return_C=True, return_integrals=False)
    
    print("""
    ===================================================
      Density-Fitting Møller-Plesset Perturbation Theory
                           {}
    ===================================================
    """.format(emoji('wine')))

    tasks = """
    Building:
      \U0001F426  Spin-Orbital arrays               {}
      \U0001F986  MO integral <OO|VV>               {}
      \U0001F9A2   Eigenvalues auxiliar array Dijab  {}"""

    print(tasks.format('','',''))
    sys.stdout.write("\033[F"*5)
    time.sleep(0.1)

    # Create Spin-Orbital arrays
    #C = np.block([
    #    [Ca,                 np.zeros(Ca.shape)],
    #    [np.zeros(Cb.shape), Cb]
    #    ]) 

    #eps = np.concatenate((epsa, epsb)) 
    eps = epsa
    nmo = len(eps)

    # re-Sorting orbitals
    #s = np.argsort(eps)
    #eps = eps[s]
    #C = C[:,s]

    print(tasks.format(emoji('check'),'',''))

    # Create slices
    nelec = Settings['nalpha'] + Settings['nbeta']
    ndocc = int(nelec/2)
    nmo = len(eps)
    nvir = nmo - ndocc
    o = slice(0, ndocc)
    v = slice(ndocc, nmo)
    
    # Build J^(-1/2) using Psi4

    molecule = psi4.geometry(Settings['molecule'])
    basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'], puream=0)
    dfbasis = psi4.core.BasisSet.build(molecule, 'DF_BASIS_MP2', Settings['df_basis'], puream=0)
    mints = psi4.core.MintsHelper(basis)
    zero = psi4.core.BasisSet.zero_ao_basis_set()

    Jinvs = np.squeeze(mints.ao_eri(dfbasis, zero, dfbasis, zero).np)
    Jinvs = psi4.core.Matrix.from_array(Jinvs)
    Jinvs.power(-0.5, 1.e-14)
    Jinvs = Jinvs.np

    # Get integrals uvP

    uvP = np.squeeze(mints.ao_eri(basis, basis, dfbasis, zero).np)

    #fh = slice(0, int(nmo/2))
    #sh = slice(int(nmo/2), nmo)

    #uvP = np.zeros((nmo, nmo, ao_uvP.shape[2]))
    #uvP[fh, fh, :] = ao_uvP
    #uvP[sh, sh, :] = ao_uvP

    #uvP = uvP[s,:,:]
    #uvP = uvP[:,s,:]
    #print(uvP.shape)

    # Get b(iaP)

    # b(uvQ)
    biaQ = np.einsum('uvP, PQ-> uvQ', uvP, Jinvs)

    # b(ivQ)
    biaQ = np.einsum('ui, uvQ-> ivQ', Ca[:,o], biaQ)

    # b(iaQ)
    biaQ = np.einsum('va, ivQ-> iaQ', Ca[:,v], biaQ)


    # Get eigenvalues Matrix D
    new = np.newaxis
    eo = eps[o]
    ev = eps[v]
    D = 1.0/(eo[:, new, new, new] - ev[new, :, new, new] + eo[new, new, :, new] - ev[new, new, new, :])

    # Get MP2 energy

    print('\n{}  Computing MP2 energy'.format('\U0001F9ED'), end= ' ')
    Emp2 = 0
    for i in range(ndocc):
        for j in range(ndocc):
            bAB = np.einsum('aQ, bQ-> ab', biaQ[i,:,:], biaQ[j,:,:])
            bBA = np.einsum('bQ, aQ-> ab', biaQ[i,:,:], biaQ[j,:,:])
            Emp2 += np.einsum('ab, ab, ab->', bAB, (2*bAB-bBA), D[i,:,j,:], optimize='optimal')


    print('\n{} MP2 Correlation Energy:    {:>16.10f}'.format(emoji('bolt'), Emp2))
    print('\U0001F308 Final Electronic Energy:   {:>16.10f}'.format(Emp2+Escf))

    psi4.set_options({'basis': Settings['basis'],
                      'df_basis_mp2' : Settings['df_basis'],
                      'e_convergence' : 1e-12, 
                      'scf_type': 'pk',
                      'puream'   : False,
                      'reference': 'uhf'})

    psi4_mp2 = psi4.energy('mp2')
    print('\n\n{} Psi4  MP2 Energy:          {:>16.10f}'.format(emoji('eyes'), psi4_mp2))

if __name__ == '__main__':

    from input import Settings
    compute_DFMP2(Settings)
