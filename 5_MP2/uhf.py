import psi4
import scipy
import sys 
sys.path.append('.')
from integrals import *
from tools import *
psi4.core.be_quiet()

#################################################################################################
##  _   _                  _                                      _____                  _     ##
## | | | |   __ _   _ __  | |_   _ __    ___    ___              |  ___|   ___     ___  | | __ ##
## | |_| |  / _` | | '__| | __| | '__|  / _ \  / _ \    _____    | |_     / _ \   / __| | |/ / ##
## |  _  | | (_| | | |    | |_  | |    |  __/ |  __/   |_____|   |  _|   | (_) | | (__  |   <  ##
## |_| |_|  \__,_| |_|     \__| |_|     \___|  \___|             |_|      \___/   \___| |_|\_\ ##
#################################################################################################

def contamination(C_a, C_b, S, nalpha, nbeta):

    # Function to evaluate spin contamination
    
    Smetric = np.einsum('ui,uv,vj->ij', C_a[:,:nalpha], S, C_b[:,:nbeta])
    return min(nalpha, nbeta) - np.vdot(Smetric, Smetric)

def compute_uhf(Settings, return_C = False, return_integrals = False):

    # READ INPUT FILE

    print('\nReading input...', end=' ')
    molecule = psi4.geometry(Settings['molecule'])
    molecule.update_geometry()
    basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'], puream=0)
    mints = psi4.core.MintsHelper(basis)
    
    nalpha = Settings['nalpha']
    nbeta = Settings['nbeta']
    Vnuc = molecule.nuclear_repulsion_energy()
    
    scf_max_iter = Settings['scf_max_iter']
    print(emoji('check'))
    
    # COMPUTE INTEGRALS
    
    # Collect basis function Info from Psi4 for and save into my BasisFunction objects
    bset = BasisSet(name=Settings['basis'])
    for ishell in range(basis.nshell()):
        shell = basis.shell(ishell)
        center = [molecule.x(shell.ncenter), molecule.y(shell.ncenter), molecule.z(shell.ncenter)]
        exp = []
        c   = []
        for i in range(shell.nprimitive):
            exp.append(shell.exp(i))
            c.append(shell.coef(i))
        b = BasisFunction(center, c, exp, shell.am)
        bset.add(b)
    
    print('Computing integrals')
    
    print(bset)
    ## Overlap
    
    S = overlap(bset)
    pS = mints.ao_overlap().np
    
    ## Kinetic
    
    T = kinetic(bset)
    
    ## Nuclei-electron potential energy (From Psi4)
    pT = mints.ao_kinetic().np
    
    ## Nuclei-electron potential energy
    V = mints.ao_potential().np
    
    ## Electron repulsion integral (From Psi4)
    
    ERI = mints.ao_eri().np
    
    # FORM ONE-ELECTRON HAMILTONIAN
    
    h = T + V
    print('Done ' + emoji('check'))
    
    # CONSTRUCT THE ORTHOGONALIZER
    
    print('Initial guess...', end=' ')
    A = scipy.linalg.fractional_matrix_power(S, -0.5)
    
    # INITIAL DENSITY MATRIX
    
    Ft_a = A.dot(h).dot(A)
    _,Ct_a = np.linalg.eigh(Ft_a)
    C_a = A.dot(Ct_a)
    D_a = np.einsum('um,vm-> uv', C_a[:,:nalpha], C_a[:,:nalpha])
    D_b = np.einsum('um,vm-> uv', C_a[:,:nbeta], C_a[:,:nbeta])   # Using C alpha, since in this guess C_a = C_b
    
    G = ERI - ERI.swapaxes(1,2)
    F_a = h + np.einsum('ps,uvps', D_a, G) + np.einsum('ps,uvps', D_b, ERI)
    F_b = h + np.einsum('ps,uvps', D_b, G) + np.einsum('ps,uvps', D_a, ERI)
    
    E = 0.5*(np.einsum('uv,uv->', D_a + D_b, h) + np.einsum('uv,uv->', D_a, F_a) + np.einsum('uv,uv->', D_b, F_b)) + Vnuc
    print(emoji('check'))
    # START SCF ITERATIONS
    
    rms_a = 1
    rms_b = 1
    dE = 1
    ite = 1
    
    # Convergency criteria
    
    dE_max = 1.e-12
    rms_max = 1.e-8
    
    print('\nStarting SCF iterations' + emoji('cycle'))
    print('Convergency criteria:')
    print('{} Max Energy Diff:  {:<4.2E}'.format(emoji('pin'),dE_max))
    print('{} Max RMS density:  {:<4.2E}'.format(emoji('pin'),rms_max))
    print('='*40)
    
    while abs(dE) > dE_max or abs(rms_a) > rms_max or abs(rms_b) > rms_max:
        dE = -E
        rms_a = D_a[:,:]
        rms_b = D_b[:,:]
        if ite > scf_max_iter:
            raise NameError('SCF cannot converge {c} {c} {c}'.format(c=emoji('crying')))
        
        Ft_a = A.dot(F_a).dot(A)
        Ft_b = A.dot(F_b).dot(A)
        eps_a, Ct_a = np.linalg.eigh(Ft_a)
        eps_b, Ct_b = np.linalg.eigh(Ft_b)
        C_a = A.dot(Ct_a)
        C_b = A.dot(Ct_b)
        D_a = np.einsum('um,vm-> uv', C_a[:,:nalpha], C_a[:,:nalpha])
        D_b = np.einsum('um,vm-> uv', C_b[:,:nbeta], C_b[:,:nbeta]) 
        F_a = h + np.einsum('ps,uvps', D_a, G) + np.einsum('ps,uvps', D_b, ERI)
        F_b = h + np.einsum('ps,uvps', D_b, G) + np.einsum('ps,uvps', D_a, ERI)
        E = 0.5*(np.einsum('uv,uv->', D_a + D_b, h) + np.einsum('uv,uv->', D_a, F_a) + np.einsum('uv,uv->', D_b, F_b)) + Vnuc
        dE += E
        rms_a = np.sqrt(np.sum(np.square(D_a-rms_a)))
        rms_b = np.sqrt(np.sum(np.square(D_b-rms_b)))
        print('Iteration {}'.format(numoji(ite)))
        print('SCF Energy:         {:>16.12f}'.format(E))
        print('Energy change:      {:>16.12f}'.format(dE))
        print('MAX rms change:     {:>16.12f}'.format(max(rms_a,rms_b)))
        print('='*40)
        ite += 1
    
    print('SCF Converged ' + 2*emoji('viva'))
    print('{} Final SCF Energy:   {:>16.12f}'.format(emoji('bolt'),E))
    print('{} Spin-contamination  {:>16.12f}'.format(emoji('ugh'), contamination(C_a, C_b, S, nalpha, nbeta)))
    
    psi4.set_options({'basis': Settings['basis'],
                      'scf_type': 'pk',
                      'e_convergence': dE_max,
                      'puream'   : False,
                      'reference': 'uhf'})
    
    # Compare to Psi4
    E_psi4, wfn_psi4 = psi4.energy('scf', return_wfn=True)
    print('{} Psi4  SCF Energy:   {:>16.12f}'.format(emoji('eyes'),E_psi4))
    
    # Compute Dipole moment
    au2debye = 1/0.393430307
    
    # Get AO dipole moment
    Dx, Dy, Dz = dipole(bset)
    nucdip = molecule.nuclear_dipole()
    
    # Get final dipole moments for the wavefunction using density matrices
    dipole_x = np.einsum('uv,uv->', D_a, Dx) + np.einsum('uv,uv->', D_b, Dx) + nucdip[0]
    dipole_y = np.einsum('uv,uv->', D_a, Dy) + np.einsum('uv,uv->', D_b, Dy) + nucdip[1]
    dipole_z = np.einsum('uv,uv->', D_a, Dz) + np.einsum('uv,uv->', D_b, Dz) + nucdip[2]
    total_dipole = np.array([dipole_x, dipole_y, dipole_z])
    
    print('\n{} Dipole Moment (atomic units):'.format(emoji('plug')))
    print('{:^8}  {:^8}  {:^8}'.format('X', 'Y', 'Z'))
    print('{:^8.4f}  {:^8.4f}  {:^8.4f}'.format(*total_dipole))
    print(' Magnitude: {:< 8.4f}'.format(np.linalg.norm(total_dipole)))
    print('\n{} Dipole Moment (Debye):'.format(emoji('plug')))
    print('{:^8}  {:^8}  {:^8}'.format('X', 'Y', 'Z'))
    print('{:^8.4f}  {:^8.4f}  {:^8.4f}'.format(*total_dipole*au2debye))
    print(' Magnitude: {:< 8.4f}'.format(np.linalg.norm(total_dipole*au2debye)))

    if return_C and return_integrals:
        return (E, C_a, C_b, eps_a, eps_b, h, ERI, Vnuc)
    elif return_C:
        return (E, C_a, C_b, eps_a, eps_b)
    elif return_integrals:
        return (E, h, ERI, Vnuc)
    else:
        return E

if __name__ == '__main__':
    
    from input import Settings
    compute_uhf(Settings)
