import psi4
import numpy as np
import sys 
sys.path.append('.')
from itertools import permutations
from det import Determinant
from uhf import compute_uhf
from tools import *
from Hamiltonian import get_Hamiltonian

def compute_CI(Settings):
    Escf, C, _ , h, g, Vnuc = compute_uhf(Settings, return_C=True, return_integrals=True)
    
    print("""
    =========================================
       Configuration Interaction Starting
                       {}
    =========================================
    """.format(emoji('brinde')))
    
    # Initial workup
    
    nmo = len(C)
    nelec = Settings['nalpha'] + Settings['nbeta']
    if nelec % 2 != 0:
        raise NameError('This code works only for closed-shell molecules')
    
    ndocc = int(nelec/2)
    nvir = nmo - ndocc
    
    # Read in excitation level
    if type(Settings["excitation_level"]) == int:
        exlv = int(Settings["excitation_level"])
    
    elif type(Settings["excitation_level"]) == str:
        if Settings["excitation_level"].lower() == 'full':
            exlv = nelec
        elif Settings["excitation_level"].lower() == 'cis':
            exlv = 1
        elif Settings["excitation_level"].lower() == 'cisd':
            exlv = 2
        elif Settings["excitation_level"].lower() == 'cisdt':
            exlv = 3
        elif Settings["excitation_level"].lower() == 'cisdtq':
            exlv = 4
        elif Settings["excitation_level"].lower() == 'cisdtqp':
            exlv = 5
        else:
            raise TypeError('Invalid excitation level: {}'.format(Settings["excitation_level"]))
    else:
        raise TypeError('Invalid excitation level: {}'.format(Settings["excitation_level"]))
    
    # Convert AO integrals to Molecular integrals
    
    h = np.einsum('up,vq,uv->pq', C, C, h, optimize='optimal')
    ERI = np.einsum('ap,bq,cr,ds,abcd->pqrs', C, C, C, C, g, optimize='optimal')
    
    ############### STEP 1 ###############
    ########  Read Active Space  #########
    
    ## The active_space must be a string containing the letters 'o', 'a' and 'u':
    ##  'o' represents frozen doubly occupied orbitals;
    ##  'a' represents active orbitals;
    ##  'u' represents frozen unoccupied orbitals.
    ## However, the active_space can also be given in three alternative ways:
    ## active_space = 'full': for a full CI computation;
    ## active_space = 'none': for a emputy active space (output energy is SCF energy);
    ## If the active_space string is shorter than expected, remaining orbitals will be
    ## considered 'u' (unoccupied)
    ## Note that, the first orbital has index 0
    
    active_space = Settings['active_space']
    
    if active_space == 'full':
        active_space = 'a'*nmo
    
    elif active_space == 'none':
        active_space = 'o'*ndocc + 'u'*nvir
    
    # If the active space string is smaller than the number of orbitals,
    # append 'u' (unoccupied) at the end
    
    while len(active_space) < nmo:
        active_space += 'u'
    
    # Check if active space size is consistem with given integrals
    if len(active_space) != nmo:
        raise NameError('Active Space size is {} than the number of molecular orbitals'.format('greater' if len(active_space) > nmo else 'smaller'))
    
    nfrozen = active_space.count('o')
    nactive = active_space.count('a')
    active_elec = nelec - 2*nfrozen
    
    # Check if number of frozen orbitals is valid (must be less than the number of electron pairs)
    if active_elec < 0:
        raise NameError('Invalid number of frozen orbitals ({}), for {} electrons'.format(nfrozen, nelec))
    
    # Check if there are enough active orbitals for the active electrons
    if 2*nactive < active_elec:
        raise NameError('{} active orbitals not enough for {} active electrons'.format(nactive, active_elec))
    
    print("Active Space:")
    out = ''
    for x in active_space:
        if x == 'o':
            out += emoji('Obold')
        elif x == 'a':
            out += emoji('O')
        elif x == 'u':
            out += emoji('X')
        else:
            raise TypeError('Invalid active space key: {}'.format(x))
    print(out + '\n')
    print("Number of active spatial orbitals:  {}".format(numoji(nactive)))
    print("Number of active electrons:         {}".format(numoji(active_elec)))
    print("Number of frozen orbitals:          {}\n".format(numoji(nfrozen)))
    
    ############### STEP 2 ###############
    ######  Generate Determinants  #######
    
    # Creating a template string. Active orbitals are represented by {}
    # Occupied orbitals are 1, and unnocupied 0. 
    # For example: 11{}{}{}000 represents a system with 8 orbitals where the two lowest ones are frozen (doubly occupied)
    # and the three highest ones are frozen (unnocupied). Thus, there are 3 active orbitals.
    
    template_space = active_space.replace('o', '1')
    template_space = template_space.replace('u', '0')
    template_space = template_space.replace('a', '{}')
    
    # Produce a reference determinant
    
    ref = Determinant(a = ('1'*ndocc + '0'*nvir), \
              b = ('1'*ndocc + '0'*nvir))
    
    # Produces a list of active electrons in active orbitals and get all permutations of it.
    # For example. Say we have a template 11{}{}{}000 as in the example above. If the system contains 6 electrons
    # we have one pair of active electrons. The list of active electrons will look like '100' and the permutations
    # will be generated (as lists): ['1', '0', '0'], ['0', '1', '0'], and ['0', '0', '1'].
    
    # Each permutation is then merged with the template above to generate a string used ot create a Determinant object
    # For example. 11{}{}{}000 is combine with ['0','1', '0'] to produce an alpha/beta string 11010000.
    # These strings are then combined to form various determinants
    
    print("Generating excitations", end=' ')
    perms = set(permutations('1'*int(active_elec/2) + '0'*int(nactive - active_elec/2)))
    print(emoji('check'))
    
    determinants = []
    for p1 in perms:
        for p2 in perms:
            newdet = Determinant(a = template_space.format(*p1), b = template_space.format(*p2))
            if ref - newdet > 2*exlv:
                continue
            else:
                determinants.append(newdet)
    
    ndets = len(determinants)
    nelements = int((ndets*(ndets+1))/2)
    print("Number of determinants:                   {}".format(numoji(ndets)))
    print("Number of unique matrix elements:         {}\n".format(numoji(nelements)))
    
    # Build Hamiltonian matrix
    print("Building Hamiltonian Matrix", end=' ')
    H = get_Hamiltonian(determinants, h, ERI)

    print(emoji('check'))
    
    # Diagonalize Hamiltonian
    print("Diagonalizing Hamiltonian Matrix", end=' ')
    Eci, Cci = np.linalg.eigh(H)
    print(emoji('check'))
    
    E0 = Eci[0] + Vnuc
    C0 = Cci[:,0]
    
    print('\nFinal CASCI Energy: {:<14.10f}'.format(E0))
    
    if exlv == nelec:
        psi4.set_options({'basis': Settings['basis'],
                          'scf_type': 'pk',
                          'puream'   : False,
                          'FROZEN_DOCC' : [nfrozen],
                          'ACTIVE' : [nactive],
                          'FCI' : True,
                          'reference': 'rhf'})
        
        print('Final DETCI Energy: {:<14.10f}'.format(psi4.energy('detci')))
    
    # Print wavefunction analysis
    
    ndet = 10
    args = np.argsort(C0*C0)
    sortedC = C0[args]
    sortedC = sortedC[::-1]
    sortedC = sortedC[:ndet]
    
    sortedD = np.array(determinants)[args]
    sortedD = sortedD[::-1]
    sortedD = sortedD[:ndet]
    
    print('\n\n' + emoji('bchart') + ' Wave function analysis')
    print('='*25)
    print('\n')
    
    print(' '*10 + emoji('top') +' Top {} determinants as voted by the public\n'.format(ndet))
    
    # Header
    l = len(ref.short_info())
    header = '    {:^7s}   {:^8s}   {:^' + str(l) + 's}   {:^9s}'
    print(header.format('%', 'Coef', 'Determinant', 'Exc lv'))
    print(' '*4 + '='*7 + '   ' + '='*8 + '   ' + '='*l  + '   ' + '='*9)
    
    # Actual info
    f = True
    for c,d in zip(sortedC, sortedD):
        sq = c*c
        ex = int((ref-d)/2)
        if f:
            print(' '*2 + emoji('trophy') + '{:7.5f}   {: 8.5f}   {}   {:^9s}'.format(sq, c, d.short_info(), numoji(ex))) 
            f = False
        else:
            print(' '*4 + '{:7.5f}   {: 8.5f}   {}   {:^9s}'.format(sq, c, d.short_info(), numoji(ex))) 

if __name__ == '__main__':
    
    from input import Settings
    compute_CI(Settings)
