import psi4
import numpy as np
import sys 
sys.path.append('.')
from uhf import compute_uhf
from tools import *
import time


def compute_SOMP2(Settings, silent=False, psi4compare=True):

    printif = print if not silent else lambda *k, **w: None

    Escf, Ca, Cb, epsa, epsb, _, g, Vnuc = compute_uhf(Settings, return_C=True, return_integrals=True, psi4compare=psi4compare, silent=silent)
    
    printif("""
    ===================================================
      Spin-Orbital Møller-Plesset Perturbation Theory
                           {}
    ===================================================
    """.format(emoji('wine')))

    tasks = """
    Building:
      \U0001F426  Spin-Orbital arrays               {}
      \U0001F986  MO integral <OO|VV>               {}
      \U0001F9A2   Eigenvalues auxiliar array Dijab  {}"""

    printif(tasks.format('','',''))
    if not silent: sys.stdout.write("\033[F"*5)

    # Create Spin-Orbital arrays
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

    printif(tasks.format(emoji('check'),'',''))
    if not silent: sys.stdout.write("\033[F"*5)
    
    # Get the MO integral OOVV

    nelec = Settings['nalpha'] + Settings['nbeta']
    # Converto to Physicists notation
    g = g.transpose(0,2,1,3)
    ERI = np.einsum('ap,bq,cr,ds,abcd->pqrs', C[:,:nelec], C[:,:nelec], C[:,nelec:], C[:,nelec:], g, optimize='optimal')
    # Antisymmetrize
    ERI = ERI - ERI.transpose(0,1,3,2)

    printif(tasks.format(emoji('check'), emoji('check'),''))
    if not silent: sys.stdout.write("\033[F"*5)

    # Get eigenvalues Matrix D
    new = np.newaxis
    eo = eps[:nelec]
    ev = eps[nelec:]
    D = 1.0/(eo[:, new, new, new] + eo[new, :, new, new] - ev[new, new, :, new] - ev[new, new, new, :])

    printif(tasks.format(emoji('check'), emoji('check'), emoji('check')))

    # Get MP2 energy

    printif('\n{}  Computing MP2 energy'.format('\U0001F9ED'), end= ' ')
    Emp2 = (1.0/4.0)*np.einsum('ijab,ijab,ijab->', ERI, ERI, D)
    printif(emoji('check'))

    printif('\n{} MP2 Correlation Energy:    {:>16.10f}'.format(emoji('bolt'), Emp2))
    printif('\U0001F308 Final Electronic Energy:   {:>16.10f}'.format(Emp2+Escf))

    if psi4compare:

        psi4.set_options({'basis': Settings['basis'],
                          'scf_type': 'pk',
                          'mp2_type' : 'conv',
                          'puream'   : False,
                          'reference': 'uhf'})

        psi4_mp2 = psi4.energy('mp2')
        printif('\n\n{} Psi4  MP2 Energy:          {:>16.10f}'.format(emoji('eyes'), psi4_mp2))

        if abs(psi4_mp2 - Emp2 - Escf) < 1.e-8:
            printif('\n         ' + emoji('books'), end = ' ')
            printif('My grade:')                                   
            printif(\
        """   
                       AAA               
                      A:::A              
                     A:::::A             
                    A:::::::A            
                   A:::::::::A           
                  A:::::A:::::A          
                 A:::::A A:::::A         
                A:::::A   A:::::A        
               A:::::A     A:::::A       
              A:::::AAAAAAAAA:::::A      
             A:::::::::::::::::::::A     
            A:::::AAAAAAAAAAAAA:::::A    
           A:::::A             A:::::A   
          A:::::A               A:::::A  
         A:::::A                 A:::::A 
        AAAAAAA                   AAAAAAA""")

    return Emp2

if __name__ == '__main__':

    from input import Settings
    compute_SOMP2(Settings)
