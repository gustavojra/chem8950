import psi4
import numpy as np
import sys 
sys.path.append('.')
from uhf import compute_uhf
from tools import *
import time

def show_progress(timings, printif):

    # Function to display the progress of computation setup

    if len(timings) > 0 and printif == print:
        sys.stdout.write("\033[F"*5)

    tasks = """
    Building:
      \U0001F426  Spin-Orbital arrays            {} 
      \U0001F986  MO integrals                   {}
      \U0001F9A2   D[ijab]                        {}"""

    x = []
    for t in timings:
        x.append(emoji('check') + '   ' + '{:10.5f} seconds'.format(t))

    while len(x) < 3:
        x.append('')

    printif(tasks.format(*x))

def CEPA_Energy(T, V):

    return (1/4)*np.einsum('mncd, mncd -> ', V, T, optimize='optimal')

def CEPA_Amplitude(T, D, Voovv, Voooo, Vvvvv, Vvoov):

    newT = np.zeros(T.shape)

    newT += Voovv
    newT += 0.5*np.einsum('cdab, ijcd -> ijab', Vvvvv, T, optimize='optimal')
    newT += 0.5*np.einsum('ijmn, mnab -> ijab', Voooo, T, optimize='optimal')

    X = np.einsum('cjmb, imac -> ijab', Vvoov, T, optimize='optimal')

    newT += X - X.transpose(1,0,2,3) - X.transpose(0,1,3,2) + X.transpose(1,0,3,2)

    newT *= D

    rms = np.sqrt(np.sum(np.square(newT - T)))/(T.size)

    return newT, rms

def compute_CEPA(Settings, silent=False, compare=True):

    t0 = time.time()

    printif = print if not silent else lambda *k, **w: None

    Escf, Ca, Cb, epsa, epsb, _, g, Vnuc = compute_uhf(Settings, return_C=True, return_integrals=True, silent=silent)
    
    printif("""
    =======================================================
                           CEPA0
      (For those who havent heard Coupled Cluster exists)
                             {}
    =======================================================
    """.format('\U0001F474'))

    tsave = []
    show_progress(tsave, printif)
    t = time.time()

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

    tsave.append(time.time() - t)
    show_progress(tsave, printif)
    t = time.time()
    
    # Get the MO integral

    nelec = Settings['nalpha'] + Settings['nbeta']
    # Converto to Physicists notation
    g = g.transpose(0,2,1,3)
    # Antisymmetrize
    g = g - g.transpose(0,1,3,2)

    # Save Slices
    ERI = lambda a,b,c,d: np.einsum('ap,bq,cr,ds,abcd->pqrs', C[:,a], C[:,b], 
                                    C[:,c], C[:,d], g, optimize='optimal')
    o = slice(0, nelec)
    v = slice(nelec, len(g))
    Voovv = ERI(o,o,v,v)
    Voooo = ERI(o,o,o,o)
    Vvvvv = ERI(v,v,v,v)
    Vvoov = ERI(v,o,o,v)

    g = None

    tsave.append(time.time() - t)
    show_progress(tsave, printif)
    t = time.time()

    # Get eigenvalues Matrix D
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

    printif('\nMP2 Energy:   {:<15.10f}\n'.format(E + Escf))

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
        T, rms = CEPA_Amplitude(T, D, Voovv, Voooo, Vvvvv, Vvoov)
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

    printif('\U0001F3C1 CEPA0 Energy:   {:<15.10f}'.format(E + Escf))
    printif('\U000023F3 CEPA0 iterations took %.2f seconds.\n' % (time.time() - t0))

    if compare:

        psi4.set_options({'basis': Settings['basis'],
                          'scf_type': 'pk',
                          'mp2_type' : 'conv',
                          'puream'   : False,
                          'reference': 'uhf'})

        psi4_lccd = psi4.energy('lccd')
        printif('\n{} Psi4  LCCD Energy:         {:>16.10f}'.format(emoji('eyes'), psi4_lccd))

        if abs(psi4_lccd - E - Escf) < 1.e-8:
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

    return (E + Escf)


if __name__ == '__main__':

    from input import Settings
    compute_CEPA(Settings)
