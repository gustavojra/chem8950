import psi4
import numpy as np
import scipy.linalg as sp
import sys

# Let the script be called anywhere
try:
    from input import Settings
except:
    sys.path.append('.')
    from input import Settings


# Miscellaneous
def emoji(key):
    stored = {
    "viva"   : b'\xF0\x9F\x8E\x89'.decode('utf-8'),
    "eyes"   : b'\xF0\x9F\x91\x80'.decode('utf-8'),
    "cycle"  : b'\xF0\x9F\x94\x83'.decode('utf-8'),
    "bolt"   : b'\xE2\x9A\xA1'.decode('utf-8'),
    "pin"    : b'\xF0\x9F\x93\x8C'.decode('utf-8'),
    "crying" : b'\xF0\x9F\x98\xAD'.decode('utf-8'),
    "pleft"  : b'\xF0\x9F\x91\x88'.decode('utf-8'),
    "whale"  : b'\xF0\x9F\x90\xB3'.decode('utf-8'),
    "books"  : b'\xF0\x9F\x93\x9A'.decode('utf-8'),
    "check"  : b'\xE2\x9C\x85'.decode('utf-8'),
    "0"      : b'\x30\xE2\x83\xA3'.decode('utf-8'),
    "1"      : b'\x31\xE2\x83\xA3'.decode('utf-8'),
    "2"      : b'\x32\xE2\x83\xA3'.decode('utf-8'),
    "3"      : b'\x33\xE2\x83\xA3'.decode('utf-8'),
    "4"      : b'\x34\xE2\x83\xA3'.decode('utf-8'),
    "5"      : b'\x35\xE2\x83\xA3'.decode('utf-8'),
    "6"      : b'\x36\xE2\x83\xA3'.decode('utf-8'),
    "7"      : b'\x37\xE2\x83\xA3'.decode('utf-8'),
    "8"      : b'\x38\xE2\x83\xA3'.decode('utf-8'),
    "9"      : b'\x39\xE2\x83\xA3'.decode('utf-8'),
    }
    return stored[key]

def numoji(i):
    i = str(i)
    out = ''
    for l in i:
        out += emoji(l) + ' '
    return out

shrek = \
r"""######################################################################################
#                                                                                    #
#                            ,.--------._                                            #
#                           /            ''.                                         #
#                         ,'                \     |"\                /\          /\  #
#                /"|     /                   \    |__"              ( \\        // ) #
#               "_"|    /           z#####z   \  //                  \ \\      // /  #
#                 \\  #####        ##------".  \//                    \_\\||||//_/   #
#                  \\/-----\     /          ".  \                      \/ _  _ \     #
#                   \|      \   |   ,,--..       \                    \/|(O)(O)|     #
#                   | ,.--._ \  (  | ##   \)      \                  \/ |      |     #
#                   |(  ##  )/   \ `-....-//       |///////////////_\/  \      /     #
#                     '--'."      \                \              //     |____|      #
#                  /'    /         ) --.            \            ||     /      \     #
#               ,..|     \.________/    `-..         \   \       \|     \ 0  0 /     #
#            _,##/ |   ,/   /   \           \         \   \       U    / \_//_/      #
#          :###.-  |  ,/   /     \        /' ""\      .\        (     /              #
#         /####|   |   (.___________,---',/    |       |\=._____|  |_/               #
#        /#####|   |     \__|__|__|__|_,/             |####\    |  ||                #
#       /######\   \      \__________/                /#####|   \  ||                #
#      /|#######`. `\                                /#######\   | ||                #
#     /++\#########\  \                      _,'    _/#########\ | ||                #
#    /++++|#########|  \      .---..       ,/      ,'##########.\|_||  Donkey By     #
#   //++++|#########\.  \.              ,-/      ,'########,+++++\\_\\ Hard'96       #
#  /++++++|##########\.   '._        _,/       ,'######,''++++++++\                  #
# |+++++++|###########|       -----."        _'#######' +++++++++++\                 #
# |+++++++|############\.     \\     //      /#######/++++ S@yaN +++\                #
#        _   _               _                            _____             _        #
#       | | | |  __ _  _ __ | |_  _ __  ___   ___        |  ___|___    ___ | | __    #
#       | |_| | / _` || '__|| __|| '__|/ _ \ / _ \ _____ | |_  / _ \  / __|| |/ /    #
#       |  _  || (_| || |   | |_ | |  |  __/|  __/|_____||  _|| (_) || (__ |   <     #
#       |_|_|_| \__,_||_| _  \__||_|   \___| \___|       |_|   \___/  \___||_|\_|    #
######################################################################################"""


# SETUP INITIAL CONDITIONS

print(shrek)

psi4.core.be_quiet()
print('\nReading input...', end=' ')
molecule = psi4.geometry(Settings['molecule'])
molecule.update_geometry()

if Settings['nalpha'] != Settings['nbeta']:
    raise NameError('No open shell stuff plz')

ndocc = Settings['nalpha']
Vnuc = molecule.nuclear_repulsion_energy()

basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'])

mints = psi4.core.MintsHelper(basis)

scf_max_iter = Settings['scf_max_iter']
print(emoji('check'))

# COMPUTE INTEGRALS

print('Computing integrals...', end=' ')
## Overlap

S = mints.ao_overlap().np

## Kinetic

T = mints.ao_kinetic().np

## Nuclei-electron potential energy

V = mints.ao_potential().np

## Electron repulsion integral

ERI = mints.ao_eri().np

# FORM ONE-ELECTRON HAMILTONIAN

h = T + V
print(emoji('check'))

# CONSTRUCT THE ORTHOGONALIZER

print('Initial guess...', end=' ')
A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = A.np

# INITIAL DENSITY MATRIX

Ft = A.dot(h).dot(A)
_,Ct = np.linalg.eigh(Ft)
C = A.dot(Ct)
D = np.einsum('um,vm-> uv', C[:,:ndocc], C[:,:ndocc])
G = 2*ERI - ERI.swapaxes(1,2)
E = 2*np.einsum('pq,pq->', D, h) + np.einsum('pq,rs,pqrs->', D, D, G) + Vnuc
print(emoji('check'))
# START SCF ITERATIONS

rms = 1
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

while abs(dE) > dE_max or abs(rms) > rms_max:
    dE = -E
    rms = D[:,:]
    if ite > scf_max_iter:
        raise NameError('SCF cannot converge {c} {c} {c}'.format(c=emoji('crying')))
    
    F = h + np.einsum('ps,uvps->uv', D, G)
    Ft = A.dot(F).dot(A)
    eps, Ct = np.linalg.eigh(Ft)
    C = A.dot(Ct)
    D = np.einsum('um,vm-> uv', C[:,:ndocc], C[:,:ndocc])
    E = 2*np.einsum('pq,pq->', D, h) + np.einsum('pq,rs,pqrs->', D, D, G) + Vnuc
    dE += E
    rms = np.sqrt(np.sum(np.square(D-rms)))
    print('Iteration {}'.format(numoji(ite)))
    print('SCF Energy:         {:>16.12f}'.format(E))
    print('Energy change:      {:>16.12f}'.format(dE))
    print('Density rms change: {:>16.12f}'.format(rms))
    print('='*40)
    ite += 1

print('SCF Converged ' + 2*emoji('viva'))
print('{} Final SCF Energy:   {:>16.12f}'.format(emoji('bolt'),E))

psi4.set_options({'basis': Settings['basis'],
                  'scf_type': 'pk',
                  'e_convergence': dE_max,
                  'reference': 'rhf'})

# Compare to Psi4

E_psi4 = psi4.energy('scf')
print('{} Psi4  SCF Energy:   {:>16.12f}'.format(emoji('eyes'),E_psi4))

if abs(E_psi4 - E) < 1.e-8:
    print('\n         ' + emoji('books'), end = ' ')
    print('My grade:\n')                                   
    print(\
"""               AAA               
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
