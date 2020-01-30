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
    "ugh"    : b'\xF0\x9F\x91\xBE'.decode('utf-8')
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

# FUNCTIONS

def contamination(C_a, C_b, S, nalpha, nbeta):
    Smetric = np.einsum('ui,uv,vj->ij', C_a[:,:nalpha], S, C_b[:,:nbeta])
    return min(nalpha, nbeta) - np.vdot(Smetric, Smetric)

# SETUP INITIAL CONDITIONS

print(shrek)

psi4.core.be_quiet()
print('\nReading input...', end=' ')
molecule = psi4.geometry(Settings['molecule'])
molecule.update_geometry()

nalpha = Settings['nalpha']
nbeta = Settings['nbeta']
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
                  'reference': 'uhf'})

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
