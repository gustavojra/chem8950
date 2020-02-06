import numpy as np
from scipy import special, linalg
np.set_printoptions(linewidth=200)

 # Input Geometry
geom = np.array([\
 [ 0.,          0.,         -0.849220457955],
 [ 0.,          0.,          0.849220457955]])

# Input atoms (Atomic charge)
atomic_nums = [1,1]

# Basis function info
nbf = 8
nbf_per_atom = np.array([4,4])

centers = np.repeat(geom, nbf_per_atom, axis=0)
exponents = np.tile(np.array([0.5, 0.4, 0.3, 0.2]), 2)

def normalize(aa):

    # Return normalization constant of a S function with exponent aa

    return ((2*aa)/np.pi)**(3/4)

def boys0(x):

    # Return the 0th Boys function for sqrt(x)

    if x < 1e-8:
        return 1 - x/3 + x**2/10 - x**3/42
    else:
        return special.erf(np.sqrt(x))*(np.sqrt(np.pi)/(2*np.sqrt(x)))

def overlap(A,B,aa,bb):

    # Return the overlap of two S functions with exponents aa and bb and position vector A and B

    Na = normalize(aa)
    Nb = normalize(bb)
    return Na*Nb*(np.pi/(aa+bb))**(3/2)*np.exp((-aa*bb*np.dot(A-B,A-B))/(aa+bb))

def kinetic(A,B,aa,bb):

    # Return the kinetic energy integral for two S functions with exponents aa and bb and position vector A and B

    w = aa*bb/(aa+bb)
    return overlap(A,B,aa,bb)*(3*w - 2*w*w*np.dot(A-B,A-B))

def potential(A,B,aa,bb,geom,charges):

    # Return the nuclear potential energy integral for two S functions with exponents aa and bb and position vector A and B
    # under the field of atoms in 'geom' with atomic numbers in 'charges'

    P = (aa*A + bb*B)/(aa+bb)
    w = aa*bb/(aa+bb)
    Na = normalize(aa)
    Nb = normalize(bb)
    out = 0
    for G,Z in zip(geom,charges):
        xi = (aa+bb)*np.dot(P-G,P-G)
        out -= Z*boys0(xi)
    return Na*Nb*(2*np.pi)/(aa+bb)*np.exp(-np.dot(A-B,A-B)*w)*out

def ERI(A,B,C,D,aa,bb,cc,dd):

    # Return the two-electron integral for four S functions with exponents aa,bb,cc,dd and positions A,B,C, and D.

    P = (aa*A + bb*B)/(aa+bb)
    Q = (cc*C + dd*D)/(cc+dd)
    T = ((aa+bb)*(cc+dd))/(aa+bb+cc+dd)*np.dot(P-Q,P-Q)
    Kab = np.sqrt(2)*np.pi**(5/4)/(aa+bb)*np.exp(-(aa*bb)/(aa+bb)*np.dot(A-B,A-B))
    Kcd = np.sqrt(2)*np.pi**(5/4)/(cc+dd)*np.exp(-(cc*dd)/(cc+dd)*np.dot(C-D,C-D))
    Na = normalize(aa)
    Nb = normalize(bb)
    Nc = normalize(cc)
    Nd = normalize(dd)
    return Na*Nb*Nc*Nd*(aa+bb+cc+dd)**(-1/2)*Kab*Kcd*boys0(T)

def nuclear_rep(geom, charges):

    # Return the total repulsion energy due to nuclei positions as given by 'geom' with atomic numbers in 'charges'

    Vnuc = 0
    nuclei = len(geom)
    for i in range(0,nuclei):
        for j in range(i+1, nuclei):
            A = geom[i]
            B = geom[j]
            Za = charges[i]
            Zb = charges[j]
            d = np.sqrt(np.dot(A-B,A-B))
            Vnuc += Za*Zb/d

    return Vnuc

def build_overlap(centers, exponents, nbf):

    # Return the overlap matrix 'S'

    S = np.zeros((nbf, nbf))
    for i in range(nbf):
        for j in range(i, nbf):
            A = centers[i]
            B = centers[j]
            aa = exponents[i]
            bb = exponents[j]
            S[i,j] = overlap(A,B,aa,bb)
            S[j,i] = S[i,j]
    return S

def build_kinetic(centers, exponents, nbf):

    # Return the overlap kinetic energy integral matrix 'T'

    T = np.zeros((nbf, nbf))
    for i in range(nbf):
        for j in range(i, nbf):
            A = centers[i]
            B = centers[j]
            aa = exponents[i]
            bb = exponents[j]
            T[i,j] = kinetic(A,B,aa,bb)
            T[j,i] = T[i,j]
    return T

def build_potential(centers, exponents, nbf, geom, charges):

    # Return the potential energy integrals 'V' matrix due to the nuclei given in 'geom' with atomic number given in 'charges'

    V = np.zeros((nbf, nbf))
    for i in range(nbf):
        for j in range(i, nbf):
            A = centers[i]
            B = centers[j]
            aa = exponents[i]
            bb = exponents[j]
            V[i,j] = potential(A,B,aa,bb, geom, charges)
            V[j,i] = V[i,j]
    return V

def build_ERI(centers, exponents, nbf):

    # Return the two-electron integral tensor 'I' (chemist's notation) 

    I = np.zeros((nbf, nbf, nbf, nbf))
    for p in range(nbf):
        for q in range(nbf):
            for r in range(nbf):
                for s in range(nbf):
                    A = centers[p]
                    B = centers[q]
                    C = centers[r]
                    D = centers[s]
                    aa = exponents[p]
                    bb = exponents[q]
                    cc = exponents[r]
                    dd = exponents[s]
                    out = ERI(A,B,C,D,aa,bb,cc,dd)
                    I[p,q,r,s] = out
    return I


# RUN RHF

emoji = {\
    "viva"   : b'\xF0\x9F\x8E\x89'.decode('utf-8'),
    "eyes"   : b'\xF0\x9F\x91\x80'.decode('utf-8'),
    "cycle"  : b'\xF0\x9F\x94\x83'.decode('utf-8'),
    "bolt"   : b'\xE2\x9A\xA1'.decode('utf-8'),
    "pin"    : b'\xF0\x9F\x93\x8C'.decode('utf-8'),
    "crying" : b'\xF0\x9F\x98\xAD'.decode('utf-8'),
    "pleft"  : b'\xF0\x9F\x91\x88'.decode('utf-8'),
    "whale"  : b'\xF0\x9F\x90\xB3'.decode('utf-8'),
    "books"  : b'\xF0\x9F\x93\x9A'.decode('utf-8'),
    "wavy"   : b'\xE3\x80\xB0'.decode('utf-8'),
    "blue"   : b'\xF0\x9F\x94\xB5'.decode('utf-8'),
    "red"    : b'\xF0\x9F\x94\xB4'.decode('utf-8'),
    "check"  : b'\xE2\x9C\x85'.decode('utf-8'),
    "link"   : b'\xF0\x9F\x94\x97'.decode('utf-8'),
    "dash"   : b'\xF0\x9F\x92\xA8'.decode('utf-8'),
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

def numoji(i):
    i = str(i)
    out = ''
    for l in i:
        out += emoji[l] + ' '
    return out

# COMPUTE INTEGRALS

print('Computing Integrals...', end = ' ')
S = build_overlap(centers,exponents, nbf)
T = build_kinetic(centers,exponents, nbf)
V = build_potential(centers, exponents, nbf, geom, atomic_nums)
I = build_ERI(centers, exponents, nbf)
Vnuc = nuclear_rep(geom, atomic_nums)
print(emoji['check'])

print('\n{} Overlap Matrix'.format(emoji['blue']+emoji['blue']))
print(S)

print('\n{} Kinetic Energy Matrix'.format(emoji['blue']+emoji['dash']))
print(T)

print('\n{} Electron-Nuclei Potential'.format(emoji['blue']+emoji['bolt']+emoji['red']))
print(V)

print('\n{} Electron-Electron Potential (slice)'.format(emoji['blue']+emoji['bolt']+emoji['blue']))
print(I[0,0,:,:])

print('\n{} Nuclei-Nuclei Potential'.format(emoji['red']+emoji['bolt']+emoji['red']))
print(Vnuc)

# SETUP INITIAL CONDITIONS

ndocc = 1
scf_max_iter = 50

# CONSTRUCT THE ORTHOGONALIZER

print('\nInitial guess...', end=' ')
A = linalg.fractional_matrix_power(S, -0.5)

# INITIAL DENSITY MATRIX

h = T + V
Ft = A.dot(h).dot(A)
_,Ct = np.linalg.eigh(Ft)
C = A.dot(Ct)
D = np.einsum('um,vm-> uv', C[:,:ndocc], C[:,:ndocc])
G = 2*I - I.swapaxes(1,2)
E = 2*np.einsum('pq,pq->', D, h) + np.einsum('pq,rs,pqrs->', D, D, G) + Vnuc
print(emoji['check'])

# START SCF ITERATIONS

rms = 1
dE = 1
ite = 1

# Convergency criteria

dE_max = 1.e-12
rms_max = 1.e-8


print('\nStarting SCF iterations' + emoji['cycle'])
print('Convergency criteria:')
print('{} Max Energy Diff:  {:<4.2E}'.format(emoji['pin'],dE_max))
print('{} Max RMS density:  {:<4.2E}'.format(emoji['pin'],rms_max))
print('='*40)

while abs(dE) > dE_max or abs(rms) > rms_max:
    dE = -E
    rms = D[:,:]
    if ite > scf_max_iter:
        raise NameError('SCF cannot converge {c} {c} {c}'.format(c=emoji['crying']))
    
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

print('SCF Converged ' + 2*emoji['viva'])
print('{} Final SCF Energy:   {:>16.12f}'.format(emoji['bolt'],E))
