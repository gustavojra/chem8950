import psi4
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)
psi4.core.be_quiet()
np.set_printoptions(precision=10, linewidth=140, suppress=True)
from input import Settings
from itertools import permutations

class BasisFunction:

    def __init__(self, center, nprimitive, coef, exponents, AM):
        
        self.center = np.array(center)
        self.nprimitive = nprimitive
        self.coef = coef
        self.exp = exponents
        self.AM = AM
        self.cartesian_func()
        self.ncart = len(self.cart)

        if AM == 0:
            self.type = 'S'
        elif AM == 1:
            self.type = 'P'
        elif AM == 2:
            self.type = 'D'
        elif AM > 2:
            # chr(70) = 'F', chr(71) = 'G', etc
            self.type = chr(67+AM)  
        else:
            raise ValueError('Invalid angular momentum value')

    def cartesian_func(self):
        
        # Returna list of all cartesian functions within this basis function
        # e.g. for a P function return [Px, Py, Pz] where Px = [1,0,0] Py = [0,1,0] and Pz = [0,0,1]

        self.cart = []
        case1 = [self.AM, 0, 0]
        self.cart += list(set(permutations(case1)))
        i = 1
        j = 1
        while self.AM - i > 0 and self.AM >= 2*i:
            case2 = [self.AM - i, i, 0]
            self.cart += list(set(permutations(case2)))
            while self.AM - i - j > 0 and j <= i:
                case3 = [self.AM - i - j, i, j]
                self.cart += list(set(permutations(case3)))
                j += 1
            i += 1
            j = 1
        self.cart = list(set(self.cart))
        self.cart.sort(reverse=True)
        
    def __str__(self):
        out = """
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Basis Function Type:        {:5s}      %%
%% Number of Primitives:       {:<5d}      %%
%% Cartesian functions:        {:<5d}      %%
%% Center:                                %%
%% {:< 7.3f} {:< 7.3f} {:< 7.3f}                %%
%% Contraction Coefficients:              %%\n"""
        nprint = self.nprimitive
        while nprint % 3 != 0:
            nprint += 1
        out += '%%'
        for c in range(nprint):
            try:
                out += ' {:< 7.5f}'.format(self.coef[c])
            except IndexError:
                out += '         '
            if (c+1) % 3 == 0:
                out += '             %%\n'
                out += '%%'
        out += " Exponents:                             %%\n%%"""
        for e in range(nprint):
            try:
                out += ' {:< 6.4e}'.format(self.exp[e])
            except IndexError:
                out += '            '
            if (e+1) % 3 == 0:
                out += '    %%\n'
                out += '%%'
        out += """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
        return out.format(self.type, self.nprimitive, self.ncart, *self.center)

def os_recursion(PA, PB, alpha, AMa, AMb):

    # For a pair of primitives return all geometric combinations
    
    x = np.zeros((AMa+1, AMb+1))
    y = np.zeros((AMa+1, AMb+1))
    z = np.zeros((AMa+1, AMb+1))    

    for a in range(AMa+1):
        for b in range(AMb+1):
            if a == 0 and b == 0:
                x[a,b] = 1.0
                y[a,b] = 1.0
                z[a,b] = 1.0

            elif b == 0:
                x[a,b] += PA[0]*x[a-1,b]
                y[a,b] += PA[1]*y[a-1,b]
                z[a,b] += PA[2]*z[a-1,b]
                if a >= 2: 
                    x[a,b] += 1.0/(2.0*alpha)*(a-1)*x[a-2,b]
                    y[a,b] += 1.0/(2.0*alpha)*(a-1)*y[a-2,b]
                    z[a,b] += 1.0/(2.0*alpha)*(a-1)*z[a-2,b]

            else:
                # (a|b+1) formula
                x[a,b] += PB[0]*x[a,b-1]
                y[a,b] += PB[1]*y[a,b-1]
                z[a,b] += PB[2]*z[a,b-1]
                if a >= 1 and b >= 1:
                    x[a,b] += 1.0/(2*alpha)*a*x[a-1,b-1]
                    y[a,b] += 1.0/(2*alpha)*a*y[a-1,b-1]
                    z[a,b] += 1.0/(2*alpha)*a*z[a-1,b-1]
                if b >= 2:
                    x[a,b] += (1.0/(2*alpha))*(b-1)*x[a,b-2]
                    y[a,b] += (1.0/(2*alpha))*(b-1)*y[a,b-2]
                    z[a,b] += (1.0/(2*alpha))*(b-1)*z[a,b-2]

    return (x,y,z)

def overlap_block(bf1, bf2):

    # Compute the overlap between two basis function.
    # Return a block with overlap of all combinations
    # e.g. for (S|P) the output is a 1x3 matrix containing (S|Px) (S|Py) and (S|Py) 

    S = np.zeros((bf1.ncart, bf2.ncart))

    A = bf1.center
    B = bf2.center
    for p1 in range(bf1.nprimitive):
        for p2 in range(bf2.nprimitive):

            aa = bf1.exp[p1]
            bb = bf2.exp[p2]
            alpha = aa + bb
            xi = aa*bb/alpha
            kappa = np.exp(-xi*np.dot(A-B,A-B))
            P = (aa*A + bb*B)/alpha
            (x,y,z) = os_recursion(P-A, P-B, alpha, bf1.AM, bf2.AM) 
            W = np.zeros((bf1.ncart, bf2.ncart))

            for kA in range(bf1.ncart):
                for kB in range(bf2.ncart):
                    caseA = bf1.cart[kA]
                    caseB = bf2.cart[kB]
                    W[kA, kB] = x[caseA[0], caseB[0]]*y[caseA[1], caseB[1]]*z[caseA[2], caseB[2]]

            S += bf1.coef[p1]*bf2.coef[p2]*kappa*W

    return S

def overlap(basisset):
    
    # Return the overlap matrix given a list o basis functions

    nbf = 0
    # First, determine the number of basis functions
    for b in basisset:
        nbf += b.ncart
    print('Number of basis functions: {}'.format(nbf))
    # Initialize the S matrix

    S = np.zeros((nbf, nbf)) 

    # Loop through basis functions
    i = 0
    for b1 in basisset:
        j = 0
        s1 = slice(i, i+b1.ncart)
        for b2 in basisset:
            print('Starting integral ({}|{})'.format(b1.type, b2.type))
            # Define slices
            s2 = slice(j, j+b2.ncart)
            print(s1)
            print(s2)
            S[s1,s2] = overlap_block(b1, b2)
            j += b2.ncart
        i += b1.ncart

    return S

# Collect basis function Info from Psi4 for and save into my BasisFunction objects

molecule = psi4.geometry(Settings['molecule'])
basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'], puream=0)
mints = psi4.core.MintsHelper(basis)

bfs = []
for ishell in range(basis.nshell()):
    shell = basis.shell(ishell)
    center = [molecule.x(shell.ncenter), molecule.y(shell.ncenter), molecule.z(shell.ncenter)]
    exp = []
    c   = []
    for i in range(shell.nprimitive):
        exp.append(shell.exp(i))
        c.append(shell.coef(i))
    b = BasisFunction(center, shell.nprimitive, c, exp, shell.am)
    bfs.append(b)



psi4S = mints.ao_overlap().np

_1s = bfs[0]
pp.pprint(overlap_block(_1s,_1s))
pp.pprint(psi4S)
