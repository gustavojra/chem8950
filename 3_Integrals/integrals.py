import psi4
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

np.set_printoptions(precision=10, linewidth=140, suppress=True)
from input import Settings

pp.pprint(Settings)

molecule = psi4.geometry(Settings['molecule'])
basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'], puream=0)
mints = psi4.core.MintsHelper(basis)

class Primitive:
    
    def __init__(self, center, exponent, AM):

        self.center = center
        self.exponent = exponent
        self.AM = AM
        self.L = AM.sum()

class BasisFunction:

    def __init__(self, center, nprimitive, coef, exponents, L):
        
        self.nprimitive = nprimitive
        self.coef = coef
        self.prims = []
        pass
        

basis = []
for ishell in range(basis.nshell()):
    shell = basis.shell(ishell)
    center = [molecule.x(shell.ncenter), molecule.y(shell.ncenter), molecule.z(shell.ncenter)]
    exp = []
    c   = []
    for i in range(shell.nprimitive):
        exp.append(shell.exp(i))
        c.append(shell.coef(i))
    b = BasisFunction(center, shell.nprimitive, c, exp, shell.am)
        
from collections import namedtuple
RecursionResults = namedtuple('RecursionResults', ['x', 'y', 'z'])

def os_recursion(A, B, alpha, AMa, AMb):

    # For a pair of primitives return all geometric combinations
    
    x = np.zeros((AMa+1, AMb+1))
    y = np.zeros((AMa+1, AMb+1))
    z = np.zeros((AMa+1, AMb+1))    

    for i in range(AMa+1):
        for j in range(AMb+1):
            if i == 0 and j == 0:
                
            

    return RecursionResults(x,y,z)
