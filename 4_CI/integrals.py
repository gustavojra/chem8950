import numpy as np
from itertools import permutations
np.set_printoptions(precision=10, linewidth=140, suppress=True)

#################################################################
##  ___           _                                   _        ## 
## |_ _|  _ __   | |_    ___    __ _   _ __    __ _  | |  ___  ##
##  | |  | '_ \  | __|  / _ \  / _` | | '__|  / _` | | | / __| ##
##  | |  | | | | | |_  |  __/ | (_| | | |    | (_| | | | \__ \ ##
## |___| |_| |_|  \__|  \___|  \__, | |_|     \__,_| |_| |___/ ##
##                             |___/                           ##
#################################################################

class BasisFunction:

    # Object that stores the information of a Basis Function (Shell)

    def __init__(self, center, coef, exponents, AM):
        
        # ARGUMENTS:
        #    center: The position of the basis function in the Cartesian space (x,y,z)
        #      coef: Contraction coefficient for each primitive
        # exponents: Exponents values for each primitive
        #        AM: Angular momentum quantum number of the basis function (e.g. 0, 1, 2, etc for S, P, D, etc)  

        if len(coef) != len(exponents):
            raise NameError('Number of contraction coefficients must match number of exponents values')

        if len(center) != 3:
            raise NameError('Center must be given as an array of Cartesian coordinates (x,y,z)')
        
        self.center = np.array(center)
        self.nprimitive = len(coef)
        self.coef = coef
        self.exp = exponents
        self.AM = AM

        # Call cartesian function to produce lists representing each Cartesian function within this basis
        self.cartesian_func()
        self.ncart = len(self.cart)

        # Get a character to represent the basis function type (e.g. S, P, D, F)
        # S, P and D are hard-coded, F and beyond are obtained automatically
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
        # This could be easily hard-coded for S, P and D functions, but to be general I created this function
        # that should work for arbitrary angular momentum

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

        # An auxiliar function to print the information of the basis function

        out = (
        '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'
        '%% Basis Function Type:        {:5s}      %%\n'
        '%% Number of Primitives:       {:<5d}      %%\n'
        '%% Cartesian functions:        {:<5d}      %%\n'
        '%% Center:                                %%\n'
        '%% {:< 10.5f} {:< 10.5f} {:< 10.5f}       %%\n'
        '%% Contraction Coefficients:              %%\n')
        nprint = self.nprimitive
        while nprint % 3 != 0:
            nprint += 1
        out += '%%'
        for c in range(nprint):
            try:
                out += ' {:< 10.5f}'.format(self.coef[c])
            except IndexError:
                out += '           '
            if (c+1) % 3 == 0:
                out += '       %%\n'
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

    # Define overloaded functions to allowed comparison of basis based on their angular momentum

    def __lt__(self, other):

        return self.AM < other.AM

    def __le__(self, other):

        return self.AM <= other.AM

    def __eq__(self, other):
        
        return self.AM == other.AM

    def __ge__(self, other):

        return self.AM >= other.AM

    def __gt__(self, other):

        return self.AM > other.AM

    def __ne__(self, other):

        return self.AM != other.AM

class BasisSet:

    # This class is just an object that stores a collection of BasisFunctions objects

    def __init__(self, name = 'Unnamed'):

        # The basis function is initialized empty

        self.basisfunctions = []
        self.slices = []
        # Number of basisfunctions counting every cartesian type
        self.nbf = 0
        self.name = name

    def __iter__(self):

        # Iteration through BasisSet returns three values: shell index, basis function and slice of the cartesian functions

        return zip(range(len(self.basisfunctions)), self.basisfunctions, self.slices)

    def __str__(self):

        # Print information of the set
        return (
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        '~~{:^40}~~\n'
        '~~{:^40}~~\n'
        '~~ Number of Shells:            {:^10}~~\n'
        '~~ Number of Cartesian Basis:   {:^10}~~\n' 
        '~~ Maximum Angular Momentum:    {:^10}~~\n'
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~').format('Basis Set Info', self.name, len(self.basisfunctions), self.nbf, self.max_AM())

    def add(self, bf):

        # Add a new BasisFunction to the set 

        if type(bf) == BasisFunction:
            self.basisfunctions.append(bf)
            self.slices.append(slice(self.nbf, self.nbf + bf.ncart))
            self.nbf += bf.ncart
        else:
            raise TypeError('Cannot add a {} to BasisSet'.format(type(bf)))

    def max_AM(self):

        # Return the maximum angular momentum within the set

        return max(self.basisfunctions).type

def os_recursion(PA, PB, alpha, AMa, AMb):

    # Perform the Osaka-Saika recustion for a pair of primitive functions
    
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
    # Return a block with overlap of all Cartesian functions spanned by the two basis functions
    # e.g. for (S|P) the output is a 1x3 matrix containing (S|Px) (S|Py) and (S|Py) 

    # Initialize the output. The sizes are given by the number of Cartesians in each basis.
    S = np.zeros((bf1.ncart, bf2.ncart))

    # Define the centers A and B
    A = bf1.center
    B = bf2.center

    # Loop through every pair of primitives
    for p1 in range(bf1.nprimitive):
        for p2 in range(bf2.nprimitive):

            # Define the exponents for A and B
            aa = bf1.exp[p1]
            bb = bf2.exp[p2]

            # Define alpha
            alpha = aa + bb
            xi = aa*bb/alpha

            # Define kappa with a normalizing factor
            kappa = (np.pi/alpha)**(3/2)*np.exp(-xi*np.dot(A-B,A-B))

            # Define the vector P
            P = (aa*A + bb*B)/alpha

            # Call the OS recursion for the current primitives
            (x,y,z) = os_recursion(P-A, P-B, alpha, bf1.AM, bf2.AM) 

            # Initialize an auxiliar array to hold the useful recursion results
            W = np.zeros((bf1.ncart, bf2.ncart))

            # Loop through each cartesian function within the basisset 
            for kA in range(bf1.ncart):
                for kB in range(bf2.ncart):

                    # For each pair of cartesians save the integral of these primitives

                    # Make shortcuts for useful info (amount of angular momentum in each cartesian direction)
                    Ax = bf1.cart[kA][0]
                    Ay = bf1.cart[kA][1]
                    Az = bf1.cart[kA][2]

                    Bx = bf2.cart[kB][0]
                    By = bf2.cart[kB][1]
                    Bz = bf2.cart[kB][2]

                    W[kA, kB] = x[Ax, Bx]*y[Ay, By]*z[Az, Bz]

            # When done with the pair of primitives, pass the results to the output
            # Multiply by the appropriate contraction coefficients and kappa
            S += bf1.coef[p1]*bf2.coef[p2]*kappa*W

    return S

def overlap(bset):
    
    # Return the overlap matrix given a BasisSet

    # Initialize the S matrix
    S = np.zeros((bset.nbf, bset.nbf)) 

    # Loop through basis functions
    for i,b1,s1 in bset:
        for j,b2,s2 in bset:
            # Only build one half, copy the results over the diagonal
            if j > i:
                break
            S[s1,s2] = overlap_block(b1, b2)
            S[s2,s1] = S[s1,s2].T

    return S

def kinetic_block(bf1, bf2):

    # Compute the kinetic energy integral between two basis function.
    # Return a block with overlap of all Cartesian functions spanned by the two basis functions
    # e.g. for (S|P) the output is a 1x3 matrix containing (S|Px) (S|Py) and (S|Py) 

    # Initialize the output. The sizes are given by the number of Cartesians in each basis.
    T = np.zeros((bf1.ncart, bf2.ncart))

    # Define the centers A and B
    A = bf1.center
    B = bf2.center

    # Loop through every pair of primitives
    for p1 in range(bf1.nprimitive):
        for p2 in range(bf2.nprimitive):

            # Define the exponents for A and B
            aa = bf1.exp[p1]
            bb = bf2.exp[p2]

            # Define alpha
            alpha = aa + bb
            xi = aa*bb/alpha

            # Define kappa with a normalizing factor
            kappa = (np.pi/alpha)**(3/2)*np.exp(-xi*np.dot(A-B,A-B))

            # Define the vector P
            P = (aa*A + bb*B)/alpha

            # Call the OS recursion for the current primitives with one extra unit of angular momentum
            (x,y,z) = os_recursion(P-A, P-B, alpha, bf1.AM+1, bf2.AM+1) 

            # Initialize auxiliar arrays to hold the useful recursion results
            Ix = np.zeros((bf1.ncart, bf2.ncart))
            Iy = np.zeros((bf1.ncart, bf2.ncart))
            Iz = np.zeros((bf1.ncart, bf2.ncart))

            # Loop through each cartesian function within the basisset 
            for kA in range(bf1.ncart):
                for kB in range(bf2.ncart):

                    # For each pair of cartesians save the integral of these primitives

                    # Make shortcuts for useful info (amount of angular momentum in each cartesian direction)
                    Ax = bf1.cart[kA][0]
                    Ay = bf1.cart[kA][1]
                    Az = bf1.cart[kA][2]

                    Bx = bf2.cart[kB][0]
                    By = bf2.cart[kB][1]
                    Bz = bf2.cart[kB][2]
                    
                    # Get Ix
                    if Ax > 0 and Bx > 0:
                        Ix[kA, kB] += Ax*Bx*x[Ax-1,Bx-1]*y[Ay,By]*z[Az,Bz]

                    Ix[kA, kB] += 4*aa*bb*x[Ax+1,Bx+1]*y[Ay,By]*z[Az,Bz]

                    if Bx > 0:
                        Ix[kA, kB] += -2*aa*Bx*x[Ax+1,Bx-1]*y[Ay,By]*z[Az,Bz]

                    if Ax > 0:
                        Ix[kA, kB] += -2*bb*Ax*x[Ax-1,Bx+1]*y[Ay,By]*z[Az,Bz]

                    # Get Iy
                    if Ay > 0 and By > 0:
                        Iy[kA, kB] += Ay*By*y[Ay-1,By-1]*x[Ax,Bx]*z[Az,Bz]

                    Ix[kA, kB] += 4*aa*bb*y[Ay+1,By+1]*x[Ax,Bx]*z[Az,Bz]

                    if By > 0:
                        Iy[kA, kB] += -2*aa*By*y[Ay+1,By-1]*x[Ax,Bx]*z[Az,Bz]

                    if Ay > 0:
                        Iy[kA, kB] += -2*bb*Ay*y[Ay-1,By+1]*x[Ax,Bx]*z[Az,Bz]

                    # Get Iz
                    if Az > 0 and Bz > 0:
                        Iz[kA, kB] += Az*Bz*z[Az-1,Bz-1]*y[Ay,By]*x[Ax,Bx]

                    Iz[kA, kB] += 4*aa*bb*z[Az+1,Bz+1]*y[Ay,By]*x[Ax,Bx]

                    if Bz > 0:
                        Iz[kA, kB] += -2*aa*Bz*z[Az+1,Bz-1]*y[Ay,By]*x[Ax,Bx]

                    if Az > 0:
                        Iz[kA, kB] += -2*bb*Az*z[Az-1,Bz+1]*y[Ay,By]*x[Ax,Bx]

            # When done with the pair of primitives, pass the results to the output
            # Multiply by the appropriate contraction coefficients and kappa
            T += 0.5*bf1.coef[p1]*bf2.coef[p2]*kappa*(Ix + Iy + Iz)

    return T

def kinetic(bset):
    
    # Return the kinetic energy matrix given a BasisSet

    # Initialize the T matrix
    T = np.zeros((bset.nbf, bset.nbf)) 

    # Loop through basis functions
    for i,b1,s1 in bset:
        for j,b2,s2 in bset:
            # Only build half, and copy results over the diagonal
            if j > i:
                break
            T[s1,s2] = kinetic_block(b1, b2)
            T[s2,s1] = T[s1,s2].T

    return T

def dipole_block(bf1, bf2):

    # Compute the three dipole moment integrals between two basis function.
    # Return a block with overlap of all Cartesian functions spanned by the two basis functions for each Dx, Dy, and Dz
    # e.g. for (S|P) the output is a 1x3 matrix containing (S|Px) (S|Py) and (S|Py) for each Dx, Dy and Dz 

    # Initialize the output. The sizes are given by the number of Cartesians in each basis.
    Dx = np.zeros((bf1.ncart, bf2.ncart))
    Dy = np.zeros((bf1.ncart, bf2.ncart))
    Dz = np.zeros((bf1.ncart, bf2.ncart))

    # Define the centers A and B
    A = bf1.center
    B = bf2.center

    # Loop through every pair of primitives
    for p1 in range(bf1.nprimitive):
        for p2 in range(bf2.nprimitive):

            # Define the exponents for A and B
            aa = bf1.exp[p1]
            bb = bf2.exp[p2]

            # Define alpha
            alpha = aa + bb
            xi = aa*bb/alpha

            # Define kappa with a normalizing factor
            kappa = (np.pi/alpha)**(3/2)*np.exp(-xi*np.dot(A-B,A-B))

            # Define the vector P
            P = (aa*A + bb*B)/alpha

            # Call the OS recursion for the current primitives with a unit of angular momentum added to the first basis function
            (x,y,z) = os_recursion(P-A, P-B, alpha, bf1.AM+1, bf2.AM) 

            # Initialize an auxiliar array to hold the useful recursion results
            Wx = np.zeros((bf1.ncart, bf2.ncart))
            Wy = np.zeros((bf1.ncart, bf2.ncart))
            Wz = np.zeros((bf1.ncart, bf2.ncart))

            # Loop through each cartesian function within the basisset 
            for kA in range(bf1.ncart):
                for kB in range(bf2.ncart):

                    # For each pair of cartesians save the integral of these primitives

                    # Make shortcuts for useful info (amount of angular momentum in each cartesian direction)
                    Ax = bf1.cart[kA][0]
                    Ay = bf1.cart[kA][1]
                    Az = bf1.cart[kA][2]

                    Bx = bf2.cart[kB][0]
                    By = bf2.cart[kB][1]
                    Bz = bf2.cart[kB][2]

                    Wx[kA, kB] = x[Ax+1, Bx]*y[Ay, By]*z[Az, Bz] + A[0]*x[Ax,Bx]*y[Ay,By]*z[Az,Bz]
                    Wy[kA, kB] = x[Ax, Bx]*y[Ay+1, By]*z[Az, Bz] + A[1]*x[Ax,Bx]*y[Ay,By]*z[Az,Bz]
                    Wz[kA, kB] = x[Ax, Bx]*y[Ay, By]*z[Az+1, Bz] + A[2]*x[Ax,Bx]*y[Ay,By]*z[Az,Bz]

            # When done with the pair of primitives, pass the results to the output
            # Multiply by the appropriate contraction coefficients and kappa
            # Negative taking in account charge of electron
            Dx -= bf1.coef[p1]*bf2.coef[p2]*kappa*Wx
            Dy -= bf1.coef[p1]*bf2.coef[p2]*kappa*Wy
            Dz -= bf1.coef[p1]*bf2.coef[p2]*kappa*Wz

    return (Dx, Dy, Dz)

def dipole(bset):
    
    # Return the dipole moment  matrices given a BasisSet

    # Initialize the D matrices
    Dx = np.zeros((bset.nbf, bset.nbf))
    Dy = np.zeros((bset.nbf, bset.nbf))
    Dz = np.zeros((bset.nbf, bset.nbf))

    # Loop through basis functions
    for i,b1,s1 in bset:
        for j,b2,s2 in bset:
            # Only build half, and copy results over the diagonal
            if j > i:
                break
            dx, dy, dz = dipole_block(b1, b2)
            Dx[s1,s2] = dx
            Dy[s1,s2] = dy
            Dz[s1,s2] = dz
            Dx[s2,s1] = dx.T
            Dy[s2,s1] = dy.T
            Dz[s2,s1] = dz.T

    return (Dx, Dy, Dz)
