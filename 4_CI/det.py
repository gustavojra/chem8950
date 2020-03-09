import numpy as np

class Determinant:

    # Class that represents a Slater determinant
    # 'a' and 'b' inputs are strings for the occupancy of alpha and beta orbitals
    # For example a = '11100', b = '11100' is a determinant with all electrons in the lowest orbitals

    def __init__(self, a, b):

        self.alpha = int(a[::-1], 2)       
        self.beta  = int(b[::-1], 2)


        # Store the order (length) of the determinant
        self.order = len(a)

        # Ensure alpha e beta strings have the same length
        if len(a) != len(b):
            raise NameError('Alpha and Beta strings must have the same length')

    def __str__(self):
        
        # Create a string representation of the determinant. 

        out = '-'*(3+self.order) + '\n'
        out += '\u03B1: ' + np.binary_repr(self.alpha, width=abs(self.order))[::-1]
        out += '\n'    
        out += '\u03B2: ' + np.binary_repr(self.beta, width=abs(self.order))[::-1]
        out += '\n' + '-'*(3+self.order)
        return out

    def __eq__(self, other):

        # Check if two determinants are the same

        return self.alpha == other.alpha and self.beta == self.neta

    def __sub__(self, other,v=False):

        # Subtracting two determinants yields the number of different orbitals between them.
        # The operations is commutative
        # Note that in this formulation single excitation => 2 different orbitals

        a = bin(self.alpha ^ other.alpha).count("1")
        b = bin(self.beta ^ other.beta).count("1")
        return a + b

    def alpha_list(self):
        
        # Returns a list representing alpha electrons. For example: '11100' -> [1, 1, 1, 0, 0]
        # Note: output has left to right ordering (as input)

        return np.array([int(x) for x in list(np.binary_repr(self.alpha, width=abs(self.order)))])[::-1]

    def beta_list(self):

        # Returns a list representing beta electrons. For example: '11100' -> [1, 1, 1, 0, 0]
        # Note: output has left to right ordering (as input)

        return np.array([int(x) for x in list(np.binary_repr(self.beta, width=abs(self.order)))])[::-1]

    def set_subtraction(self, other):
        
        # Return the bits that are occupied (on) in self but not in other

        a = self.alpha & (~other.alpha)
        b = self.beta & (~other.beta)

        return (a, b)

    def set_subtraction_list(self, other):
        
        # Return a list with the position of orbitals that are in self but not in other

        ba, bb = self.set_subtraction(other)

        i = 1
        a = []
        b = []
        p = 0
        # Loop through the binary ba, bb and check where it is on
        while i <= max(ba, bb):
            if i & ba:
                a.append(p)
            if i & bb:
                b.append(p)
            p += 1
            i = i << 1

        return (a, b)

    def phase(self, other):

        # Returns the phase to place the two determinants in maximum coincidence

        # If the determinants differ by more than double excitations, return zero
        if self - other > 4:
            return 0

        det1  = np.binary_repr(self.alpha, width=abs(self.order)) 
        det1 += np.binary_repr(self.beta, width=abs(self.order)) 
        det1 = int(det1, 2)

        det2  = np.binary_repr(other.alpha, width=abs(other.order)) 
        det2 += np.binary_repr(other.beta, width=abs(other.order)) 
        det2 = int(det2, 2)

        # Get the orbitals indexes of interest
        x1 = det1 & (det1 ^ det2)
        x2 = det2 & (det1 ^ det2)

        if self - other == 4:
            p = 0
            i = 1
            px1 = []
            px2 = []
            while i < max(x1,x2):
                if i & (det1 & det2):
                        p += 1
                if i & x1:
                    px1.append(p)
                if i & x2:
                    px2.append(p)
                i = i << 1
            p = abs(px1[0]-px2[0]) + abs(px1[1]-px2[1])
            return (-1)**p
            
        # If they only differ by a pair of orbitals:
        else:
            l = min(x1,x2)
            u = max(x1,x2)
            p = 0
            while l < u:
                u = u >> 1
                if u & (det1 & det2):
                    p += 1
            return (-1)**p
            
