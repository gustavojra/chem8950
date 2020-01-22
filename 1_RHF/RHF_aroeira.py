import psi4
import numpy as np
import scipy.linalg as sp
from input import Settings


# SETUP INITIAL CONDITIONS

molecule = psi4.geometry(Settings['molecule'])
molecule.update_geometry()

if Settings['nalpha'] != Settings['nbeta']:
    raise NameError('No open shell stuff plz')

ndocc = Settings['nalpha']
Vnuc = molecule.nuclear_repulsion_energy()

basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'])

mints = psi4.core.MintsHelper(basis)

# COMPUTE INTEGRALS

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

# CONSTRUCT THE ORTHOGONALIZER

A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = A.np

# INITIAL DENSITY MATRIX

