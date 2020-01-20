import psi4
import numpy as np
from input import Settings


# Setup initial conditions

molecule = psi4.geometry(Settings['molecule'])
molecule.update_geometry()

if Settings['nalpha'] != Settings['nbeta']:
    raise NameError('No open shell stuff plz')

ndocc = Settings['nalpha']
Vnuc = molecule.nuclear_repulsion_energy()

basis = psi4.core.BasisSet.build(molecule, 'BASIS', Settings['basis'])

mints = psi4.core.MintsHelper(basis)


