import sys
import psi4
sys.path.append('/home/aroeira/chem8950/Combined/')
sys.path.append('/home/aroeira/Tchau-Spin/CC_codes/')

from RCCD import RCCD
from CCD import compute_CCD
from input import Settings


psi4.core.be_quiet()
mol = psi4.geometry(Settings["molecule"])
mol.update_geometry()

psi4.set_options({
    'basis' : Settings['basis'],
    'scf_type' : 'pk',
    'puream' : 0,
    'e_convergence': 1e-12,
    'reference' : 'rhf'})

Ehf, wfn = psi4.energy('scf', return_wfn = True)
A = RCCD(wfn, CC_CONV = 8, E_CONV = 12)

B = compute_CCD(Settings)

print('RCCD Energy:     {:<15.10f}'.format(A))
print('SOCCD Energy:    {:<15.10f}'.format(B))

