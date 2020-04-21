import sys
sys.path.append('/home/aroeira/chem8950/Combined/')
from dfmp2 import compute_DFMP2
import os

print('Running tests for Closed-shell')

closed_shell_mols = ['1_ammonia_dz',    '2_hcl_321g',      '3_methane_631g',  '4_water_dz',      '5_water_sto3g',   '6_krypton']

os.chdir('a_closed_shell')

for m in closed_shell_mols:
    print('Running {}'.format(m))
    os.chdir(m)
    os.system('python /home/aroeira/chem8950/Combined/dfmp2.py')
    print('Running {}'.format(m))
    os.chdir('..')
    input('\nHit any key...')
