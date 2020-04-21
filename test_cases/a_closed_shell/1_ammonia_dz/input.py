Settings = dict()

Settings["basis"] = "3-21g"
Settings["df_basis"] = "cc-pvdz-ri"
Settings["molecule"] = """
  0 1
  N       -3.8452193033      2.7754820524     -0.0160543962
  H       -2.8021108858      2.7866454563      0.0011715583
  H       -4.1832422319      3.7619652723     -0.0490374569
  H       -4.1832466832      2.3424694447      0.8709264201
  symmetry c1
"""
Settings["nalpha"] = 5
Settings["nbeta"] = 5
Settings["scf_max_iter"] = 50
Settings["active_space"] = 'oooaaaaa'
Settings["excitation_level"] = 'full'
Settings["cc_max_iter"]=75
