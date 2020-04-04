Settings = dict()

Settings["basis"] = "sto-3g"
Settings["molecule"] = """
  0 1
  O
  H 1 R
  H 1 R 2 A
  R = 0.9
  A = 104.5
  symmetry c1
"""
Settings["nalpha"] = 5
Settings["nbeta"] = 5
Settings["scf_max_iter"] = 50
Settings["active_space"] = 'full'
Settings["excitation_level"] = 'full'
