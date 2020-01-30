Settings = dict()

Settings["basis"] = "STO-3G"
Settings["molecule"] = """
  0 2
  O
  H 1 R
  R = 1.0
  A = 104.5
  symmetry c1
"""
Settings["nalpha"] = 5
Settings["nbeta"] = 4
Settings["scf_max_iter"] = 100
