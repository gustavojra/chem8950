Settings = dict()

Settings["basis"] = "cc-pvdz"
Settings["molecule"] = """
  0 1
  O
  H 1 R
  H 1 R 2 A
  R = 1.0
  A = 104.5
  symmetry c1
"""
Settings["nalpha"] = 5
Settings["nbeta"] = 5
Settings["scf_max_iter"] = 50
