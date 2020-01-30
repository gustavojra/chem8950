Settings = dict()

Settings["basis"] = "cc-pVDZ"
Settings["molecule"] = """
  1 2
  O
  H 1 R
  H 1 R 2 A
  R = 1.0
  A = 104.5
  symmetry c1
"""
Settings["nalpha"] = 5
Settings["nbeta"] = 4
Settings["scf_max_iter"] = 50
