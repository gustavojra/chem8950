Settings = dict()

Settings["basis"] = "3-21g"
Settings["molecule"] = """
  0 2
  O
  N 1 R
  R = 1.15
  symmetry c1
"""
Settings["nalpha"] = 8
Settings["nbeta"] = 7
Settings["scf_max_iter"] = 50
