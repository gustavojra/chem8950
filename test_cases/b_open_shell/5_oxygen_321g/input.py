Settings = dict()

Settings["basis"] = "3-21g"
Settings["df_basis"] = "3-21g"
Settings["molecule"] = """
  0 3
  O
  O 1 R
  R = 1.3
  symmetry c1
"""
Settings["nalpha"] = 8
Settings["nbeta"] = 8
Settings["scf_max_iter"] = 50
