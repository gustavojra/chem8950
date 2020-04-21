Settings = dict()

Settings["basis"] = "cc-pvdz"
Settings["df_basis"] = "cc-pvdz-ri"
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
Settings["cc_max_iter"]=75
