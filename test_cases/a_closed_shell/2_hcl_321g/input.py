Settings = dict()

Settings["basis"] = "3-21g"
Settings["df_basis"] = "3-21g"
Settings["molecule"] = """
  0 1
  H
  Cl 1 R
  R = 1.376
  symmetry c1
"""
Settings["nalpha"] = 9
Settings["nbeta"] = 9
Settings["scf_max_iter"] = 50
Settings["cc_max_iter"]=75
