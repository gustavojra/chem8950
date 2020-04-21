Settings = dict()

Settings["basis"] = "cc-pVTZ"
Settings["df_basis"] = Settings["basis"] + "-ri"
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
Settings["cc_max_iter"]=75
