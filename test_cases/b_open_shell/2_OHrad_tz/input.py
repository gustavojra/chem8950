Settings = dict()

Settings["basis"] = "cc-pVTZ"
Settings["df_basis"] = Settings["basis"] + "-ri"
Settings["molecule"] = """
  0 2
  O
  H 1 R
  R = 1.0
  symmetry c1
"""
Settings["nalpha"] = 5
Settings["nbeta"] = 4
Settings["scf_max_iter"] = 50
Settings["cc_max_iter"]=75
