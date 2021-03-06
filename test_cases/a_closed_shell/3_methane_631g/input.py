Settings = dict()

Settings["basis"] = "6-31g"
Settings["df_basis"] = "6-31g"
Settings["molecule"] = """
  0 1
  C       -4.0431729517      2.5597260362      0.0000003487
  H       -2.9337721641      2.5597251733     -0.0000015634
  H       -4.4129710787      3.2487329727      0.7869494539
  H       -4.4129737734      1.5337057469      0.2032239344
  H       -4.4129747942      2.8967402552     -0.9901704399
  symmetry c1
"""
Settings["nalpha"] = 5
Settings["nbeta"] = 5
Settings["scf_max_iter"] = 50
Settings["cc_max_iter"]=75
