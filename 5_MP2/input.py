Settings = dict()
Settings['df_basis'] = "cc-pvtz-ri"
Settings["basis"] = "cc-pvtz"
Settings["molecule"] = """
    1 2
    O
    H 1 R
    H 1 R 2 A
    R = 0.9
    A = 104.5
    symmetry c1
"""
Settings["nalpha"] = 5
Settings["nbeta"] = 4
Settings["scf_max_iter"] = 50
