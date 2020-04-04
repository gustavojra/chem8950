from input import Settings

if 'active_space' in Settings:
    print('TEST')
    from CI import compute_CI
    compute_CI(Settings) 
else:
    from uhf import compute_uhf
    compute_uhf(Settings)
