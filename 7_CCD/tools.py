import numpy as np
# Miscellaneous

def emoji(key):
    stored = {
    "viva"   : b'\xF0\x9F\x8E\x89'.decode('utf-8'),
    "eyes"   : b'\xF0\x9F\x91\x80'.decode('utf-8'),
    "cycle"  : b'\xF0\x9F\x94\x83'.decode('utf-8'),
    "bolt"   : b'\xE2\x9A\xA1'.decode('utf-8'),
    "pin"    : b'\xF0\x9F\x93\x8C'.decode('utf-8'),
    "crying" : b'\xF0\x9F\x98\xAD'.decode('utf-8'),
    "pleft"  : b'\xF0\x9F\x91\x88'.decode('utf-8'),
    "whale"  : b'\xF0\x9F\x90\xB3'.decode('utf-8'),
    "books"  : b'\xF0\x9F\x93\x9A'.decode('utf-8'),
    "check"  : b'\xE2\x9C\x85'.decode('utf-8'),
    "0"      : b'\x30\xE2\x83\xA3'.decode('utf-8'),
    "1"      : b'\x31\xE2\x83\xA3'.decode('utf-8'),
    "2"      : b'\x32\xE2\x83\xA3'.decode('utf-8'),
    "3"      : b'\x33\xE2\x83\xA3'.decode('utf-8'),
    "4"      : b'\x34\xE2\x83\xA3'.decode('utf-8'),
    "5"      : b'\x35\xE2\x83\xA3'.decode('utf-8'),
    "6"      : b'\x36\xE2\x83\xA3'.decode('utf-8'),
    "7"      : b'\x37\xE2\x83\xA3'.decode('utf-8'),
    "8"      : b'\x38\xE2\x83\xA3'.decode('utf-8'),
    "9"      : b'\x39\xE2\x83\xA3'.decode('utf-8'),
    "ugh"    : b'\xF0\x9F\x91\xBE'.decode('utf-8'),
    "plug"   : b'\xF0\x9F\x94\x8C'.decode('utf-8'),
    "brinde" : b'\xF0\x9F\x8D\xBB'.decode('utf-8'),
    "X"      : b'\xE2\x9D\x8C'.decode('utf-8'),
    "O"      : b'\xE2\xAD\x95'.decode('utf-8'),
    "Obold"  : b'\xF0\x9F\x94\xB4'.decode('utf-8'),
    "bchart" : b'\xF0\x9F\x93\x8A'.decode('utf-8'),
    "top"    : b'\xF0\x9F\x94\x9D'.decode('utf-8'),
    "trophy" : b'\xF0\x9F\x8F\x86'.decode('utf-8'),
    "wine"   : '\U0001F377'
    }
    return stored[key]

def numoji(i):
    i = str(i)
    out = ''
    for l in i:
        out += emoji(l) + ' '
    return out

def printtensor(T):
    dims = np.shape(T)
    rank = len(dims)
    i_range = range(0,dims[0])
    j_range = range(0,dims[1])
    for i in i_range:
        for j in j_range:
            print('Page ({},{},*,*)\n'.format(i,j))
            printmatrix(T[i,j,:,:])

def printmatrix(M):
    dims = np.shape(M)
    rank = len(dims)
    out = ' '*5
    i_range = range(0,dims[0])
    j_range = range(0,dims[1])
    # header
    for j in j_range:
        out += '{:>11}    '.format(j)
    out += '\n'
    for i in i_range:
        out += '{:3d}  '.format(i)
        for j in j_range:
            out += '{:< 10.8f}    '.format(M[i,j])
        out += '\n'
    print(out)
