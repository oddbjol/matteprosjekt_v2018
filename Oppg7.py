from Stupebrett import Stupebrett

import numpy as np
from numpy.linalg import cond
import matplotlib.pyplot as plt

EPS = np.finfo(np.float).eps

def main():
    brett = Stupebrett()

    MAX_COND = 6
    L = 2

    x = np.zeros(11)
    numerisk_svar = np.zeros(11)
    kondA = np.zeros(11)
    teoretisk_feil = np.zeros(11)
    feil = np.zeros(11)

    for i in range(0, 11):
        n = 10 * 2**(i+1)

        x[i] = n
        numerisk_svar[i] = brett.finn_y(n, Stupebrett.kraft_av_person)[-1]

        A = brett.lagA(n)

        if i < MAX_COND:
            kondA[i] = cond(A.toarray())

        teoretisk_feil[i] = L**2 / n**2


    # Vi "ekstrapolerer" de høyeste verdiene av kondA basert på de forrige.
    mult = kondA[MAX_COND-1] / kondA[MAX_COND-2]
    for i in range(MAX_COND, 11):
        kondA[i] = kondA[i-1] * mult

    feil = teoretisk_feil + kondA*EPS

    lavest_feil_index = np.where(feil == feil.min())[0][0]

    print(lavest_feil_index)
    print(20 * 2**lavest_feil_index)
    print(numerisk_svar[lavest_feil_index], "meter forflytning i L")

    print(('n' + "\t" + 'numerisk svar' + "\t" + 'feil').expandtabs(30))
    for i in range(11):
        print((str(x[i]) + "\t" + str(numerisk_svar[i]) + "\t" + str(feil[i])).expandtabs(30))


if __name__ == '__main__':
    main()