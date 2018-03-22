from Stupebrett import Stupebrett

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond

EPS = np.finfo(np.float).eps


def main():

    brett = Stupebrett()

    MAX_COND = 6
    L = 2  # meter

    x = np.zeros(11)
    fasit_svar = np.zeros(11)
    numerisk_svar = np.zeros(11)
    kondA = np.zeros(11)
    teoretisk_feil = np.zeros(11)

    for i in range(0, 11):
        n = 10 * 2**(i+1)

        x[i] = n
        numerisk_svar[i] = brett.finn_y(n, Stupebrett.kraft_av_haug)[-1]
        fasit_svar[i] = brett.fasit_y_medhaug(n)[-1]

        A = brett.lagA(n)

        if i < MAX_COND:
            kondA[i] = cond(A.toarray())

        teoretisk_feil[i] = (L/n) ** 2

    feil = np.abs(fasit_svar - numerisk_svar)

    print(('n' + "\t" + 'numerisk svar' + "\t" + 'fasitsvar' + "\t" + 'feil').expandtabs(30))
    for i in range(11):
        print((str(x[i]) + "\t" + str(numerisk_svar[i]) + "\t" + str(fasit_svar[i]) + "\t" + str(feil[i])).expandtabs(30))

    # Vi "ekstrapolerer" de høyeste verdiene av kondA basert på de forrige.
    mult = kondA[MAX_COND-1] / kondA[MAX_COND-2]
    for i in range(MAX_COND, 11):
        kondA[i] = kondA[i-1] * mult


    plt.title("Numerisk vs korrekt løsning")
    plt.loglog(x, -numerisk_svar, marker='o', markersize=4, label='Numerisk løsning', basex=2)
    plt.loglog(x, -fasit_svar, marker='o', markersize=4, label='Korrekt løsning', basex=2)

    plt.xlabel("n")
    plt.ylabel("meter")
    plt.legend()
    plt.show()


    plt.title("Feil")
    plt.loglog(x, feil, marker='o', markersize=4, label='Feil', basex=2, color='red')
    plt.loglog(x, kondA*EPS, marker='o', markersize=4, label='Kond(A)*EPS', basex=2)
    plt.loglog(x, teoretisk_feil, marker='o', markersize=4, label='Teoretisk feil', basex=2)
    #plt.loglog(x, np.abs(teoretisk_feil+(kondA*EPS))*1.5, marker='o', markersize=4, label='Kond(A)*EPS + Teoretisk_Feil', basex=2)

    plt.xlabel("n")
    plt.ylabel("feil (meter)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
