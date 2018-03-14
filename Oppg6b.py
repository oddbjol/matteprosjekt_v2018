from Stupebrett import Stupebrett

import numpy as np
import matplotlib.pyplot as plt

def main():

    brett = Stupebrett()

    x = np.zeros(12)
    fasit_svar = np.zeros(12)
    numerisk_svar = np.zeros(12)

    for i in range(11 + 1):
        n = 20 * 2**i

        x[i] = n
        numerisk_svar[i] = brett.finn_y(n, Stupebrett.kraft_av_haug)[-1]
        fasit_svar[i] = brett.fasit_y_medhaug(n)[-1]

    feil = np.abs(fasit_svar - numerisk_svar)

    for i in range(11 + 1):
        print(x[i], "\t", numerisk_svar[i], "\t", fasit_svar[i])

    plt.plot(x, -numerisk_svar, color='blue')
    plt.plot(x, -fasit_svar, 'r-')

    plt.show()


if __name__ == '__main__':
    main()
