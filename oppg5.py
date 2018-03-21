from Stupebrett import Stupebrett
import numpy as np
import matplotlib.pyplot as plt



def main():

    brett = Stupebrett()

    feil_i_L = np.zeros(12)
    x = np.zeros(12)
    laga = np.zeros(12)

    for i in range(0, 12):
        n = 20 * (2**i)

        x[i] = n
        feil_i_L[i] = np.abs(brett.fasit_y(n) - brett.finn_y(n))[n - 1]
        laga = brett.lagA(i)
        # print(i, feil_i_L)

    for i in range(11 + 1):
        print(x[i], "\t", feil_i_L, "\t", brett.lagA(x))

    plt.plot(x, feil_i_L, color='blue')
    plt.plot(x, laga, color='red')
    plt.show()


if __name__ == '__main__':
    main()
