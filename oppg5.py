from Stupebrett import Stupebrett
import numpy as np
import matplotlib.pyplot as plt


def main():

    brett = Stupebrett()

    # global x
    # global kondisjonsTallA

    # feil_i_L = np.zeros(12)
    # x = np.zeros(12)


    for i in range(0, 12):
        n = 20 * (2**i)

        #x[i] = n
        feil_i_L = np.abs(brett.fasit_y(n) - brett.finn_y(n))[n - 1]
        kondisjonsTallA = np.linalg.cond(brett.lagA(n).toarray())
        print(n, feil_i_L, "\t", "condition", kondisjonsTallA)


        #print(i, feil_i_L)
        # print(kondisjonsTallA)
        # plt.plot(kondisjonsTallA)


    # for i in range(11 + 1):
    #     print(x[i], "\t", "N in", feil_i_L, "\t", "Feil")

    # print(np.linalg.cond(brett.lagA(x).toarray(x)))
    # print(kondisjonsTallA)

    #plt.plot(x, feil_i_L, color='blue')

    #plt.show()


if __name__ == '__main__':
    main()
