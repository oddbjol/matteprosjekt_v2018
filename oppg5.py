from Stupebrett import Stupebrett
import numpy as np
import matplotlib.pyplot as plt


brett = Stupebrett(L=2, w=0.3, d=0.03, p=480, E=1.3*10**10)

def pointOferror():

        x = np.zeros(11)
        fasit = np.zeros(11)
        kalkulering = np.zeros(11)

        for i in range(0, 11):
            n = 10 * 2 ** (i + 1)

            x[i] = n
            kalkulering[i] = brett.finn_y(n)[-1]
            fasit[i] = brett.fasit_y(n)[-1]

        feil_i_L = np.abs(fasit - kalkulering)

        for i in range(0, 11):
            print(int(x[i]), '\t',"kaluering", kalkulering[i], '\t',"fasit", fasit[i], '\t',"feil", feil_i_L[i])


def plot_kondisjonsTallA():

    kondisjonsA = np.zeros(11)
    x = np.zeros(11)
    feil = np.zeros(11)
    L = 2

    for i in range(0, 11):
        n = 10 * 2 ** (i + 1)

        x[i] = n
        # feil[i] = L ** 2 / n ** 2
        kondisjonsA[i] = np.linalg.cond(brett.lagA(n).toarray())
        print(x, kondisjonsA[i])

    plt.loglog(x, kondisjonsA, color='green', marker='o', markersize=2, label='kondisjonsTall-A', basex=2)
    plt.loglog(x, feil, color='red', marker='o', markersize=2, label='feil', basex=2)

    plt.legend()
    plt.show()


#print(plot_kondisjonsTallA())
#print('n\tkondisjonsA\t\t\tfasit\t\t\tfeil')
pointOferror()






    # feil_i_L = np.zeros(12)
    # x = np.zeros(12)


    # for i in range(0, 11):
    #     n = 10 * 2**(i + 1)
    #
    #     feil_i_L = np.abs(brett.fasit_y(n) - brett.finn_y(n))[n - 1]
    #     kondisjonsTallA = np.linalg.cond(brett.lagA(n).toarray())
    #     print(n, feil_i_L, "\t", kondisjonsTallA)