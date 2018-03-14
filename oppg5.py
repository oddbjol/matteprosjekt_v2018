import numpy as np
import matplotlib as plt



from Stupebrett import Stupebrett

brett = Stupebrett(L=2, w=0.3, d=0.03, p=480, E=1.3*10**10)


def utregning():

    for i in range(0, 12):

        n = 20 * (2**i)

        feil_i_L = np.abs(brett.fasit_y(n) - brett.finn_y(n))[n - 1]
        #print(i, feil_i_L)


    print("ferdig", "20.... 10 * 2^11 oppnåd")




utregning()



