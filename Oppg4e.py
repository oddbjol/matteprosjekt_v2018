from Stupebrett import Stupebrett
import numpy as np
from numpy import linalg as la
import math

stupebrett = Stupebrett()

#Skal sammenligne svaret i oppgave 3 (yc med n= 20) MED Ye (n=20)
# Skal også finne foroverfeilen ||Yc - ye||

def printVectorInbase2(vector):
    mystr = "[ "
    for i in range(vector.size):
        mantissa, exp = math.frexp(vector[i])
        mystr += str(mantissa) + "*2^" + str(exp) + "  "
    mystr += " ]"
    print(mystr)

def printNumberInBase2(number):
    mantissa, exp = math.frexp(number)
    print(str(mantissa) + "*2^" + str(exp))


def main():
    n = 20
    h=0.1



    # Finner Yc-vetoren med n = 20 fra oppgave 3
    Yc = stupebrett.finn_y(n)
    print("Yc-vektor for stupebrett med n=20: ", Yc)

    # Finner Ye-vektoren med n = 20 for denne oppgaven.
    Ye = stupebrett.fasit_y(n)
    print("Ye-vektor med n=20 for denne oppgaven: ", Ye)

    diff = Ye - Yc

    print("Differansen mellom Ye og Yc vektor med base 10: ", diff)

    print("Differansen mellom Ye og Yc-vektor med base 2: ")
    printVectorInbase2(diff)


    forwardError = la.norm((diff), ord=1)
    print("Differansen mellom Ye og Yc-vektor med base 2 og enernormen: ")
    printNumberInBase2(forwardError)



if __name__ == '__main__':
    main()