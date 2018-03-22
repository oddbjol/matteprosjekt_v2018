from Stupebrett import Stupebrett
import numpy as np
from numpy import linalg as la

stupebrett = Stupebrett()

#Skal sammenligne svaret i oppgave 3 (yc med n= 20) MED Ye (n=20)
# Skal også finne foroverfeilen ||Yc - ye||


def main():
    n = 20
    h=0.1



    # Finner Yc-vetoren med n = 20 fra oppgave 3
    Yc = stupebrett.finn_y(n)
    print("Yc-vektor for stupebrett med n=20: ", Yc)

    # Finner Ye-vektoren med n = 20 for denne oppgaven.
    Ye = stupebrett.fasit_y(n)
    print("Ye-vektor med n=20 for denne oppgaven: ", Ye)

    diff = abs(Ye - Yc)
    print("Differansen mellom Ye og Yc vektor med n=20: ", diff)

    print()
    print()
    print()


    # Oppgave 4c
    # y''''(c) = (1/h^4) * AYe

    A = stupebrett.lagA(n)
    AYvector = np.dot(A.toarray(), Ye)
    Yc_fourthDerivative = AYvector/ h**4

    print("The Ye vector: ", Ye)
    print("The A matrix: ", A)
    print("Yc fourthDerivative: ", Yc_fourthDerivative)

    # Oppgave 4d
    # y''''(e) = (b/h^4)
    # Foroverfeil = ||Ye - Yc|| 1-norm
    # Relativ foroverfeil = (||Ye - Yc|| 1-norm) / ||Ye ||
    b_vector = stupebrett.lagB(n)
    Ye_fourthDerivative = b_vector/ h**4
    print("The b-vector: ", b_vector)
    print("Ye fourthDerivative: ", Ye_fourthDerivative)

    #forwardError = la.norm((Ye_fourthDerivative - Yc_fourthDerivative), ord=1)
    #print("Forward error with norm 1: ", forwardError)

    #relativeForwardError = la.norm((forwardError/ Ye_fourthDerivative), ord=1)
    #print("Relative forvarderror with norm 1: ", relativeForwardError)

    forwardErrorWithInfNorm = la.norm((Ye_fourthDerivative - Yc_fourthDerivative), np.inf)
    print("Forward error with infinity norm: ",forwardErrorWithInfNorm)

    relativeForwardErrorWithInfNorm = la.norm((forwardErrorWithInfNorm/Ye_fourthDerivative), np.inf)
    print("Relative forward error with infinity norm:", relativeForwardErrorWithInfNorm)

    # Skal Anta Relativ bakoverfeil er Epsilon - mach = 2^-52
    # Skal regne ut feilforstørring og sammenenlign med kondisjonstallet til A
    # Hva ser jeg???

    errorMagnificationFactor = relativeForwardErrorWithInfNorm/ 2**-52
    print("Feilforstørring: ")
    print("The error magnification factor: ", errorMagnificationFactor)
    #b = A.reshape(10,10)
    #print(b)
    A = A.toarray()
    conditionNumberOfA = la.cond(A)
    print("Conditionnumber of A: ", conditionNumberOfA)























if __name__ == '__main__':
    main()