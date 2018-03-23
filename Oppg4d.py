from Stupebrett import Stupebrett
import numpy as np
from numpy import linalg as la


stupebrett = Stupebrett()




def main():
    n = 10
    h=0.2




    # Oppgave 4d
    # y''''(e) = (b/h^4)
    # Foroverfeil = ||Ye - Yc|| 1-norm
    # Relativ foroverfeil = (||Ye - Yc|| 1-norm) / ||Ye ||
    Ye_vector = stupebrett.fasit_y(n)
    A = stupebrett.lagA(n)
    AYvector = np.dot(A.toarray(), Ye_vector)
    Yc_fourthDerivative = AYvector/ h**4
    b_vector = stupebrett.lagB(n)
    Ye_fourthDerivative = b_vector/ h**4
    print("Resultater fra Oppgave 4 d")
    print("b-vektoren: ", b_vector, sep="\n")
    print()
    print("Ye-fjerderivert: ", Ye_fourthDerivative, sep="\n")

    print()
    forwardErrorWithInfNorm = la.norm((Ye_fourthDerivative - Yc_fourthDerivative), np.inf)
    print("Foroverfeil med uenendelignorm: ",forwardErrorWithInfNorm)
    print()
    relativeForwardErrorWithInfNorm = la.norm((forwardErrorWithInfNorm/Ye_fourthDerivative), np.inf)
    print("Relativ foroverfeil med uendelignorm:", relativeForwardErrorWithInfNorm)

    # Skal Anta Relativ bakoverfeil er Epsilon - mach = 2^-52
    # Skal regne ut feilforstørring og sammenenlign med kondisjonstallet til A
    # Hva ser jeg???

    errorMagnificationFactor = relativeForwardErrorWithInfNorm/ 2**-52
    print()
    print("Feilforstørringsfaktoren: ", errorMagnificationFactor)
    A = A.toarray()
    conditionNumberOfA = la.cond(A)
    print()
    print("Kondisjonstall for matrise A: ", conditionNumberOfA)



if __name__ == '__main__':
    main()