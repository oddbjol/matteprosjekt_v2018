from Stupebrett import Stupebrett
import numpy as np
from numpy import linalg as la


stupebrett = Stupebrett()




def main():
    n = 10
    h=0.2

    # Oppgave 4c
    # y''''(c) = (1/h^4) * AYe
    Ye_vector = stupebrett.fasit_y(n)
    A = stupebrett.lagA(n)

    AYe_vector = np.dot(A.toarray(), Ye_vector)
    Yc_fourthDerivative = AYe_vector/ h**4

    print("Resultater fra Oppgave 4 c")
    print()
    print("Ye-vektoren: ", Ye_vector, sep="\n")
    print()
    #print("A matrisen: ", A)
    print("Yc-vektor fjerdederivert: ", Yc_fourthDerivative, sep="\n")




if __name__ == '__main__':
    main()