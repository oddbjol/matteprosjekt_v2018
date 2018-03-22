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

    AYvector = np.dot(A.toarray(), Ye_vector)
    Yc_fourthDerivative = AYvector/ h**4

    print("Ye-vektoren: ", Ye_vector, sep="\n")
    print("A matrisen: ", A)
    print("Yc-vektor fjerdederivert: ", Yc_fourthDerivative, sep="\n")



if __name__ == '__main__':
    main()