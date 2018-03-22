from Stupebrett import Stupebrett
import numpy as np


stupebrett = Stupebrett()




def main():
    n = 10
    h=0.2

    # Oppgave 4c
    # y''''(c) = (1/h^4) * AYe
    Ye_vector = stupebrett.fasit_y(n)
    A = stupebrett.lagA(n)
    AYvector = np.dot(A, Ye_vector)
    Yc_fourthDerivative = AYvector/ h**4

    print("The Ye vector: ", Ye_vector)
    print("The A matrix: ", A)
    print("Yc fourthDerivative: ", Yc_fourthDerivative)

    # Oppgave 4d
    # y''''(e) = (b/h^4)
    # Foroverfeil = ||Ye - Yc||
    b_vector = stupebrett.lagB(n)
    Ye_fourthDerivative = b_vector/ h**4
    print("The b-vector: ", b_vector)
    print("Ye fourthDerivative: ", Ye_fourthDerivative)

    diff = Ye_vector - b_vector
    print("Forward error: ", diff)
















if __name__ == '__main__':
    main()