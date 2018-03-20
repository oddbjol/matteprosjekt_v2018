from Stupebrett import Stupebrett
import numpy as np


stupebrett = Stupebrett()




def main():
    n = 10
    h=0.2
    Ye_vector = stupebrett.fasit_y(n)
    A = stupebrett.lagA(n)
    AYvector = np.dot(A.toarray(), Ye_vector)
    Yx_fourthDerivative = AYvector/ h**4

    print("The Ye vector: ", Ye_vector)
    print("The A matrix: ", A)
    print("Yx fourthDerivative: ", Yx_fourthDerivative)


    #b_vector = stupebrett.lagB(n)
    #print("The b-vector: ", b_vector)





if __name__ == '__main__':
    main()