import numpy as np

from Stupebrett import Stupebrett

np.set_printoptions(linewidth=1000)


def main():

    brett = Stupebrett()

    print("beregnet y-vektor: ", brett.finn_y(10))


if __name__ == '__main__':
    main()