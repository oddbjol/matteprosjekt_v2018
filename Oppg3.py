import scipy as sp
import numpy as np

from Stupebrett import Stupebrett

n = 20 # antall segmenter
h = 2/20 #segmentlengde

brett = Stupebrett(L=2, w=0.3, d=0.03, p=480, E=1.3*10**10)

y = brett.solve(20)


print("beregnet y-vektor: ", y)
print("y-vektor fra fasit: ", brett.correct_displacement(20))