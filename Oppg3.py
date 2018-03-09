import numpy as np

from Stupebrett import Stupebrett

np.set_printoptions(linewidth=1000)

brett = Stupebrett(L=2, w=0.3, d=0.03, p=480, E=1.3*10**10)

# Disse bør flyttes til sine respektive oppgave-filer senere...

print("beregnet y-vektor: ", brett.finn_y(20))
print("y-vektor fra fasit: ", brett.fasit_y(20))

print("y-vektor fra fasit, med haug: ", brett.fasit_y_medlast(20))
