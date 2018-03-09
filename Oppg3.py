from Stupebrett import Stupebrett

brett = Stupebrett(L=2, w=0.3, d=0.03, p=480, E=1.3*10**10)

print("beregnet y-vektor: ", brett.solve(20))
print("y-vektor fra fasit: ", brett.correct_displacement(20))