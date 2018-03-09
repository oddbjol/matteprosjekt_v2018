# OBSOBSOBS:
# For å bruke denne unittest-fila, installer først pytest med pip og kjør deretter pytest i terminal.
#
# pip3 install pytest
# pytest -v

import numpy as np
from scipy.constants import g

from Stupebrett import Stupebrett

L = 1
w = 1
d = 1
p = 1
E = 1
I = w * d ** 3 / 12

brett = Stupebrett(1, 1, 1, 1, 1)


# Sjekker at et ideelt stupebrett på 1x1x1 og alle konstanter satt til 1 gir forventet b-vektor.
def test_lagB_basic():
    n = 1
    h = 1
    b = brett.lagB(n)
    f = h * w * d * p * -g
    b_test = f * (h ** 4 / (E * I))
    assert (b == [b_test] * n).all()


# Sjekker at komponentene til b er konstant over hele brettet når man kun har egenvekt på brettet.
def test_lagB_konstant_kraft():
    b = brett.lagB(1)
    assert b.min() == b.max()

    b = brett.lagB(2)
    assert b.min() == b.max()

    b = brett.lagB(10)
    assert b.min() == b.max()

    b = brett.lagB(100)
    assert b.min() == b.max()


# Sjekker at komponentene i b-vektoren øker invers proporsjonalt med n**5 som forventet.
def test_lagB_proporsjonal_med_n():
    b1 = brett.lagB(10)
    b2 = brett.lagB(20)
    b3 = brett.lagB(30)
    b4 = brett.lagB(40)

    assert np.isclose(b2[0], b1[0] / 2 ** 5)
    assert np.isclose(b3[0], b1[0] / 3 ** 5)
    assert np.isclose(b4[0], b1[0] / 4 ** 5)


# Sjekker at den minste mulige A-matrisen ser ut som den skal.
def test_lagA_minimum():
    A = brett.lagA(5)

    assert np.isclose(A[0, :].toarray(), [16, -9, 8 / 3, -1 / 4, 0]).all()
    assert np.isclose(A[1, :].toarray(), [-4, 6, -4, 1, 0]).all()
    assert np.isclose(A[2, :].toarray(), [1, -4, 6, -4, 1]).all()
    assert np.isclose(A[3, :].toarray(), [0, 16 / 17, -60 / 17, 72 / 17, -28 / 17]).all()
    assert np.isclose(A[4, :].toarray(), [0, -12 / 17, 96 / 17, -156 / 17, 72 / 17]).all()


# Sjekker at den nest minste mulige A-matrisen ser ut som den skal.
def test_lagA_minimum_pluss():
    A = brett.lagA(6)

    assert np.isclose(A[0, :].toarray(), [16, -9, 8 / 3, -1 / 4, 0, 0]).all()
    assert np.isclose(A[1, :].toarray(), [-4, 6, -4, 1, 0, 0]).all()
    assert np.isclose(A[2, :].toarray(), [1, -4, 6, -4, 1, 0]).all()
    assert np.isclose(A[3, :].toarray(), [0, 1, -4, 6, -4, 1]).all()
    assert np.isclose(A[4, :].toarray(), [0, 0, 16 / 17, -60 / 17, 72 / 17, -28 / 17]).all()
    assert np.isclose(A[5, :].toarray(), [0, 0, -12 / 17, 96 / 17, -156 / 17, 72 / 17]).all()


# Sjekker at brettet konstrueres riktig.
def test_init():
    L = 1
    w = 1
    d = 1
    p = 1
    E = 1
    I = w * d ** 3 / 12

    brett2 = Stupebrett(L, w, d, p, E, lambda x: x * 3)

    assert brett2.L == L
    assert brett2.w == w
    assert brett2.d == d
    assert brett2.p == p
    assert brett2.E == E
    assert brett2.I == I
    assert brett2.force_func(1) == 3


# Sjekker at løsning blir ca. lik fasit.
def test_solve_vs_correct():
    assert np.isclose(brett.solve(10), brett.correct_displacement(10)).all()
    assert np.isclose(brett.solve(20), brett.correct_displacement(20)).all()