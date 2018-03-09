import scipy as sp
import numpy as np

from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.constants import g

from scipy import sparse

np.set_printoptions(linewidth=2000)

class Stupebrett:
    def __init__(self, L, w, d, p, E, force_func=None):
        """ Opprett et nytt stupebrett.
        
        :param L: Lengden på stupebrettet (meter)
        :param w: Bredde på stupebrettet (meter)
        :param d: Tykkelse på stupebrettet (meter)
        :param p: Massetetthet på stupebrettet (kg/m^3)
        :param E: Young-modulus på stupebrettet (Pascals som er N / m^2)
        :param force_func: En funksjon av x som gir ut evt. ekstra kraft som trykker ned på brettet.
                           Denne kan sløyfes, da tas det kun hensyn til vekten av brettet. 
        """
        self.L = L
        self.w = w
        self.d = d
        self.p = p
        self.E = E
        self.I = w * d ** 3 / 12

        if force_func is None:
            self.force_func = lambda x: 0  # Hvis det ikke er spesifisert en tilleggskraft, blir denne alltid 0.
        else:
            self.force_func = force_func

    def lagA(self, n):
        """ Lager en nxn koeffisientmatrise for å finne løsning (vertikal forflytning) for Euler-Bernolibjelken
            Koeffisientene representerer forflytning i y-retning.
        :param n: Dimensjon på matrisen
        :return: Koeffisientmatrisen
        """
        e = sp.ones(n)
        A = spdiags([e, -4 * e, 6 * e, -4 * e, e], [-2, -1, 0, 1, 2], n, n)
        A = lil_matrix(A)

        B = csr_matrix([[16, -9, 8 / 3, -1 / 4],
                        [16 / 17, -60 / 17, 72 / 17, -28 / 17],
                        [-12 / 17, 96 / 17, -156 / 17, 72 / 17]])

        A[0, 0: 4] = B[0, :]
        A[n - 2, n - 4:n] = B[1, :]
        A[n - 1, n - 4:n] = B[2, :]
        return A

    def lagB(self, n):
        """ Regner ut vektoren med krefter som trykker ned på brettet, ganger (h ** 4 / (self.E * self.I))
        
        Dette er en forflytningsvektor som brukes for å løse Euler-Bernoulli ligningssystemet.
        
        Dersom det er spesifisert en kraftfunksjon for tilleggskraft i konstruktør, tas denne også med i beregningen.
        Hvis ikke blir det kun vekten av selve brettet.
        Komponent 0 i vektoren tilhører segment 0, osv.
        
        :param n: Antall segmenter brettet skal deles opp i
        :return: forflytningsvektoren b, med en komponent per segment på brettet.
        """
        b = np.empty(n)

        h = self.L / n                    # lengden på ett segment på brettet
        f = h * self.w * self.d * self.p * -g  # kraften som trykker ned på hvert segment pga vekta på brettet.

        for i in range(n):
            b[i] = f + self.force_func(i * h)  # Her legger vi til evt. tilleggskraft på brettet

        return b * (h ** 4 / (self.E * self.I))

    def solve(self, n):
        """ Finner forflytniningsvektor for brettet når man betrakter det som n segmenter.
        Funksjonen tar i betraktning både egenvekt og evt. ekstra vekt fra force_func.
        :param n: Antall segmenter
        :return: Forflytningsvektoren y
        """

        A = self.lagA(n)
        b = self.lagB(n)

        y = spsolve(self.lagA(n), self.lagB(n))

        y = np.r_[0, y]  # b-vektoren inneholder kun y1, y2, ..., yn. Ta med y0, som alltid er 0, manuelt.

        return y

    def correct_displacement(self, n):
        b = np.empty(n + 1)

        h = self.L / n
        f = h * self.w * self.d * self.p * -g  # kraften som trykker ned på hvert segment pga vekta på brettet.

        for i in range(n + 1):
            x = i * h
            b[i] = (f/(24 * self.E * self.I)) * x**2 * (x ** 2 - 4 * self.L * x + 6 * self.L**2)

        return b

