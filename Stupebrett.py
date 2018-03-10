import scipy as sp
import numpy as np

from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.constants import g
from scipy.constants import pi

from math import sin


class Stupebrett:
    def __init__(self, L, w, d, p, E):
        """ Opprett et nytt stupebrett.
        
        :param L: Lengden på stupebrettet (meter)
        :param w: Bredde på stupebrettet (meter)
        :param d: Tykkelse på stupebrettet (meter)
        :param p: Massetetthet på stupebrettet (kg/m^3)
        :param E: Young-modulus på stupebrettet (Pascals som er N / m^2)
        """
        self.L = L
        self.w = w
        self.d = d
        self.p = p
        self.E = E
        self.I = w * d ** 3 / 12

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
        #  Radbasert komprimert sparse matrix for bedre ytelse ved matrise-vektor-produkt
        return A.tocsr()

    def lagB(self, n, force_func=None):
        """ Regner ut forflytningsvektoren som brukes for å løse Euler-Bernoulli ligningssystemet.
        
        Hver komponent er kraften som trykker ned på brettet i dette punktet, ganger (h ** 4 / (E * I))
        
        Dersom det er spesifisert en kraftfunksjon for tilleggskraft i konstruktør, tas denne også med i beregningen.
        Hvis ikke blir det kun vekten av selve brettet.
        Komponent 0 i vektoren tilhører på venstresiden av segment 0, osv.
        
        :param n: Antall segmenter brettet skal deles opp i
        :param force_func: En funksjon av (x, L, n) som gir ut evt. ekstra kraft som trykker ned på brettet.
                           Denne kan sløyfes, da tas det kun hensyn til vekten av brettet.
        :return: forflytningsvektoren b, med en komponent per segment på brettet.
        """

        h = self.L / n  # lengden på ett segment på brettet
        f = h * self.w * self.d * self.p * -g  # kraften som trykker ned på hvert segment pga vekta på brettet.
        b = np.ones(n) * f

        if force_func is not None:
            for i in range(n):
                b[i] = f + force_func(i * h, self.L, n)  # Her legger vi til evt. tilleggskraft på brettet

        return b * (h ** 4 / (self.E * self.I))

    def finn_y(self, n, force_func=None):
        """ Finner forflytniningsvektor for brettet når man betrakter det som n segmenter.
        Funksjonen tar i betraktning både egenvekt og evt. ekstra vekt fra force_func.
        :param force_func: En funksjon av (x, L, n) som gir ut evt. ekstra kraft som trykker ned på brettet.
                           Denne kan sløyfes, da tas det kun hensyn til vekten av brettet. 
        :param n: Antall segmenter
        :return: Forflytningsvektoren y
        """

        y = spsolve(self.lagA(n), self.lagB(n, force_func))

        y = np.r_[0, y]  # b-vektoren inneholder kun y1, y2, ..., yn. Ta med y0, som alltid er 0, manuelt.

        return y

    def fasit_y(self, n):
        """ Gir fasit-svar for forflytningsvektor når man har kun egenvekt av stupebrettet
        :param n: Antall segmenter stupebrettet skal deles opp i
        :return: Forflytningsvektoren y
        """
        b = np.empty(n + 1)

        h = self.L / n
        f = h * self.w * self.d * self.p * -g  # kraften som trykker ned på hvert segment pga vekta på brettet.

        for i in range(n + 1):
            x = i * h
            b[i] = (f / (24 * self.E * self.I)) * x ** 2 * (x ** 2 - 4 * self.L * x + 6 * self.L ** 2)

        return b

    def fasit_y_medhaug(self, n):
        """ Gir fasit-svar for forflytningsvektor når man har egenvekt av brettet OG en sinusformet haug.
        :param n: Antall segmenter stupebrettet skal deles opp i
        :return: Forflytningsvektoren y
        """

        y1 = self.fasit_y(n)  # forflytning pga egenvekt
        y2 = np.empty(n + 1)  # forflytning pga haugen... fylles inn senere

        p2 = 100  # kg/m^3 på haugen

        h = self.L / n
        for i in range(n + 1):
            x = i * h
            y2[i] = (self.L ** 3 / pi ** 3) * sin(pi * x / self.L) - \
                x ** 3 / 6 + self.L * x ** 2 / 2 - self.L ** 2 * x / pi ** 2
        y2 *= (-g * p2 * self.L) / (self.E * self.I * pi)

        return y1 + y2  # total forflytning

    @staticmethod
    def kraft_av_haug(x, L, n):
        """ Gir ut kraften som en sinusformet haug virker med på stupebrettet, ved distanse x fra festepuntket/veggen
        Denne kan brukes som inndata når brettet opprettes.
        
        :param x: Kraften beregnes for x meter ut på brettet
        :param L: Lengde på brettet
        :param n: Antall segmenter som brettet deles opp i. Brukes ikke.
        :return: kraft som haugen utøver på brettet ved punktet x, i Newtons.
        """

        p = 100  # massetetthet på haugen. kg / m^3

        return -p * g * sin(pi/L * x)

    # todo: test metoden.
    @staticmethod
    def kraft_av_person(x, L, n):
        """ Gir ut kraften som en person som står ytterst på brettet, 
            virker med på brettet ved distanse x fra festepunktet/veggen
            
            Dette gjelder for en person som er 30cm bred, og som veier 50kg.
        
        :param x: Kraften beregnes for x meter ut på brettet
        :param L: Lengde på brettet
        :param n: Antall segmenter som brettet deles opp i. Få segmenter betyr mye kraft per segment.
        :return: kraft som personen utøver på brettet ved punktet x, i Newtons.
        """

        person_vekt = 50       # kg
        person_bredde = 0.3    # m

        h = L / n

        if (L - person_bredde) <= x <= L:  # x er under personen, så vi hensyntar personens vekt
            return -g * (person_vekt / person_bredde) * h # Kraft personen utøver per meter, ganger segmentlengden.
        else:  # x er innenfor området som personen står på, anta 0 kraft nedover fra personen.
            return 0
