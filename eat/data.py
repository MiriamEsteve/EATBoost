import numpy as np
import pandas as pd
from math import e

#sd = semilla, N = tam.muestra, nX = num. X, nY = num. Y, border = % frontera, noise = ruido {0/1}
class Data:
    def __init__(self, sd, N):
        self.sd = sd
        self.N = N
        self.nX = 2
        self.nY = 2

        # Seed random
        np.random.seed(self.sd)

        # DataFrame vacio
        self.data = pd.DataFrame()

        # P1 (Generar de forma aleatoria x1, x2 y z)
        self._generate_X_Z()



    def _generate_X_Z(self):
        # Generar nX
        for x in range(self.nX):
            # Generar X's
            self.data["x" + str(x + 1)] = np.random.uniform(5, 50, self.N)

        # Generar z
        z = np.random.uniform(-1.5, 1.5, self.N)

        # Generar cabeceras nY
        for y in range(self.nY):
            self.data["y" + str(y + 1)] = None

        # Ln de x1 y x2
        ln_x1 = np.log(self.data["x1"])
        ln_x2 = np.log(self.data["x2"])

        # Operaciones para ln_y1_ast
        op1 = -1 + 0.5 * z + 0.25 * (z ** 2) - 1.5 * ln_x1
        op2 = -0.6 * ln_x2 + 0.2 * (ln_x1 ** 2) + 0.05 * (ln_x2 ** 2) - 0.1 * ln_x1 * ln_x2
        op3 = 0.05 * ln_x1 * z - 0.05 * ln_x2 * z
        ln_y1_ast = -(op1 + op2 + op3)

        # Y de ese valor determinamos y1*=exp(ln(y1*))
        self.data["y1"] = np.exp(ln_y1_ast)

        # P3(Calculamos ln(y2*) como z + ln(y1*). Del ln(y2*), sacamos y2* = exp(ln(y2*))
        self.data["y2"] = np.exp(ln_y1_ast + z)


class Data2:
    def __init__(self, N, nX):
        self.N = N
        self.nX = nX

        # DataFrame vacio
        self.data = pd.DataFrame()

        # Generate nX
        for x in range(self.nX):
            self.data["x" + str(x + 1)] = np.random.uniform(1, 10, self.N)

        self.u = abs(np.random.normal(0, 0.4, self.N))

        if self.nX == 1:
            self.generate_3()
        elif self.nX == 3:
            self.generate_3()
        elif self.nX == 6:
            self.generate_6()
        elif self.nX == 9:
            self.generate_9()
        elif self.nX == 12:
            self.generate_12()
        elif self.nX == 15:
            self.generate_15()
        else:
            print("Error. Input size")

    def generate_3(self):
        y = 3 * np.log(self.data["x1"])
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_6(self):
        y = 3 * (self.data["x1"] ** 0.05) * (self.data["x2"] ** 0.001) * (self.data["x3"] ** 0.004) \
            * (self.data["x4"] ** 0.045) * (self.data["x5"] ** 0.1) * (self.data["x6"] ** 0.3)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_9(self):
        y = 3 * (self.data["x1"] ** 0.005) * (self.data["x2"] ** 0.001) * (self.data["x3"] ** 0.004) \
            * (self.data["x4"] ** 0.005) * (self.data["x5"] ** 0.001) * (self.data["x6"] ** 0.004) \
            * (self.data["x7"] ** 0.08) * (self.data["x8"] ** 0.1) * (self.data["x9"] ** 0.3)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_12(self):
        y = 3 * (self.data["x1"] ** 0.005) * (self.data["x2"] ** 0.001) * (self.data["x3"] ** 0.004) \
            * (self.data["x4"] ** 0.005) * (self.data["x5"] ** 0.001) * (self.data["x6"] ** 0.004) \
            * (self.data["x7"] ** 0.08) * (self.data["x8"] ** 0.05) * (self.data["x9"] ** 0.05) \
            * (self.data["x10"] ** 0.075) * (self.data["x11"] ** 0.025) * (self.data["x12"] ** 0.2)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y

    def generate_15(self):
        y = 3 * (self.data["x1"] ** 0.005) * (self.data["x2"] ** 0.001) * (self.data["x3"] ** 0.004) \
            * (self.data["x4"] ** 0.005) * (self.data["x5"] ** 0.001) * (self.data["x6"] ** 0.004) \
            * (self.data["x7"] ** 0.08) * (self.data["x8"] ** 0.05) * (self.data["x9"] ** 0.05) \
            * (self.data["x10"] ** 0.05) * (self.data["x11"] ** 0.025) * (self.data["x12"] ** 0.025) \
            * (self.data["x13"] ** 0.025) * (self.data["x14"] ** 0.025) * (self.data["x15"] ** 0.15)
        self.data["y"] = y * e ** -self.u
        self.data["yD"] = y