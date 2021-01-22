import copy
import math
from graphviz import Digraph

INF = math.inf

class BoostEAT:
    def __init__(self, matrix, x, y, numStop):
        'Contructor for EAT tree'
        self._checkDeep_enter_parameters(matrix, x, y, numStop)
        self.nX = len(x)  # Num. var. ind.
        self.nY = len(y)  # Num. var. obj

        self.N = len(self.matrix)  # Num. rows in dataset

        self.numStop = numStop

        # Root node
        self.t = {
            "id": 0,
            "F": -1,  # Father
            "SL": -1,  # Son Left
            "SR": -1,  # Son right
            "index": self.matrix.index.values,  # index of matrix
            "varInfo": [[INF, INF, -1]] * self.nX,
            # Array for each xi containing the R(tL) and R(tR) and the alternative s
            "R": -1,  # R(t)
            "errMin": INF,
            "xi": -1,
            "s": -1,
            "y": [max(self.matrix.iloc[:, self.y[i]]) for i in range(self.nY)],  # Estimation, maximum value in matrix
            "a": [min(self.matrix.iloc[:, i]) for i in range(self.nX)],  # Minimal coordenate
            "b": [INF] * self.nX
        }

        # Calculate R(t)
        self.t["R"] = self.mse(self.t["index"], self.t["y"])

        # Tree
        self.tree = [self.t.copy()]

        # List of leaf nodes
        self.leaves = [self.t["id"]]


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class EXIT(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return style.YELLOW + "\n\n" + self.message + style.RESET

