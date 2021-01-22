import copy
import math
from graphviz import Digraph

INF = math.inf

class EATBoost:
    def __init__(self, matrix, x, y, numStop, J, M, v):
        'Contructor for BoostEAT tree'
        self._checkBoost_enter_parameters(matrix, x, y, numStop)
        self.nX = len(x)  # Num. var. ind.
        self.nY = len(y)  # Num. var. obj
        self.N = len(self.matrix)  # Num. rows in dataset

        self.numStop = numStop #Stop rule
        self.J = J #Num. final leaves
        self.M = M #Num. steps
        self.v = v #Learning rate

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

    def _checkBoost_enter_parameters(self, matrix, x, y, numStop):
        #var. x and var. y have been procesed
        if type(x[0]) == int or type(y[0]) == int:
            return
        else:
            self.matrix = matrix.loc[:, x + y]  # Order variables
            self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
            self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix

        if len(matrix) == 0:
            raise EXIT("ERROR. The dataset must contain data")
        elif len(x) == 0:
            raise EXIT("ERROR. The inputs of dataset must contain data")
        elif len(y) == 0:
            raise EXIT("ERROR. The outputs of dataset must contain data")
        elif numStop < 1:
            raise EXIT("ERROR. The numStop must be 1 or higher")
        else:
            cols = x + y
            for col in cols:
                if col not in matrix.columns.tolist():
                    raise EXIT("ERROR. The names of the inputs or outputs are not in the dataset")

            for col in x:
                if col in y:
                    raise EXIT("ERROR. The names of the inputs and the outputs are overlapping")


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

