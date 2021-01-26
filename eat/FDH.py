import pandas as pd
import math
INF = math.inf

class FDH:
    def __init__(self, matrix, x, y, numStop, fold):
        self.xCol = x
        self.yCol = y
        self._check_enter_parameters(matrix, x, y, numStop, fold)
        self.matrix = matrix.loc[:, x + y]  # Order variables
        self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix
        self.NSample = len(matrix)
        self.Sample = matrix.copy()

    'Destructor'
    def __del__(self):
        try:
            del self.N
            del self.matrix
            del self.td
            del self.nX
            del self.nY
            del self.numStop
            del self.x
            del self.y
        except Exception:
            pass

    def fit(self):
        yMax = -INF
        for n in range(self.N):
            newMax = True
            for i in range(len(XArray)):
                if i < self.y[0]:
                    if self.matrix.iloc[n, i] > XArray[i]:
                        newMax = False
                        break
                # Else if en caso de que la 'y' no esté en la última columna
                elif i > self.y[0]:
                    if self.matrix.iloc[n, i + 1] > XArray[i]:
                        newMax = False
                        break

            if newMax and yMax < self.matrix.iloc[n, self.y[0]]:
                yMax = self.matrix.iloc[n, self.y[0]]

        return yMax

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
