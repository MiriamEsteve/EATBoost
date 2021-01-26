import pandas as pd
import math
INF = math.inf

class FDH:
    def __init__(self, matrix, x, y):
        self.xCol = x
        self.yCol = y
        self.matrix = matrix.loc[:, x + y]  # Order variables
        self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix
        self.N = len(self.matrix)
        self.nX = len(self.x)
        self.nY = len(self.y)

    'Destructor'
    def __del__(self):
        try:
            del self.N
            del self.matrix
            del self.nX
            del self.nY
            del self.x
            del self.y
            del self.xCol
            del self.yCol

        except Exception:
            pass

    def predict(self):
        for i in range(self.N):
            pred = self._predictor(self.matrix.iloc[i, self.x])
            for j in range(self.nY):
                self.matrix.loc[i, "p_" + str(self.yCol[j])] = pred[j]
        return self.matrix

    def _predictor(self, XArray):
        yMax = [-INF] * self.nY
        for n in range(self.N):
            newMax = True
            for i in range(len(XArray)):
                for j in range(self.nY):
                    if i < self.y[j]:
                        if self.matrix.iloc[n, i] > XArray[i]:
                            newMax = False
                            break
            for j in range(self.nY):
                if newMax and yMax[j] < self.matrix.iloc[n, self.y[j]]:
                    yMax[j] = self.matrix.iloc[n, self.y[j]]

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
