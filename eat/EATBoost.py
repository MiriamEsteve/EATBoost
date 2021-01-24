import copy
import math
from eat.deep_EAT_for_EATBoost import deepEATBoost
import pandas as pd

INF = math.inf

class EATBoost:
    def __init__(self, matrix, x, y, numStop, J, M, v):
        'Constructor for BoostEAT tree'
        self._checkBoost_enter_parameters(matrix, x, y, numStop)

        self.matrix = matrix.loc[:, x + y]  # Order variables
        self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix

        self.nX = len(x)  # Num. var. ind.
        self.nY = len(y)  # Num. var. obj
        self.N = len(self.matrix)  # Num. rows in dataset

        self.numStop = numStop #Stop rule
        self.J = J #Num. final leaves
        self.M = M #Num. steps
        self.v = v #Learning rate

        self.originalMatrix = self.matrix
        #70%-30%
        self.training = self.matrix.sample(frac=0.7).reset_index(drop=True)
        self.test = self.matrix.drop(list(self.training.index)).reset_index(drop=True)

        self.r = [[0]*self.nY for i in range(self.N)]   # residuals
        self.pred = [[max(self.matrix.iloc[:, self.y[i]]) for i in range(self.nY)] for i in range(self.N)]  # Prediction at m-iteration

    def fit_eat_boost(self):
        self.best_combination_eat_boost()
        self.calculate_eat_boost()

    def best_combination_eat_boost(self):
        J = self.J
        M = self.M
        v = self.v
        self.bestJ = -1
        self.bestM = -1
        self.bestv = -1
        self.mse = 0
        mse_min = INF

        #Check all combinations (J, M, v)
        for self.J in J:
            for self.M in M:
                for self.v in v:
                    #Built EATBoost
                    self.matrix = self.training
                    self.calculate_eat_boost()
                    #predict
                    self.matrix = self.test
                    self._predictData(self.test)

                    #Calculate MSE
                    # TEST
                    for register in range(len(self.test)):
                        for j in range(self.nY):
                            self.mse += (self.test.iloc[register, self.y[j]] - self.pred[j]) ** 2
                    if self.mse < mse_min:
                        mse_min = self.mse
                        self.bestJ = self.J
                        self.bestM = self.M
                        self.bestv = self.v

        #Save best combination
        self.J = self.bestJ
        self.M = self.bestM
        self.v = self.bestv
        self.matrix = self.originalMatrix

    def calculate_eat_boost(self):
        # Step 2
        for m in range(1, self.M):  # 0 is already calculated (at init)
            # Get residuals
            for i in range(self.N):
                for j in range(self.nY):
                    self.r[i][j] = self.matrix.iloc[i, self.y[j]] - self.pred[i][j]
            # Fit deep EAT
            matrix_residuals = (self.matrix.iloc[:,self.x]).join(pd.DataFrame.from_records(self.r))
            deep_eat = deepEATBoost(matrix_residuals, self.x, self.y, self.numStop, self.J)
            deep_eat.fit_deep_EAT()
            # Update prediction
            deep_eat_pred = deep_eat._predict()
            for i in range(self.N):
                for j in range(self.nY):
                    self.pred[i][j] += deep_eat_pred.iloc[i, j]

            #NO ESTOY SEGURA
            self.tree = deep_eat

    def predict(self):
        return self.matrix.join(pd.DataFrame.from_records(self.pred))

    def _predictData(self, data):
        for i in range(self.N):
            pred = self._predictor(self.tree, data.iloc[i, self.x])
            for j in range(self.nY):
                self.pred[i][j] = pred[j]

    def _predictor(self, tree, register):
        ti = 0  # Root node
        while tree[ti]["SL"] != -1:  # Until we don't reach an end node
            if register.iloc[tree[ti]["xi"]] < tree[ti]["s"]:
                ti = self._posIdNode(tree, tree[ti]["SL"])
            else:
                ti = self._posIdNode(tree, tree[ti]["SR"])
        return tree[ti]["y"]

    def _checkBoost_enter_parameters(self, matrix, x, y, numStop):

        #var. x and var. y have been procesed
        if type(x[0]) == int or type(y[0]) == int:
            return

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

