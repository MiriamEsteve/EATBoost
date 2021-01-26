import copy
import math
from eat.deep_EAT_for_EATBoost import deepEATBoost
import pandas as pd
import numpy as np
# import pylab

INF = math.inf

class EATBoost:
    def __init__(self, matrix, x, y, numStop):
        'Constructor for BoostEAT tree'
        #self._checkBoost_enter_parameters(matrix, x, y, numStop)

        self.matrix = matrix.loc[:, x + y]  # Order variables
        self.originalMatrix = matrix.loc[:, x + y]  # Order variables
        self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix
        self.xCol = x
        self.yCol = y

        self.nX = len(x)  # Num. var. ind.
        self.nY = len(y)  # Num. var. obj
        self.N = len(self.matrix)  # Num. rows in dataset
        self.NSample = len(self.matrix)

        self.numStop = numStop  #Stop rule
        self.J = 0 #Num. final leaves
        self.M = 0 #Num. steps
        self.v = 0 #Learning rate

        self.trees = []

        self.r = [[0]*self.nY for i in range(self.N)]   # residuals
        self.pred = [[max(self.matrix.iloc[:, self.y[i]]) for i in range(self.nY)] for i in range(self.N)]  # Prediction at m-iteration
        self.f0 = self.pred[0].copy()
        self.tree = -1


    def _generateFolds(self, folds):
        numRowsFold = math.floor(self.N / folds)

        test, training = [], []

        for v in range(folds):
            # Test
            test.append(self.originalMatrix.sample(n=numRowsFold).reset_index(drop=True))
            # Training
            training.append(self.originalMatrix.drop(list(test[v].index)).reset_index(drop=True))

        return (test, training)

    def gridCV(self, arrJ, arrM, arrv, folds):

        # Built EATBoost
        originalMatrix = self.matrix

        # Generate folds for CV
        (test, training) = self._generateFolds(folds)

        # Dataframe to save results
        result = pd.DataFrame([], columns=["M", "J", "v", "MSE", "std"])

        #Check all combinations (J, M, v)
        for m in arrM:
            for j in arrJ:
                for v in arrv:
                    #mseList = []
                    mse = 0
                    for k in range(folds):
                        self.matrix = training[k]
                        self.N = len(self.matrix)
                        self.fit_eat_boost(j, m, v)

                        #predict
                        self.matrix = test[k]
                        self.N = len(self.matrix)
                        self.predict(test[k], self.xCol)
                        #Calculate MSE --> TEST
                        for register in range(self.N):
                            for e in range(self.nY):
                                mse += ((test[k].iloc[register, self.y[e]] - self.pred[register][e]) ** 2)
                        #mseList.append(mse)
                    mse *= (1/self.NSample)
                    #mseFold = np.mean(mseList)
                    #mseStd = np.std(mseList)
                    result = result.append({"M": m, "J": j, "v": v, "MSE": mse}, ignore_index=True)

        self.matrix = originalMatrix
        self.N = len(self.matrix)

        result = result.sort_values("MSE", ignore_index=True)
        result = result.astype({"M": int, "J": int, "v": float, "MSE": float})
        return result

    def gridTestSample(self, arrJ, arrM, arrv):

        # Built EATBoost
        originalMatrix = self.matrix
        # 70%-30%
        training = self.matrix.sample(frac=0.7).reset_index(drop=True)
        test = self.matrix.drop(list(training.index)).reset_index(drop=True)

        # Dataframe to save results
        result = pd.DataFrame(columns=["M", "J", "v", "MSE"])

        #Check all combinations (J, M, v)
        for m in arrM:
            print("J: ", arrJ)
            for j in arrJ:
                for v in arrv:
                    mse = 0 #Ini. MSE

                    self.matrix = training
                    self.N = len(self.matrix)
                    self.fit_eat_boost(j, m, v)

                    #predict
                    self.matrix = test
                    self.N = len(self.matrix)
                    self.predict(test, self.xCol)
                    #Calculate MSE --> TEST
                    for register in range(self.N):
                        for e in range(self.nY):
                            mse += ((test.iloc[register, self.y[e]] - self.pred[register][e]) ** 2)*(1/self.NSample)
                    result = result.append({"M": m, "J": j, "v": v, "MSE": mse}, ignore_index=True)

        self.matrix = originalMatrix
        self.N = len(self.matrix)

        result = result.sort_values("MSE", ignore_index=True)
        result = result.astype({"M": int, "J": int, "v": float, "MSE": float})
        return result


    def fit_eat_boost(self, J, M, v):
        self.J = J
        self.M = M
        self.v = v
        # Step 2
        for m in range(self.M):  # 0 is already calculated (at init)
            # Get residuals
            for i in range(self.N):
                for j in range(self.nY):
                    self.r[i][j] = self.matrix.iloc[i, self.y[j]] - self.pred[i][j]
            # Fit deep EAT
            matrix_residuals = (self.matrix.iloc[:, self.x]).join(pd.DataFrame.from_records(self.r))
            deep_eat = deepEATBoost(matrix_residuals, self.x, self.y, self.numStop, self.J)
            deep_eat.fit_deep_EAT()
            self.trees.append(deep_eat.tree)
            # Update prediction
            deep_eat_pred = deep_eat._predict()
            for i in range(self.N):
                for j in range(self.nY):
                    self.pred[i][j] += self.v*deep_eat_pred.iloc[i, j]


    # Prediction of eat boot
    def predict(self, data, x):
        if type(data) == list:
            return self._predictor(pd.Series(data))

        data = pd.DataFrame(data)
        #Check if columns X are in data
        # self._check_columnsX_in_data(data, x)
        #Check length columns X
        if len(data.loc[0, x]) != self.nX:
            raise EXIT("ERROR. The register must be a length of " + str(self.nX))

        x = data.columns.get_indexer(x).tolist()  # Index var.ind in matrix

        for i in range(len(data)):
            pred = self._predictor(data.iloc[i, x])
            for j in range(self.nY):
                data.loc[i, "p_" + str(self.yCol[j])] = pred[j]
        return data

    def _predictor(self, register):
        f = np.array(self.f0)
        for tree in self.trees:
            f += self.v*np.array(self._deep_eat_predictor(tree, register))
        return f


    # Methods to predict in deep_eat trees
    def _deep_eat_predictor(self, tree, register):
        ti = 0  # Root node
        while tree[ti]["SL"] != -1:  # Until we don't reach an end node
            if register.iloc[tree[ti]["xi"]] < tree[ti]["s"]:
                ti = self._posIdNode(tree, tree[ti]["SL"])
            else:
                ti = self._posIdNode(tree, tree[ti]["SR"])
        return tree[ti]["y"]

    def _posIdNode(self, tree, idNode):
        for i in range(len(tree)):
            if tree[i]["id"] == idNode:
                return i
        return -1


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

    # graphic(dataset)
    def grafico2D(self, datos):
        # Ordenar "X" para que el gráfico no se confunda al dibujar
        datos = datos.sort_values(by=["x1"])

        # ------------  Graphic Data ---------------------
        my_label = 'Data'
        pylab.plot(datos['x1'], datos['y'], 'bo', color="b", markersize=5, label=my_label)

        # ------------  Graphic frontera Dios ---------------------
        my_label = 'Th Frontier'
        pylab.plot(datos['x1'], datos['yD'], 'r--', label=my_label)  # Experimentos Monte Carlo

        # --------------- Graphic FDH ----------------------------
        # my_label = "FDH"
        # pylab.step(datos['x1'], datos["yFDH"], 'r', color="g", label=my_label, where="post")

        # --------------- Graphic mono_EAT ----------------------------
        my_label = "EAT"
        pylab.step(datos['x1'], datos["p_y"], 'r', color="c", label=my_label, where="post")

        # --------------- Graphic EAT_DEA ----------------------------
        # my_label = "EAT_DEA"
        # pylab.plot(datos['X'], datos["y_DEA_EAT"], 'r-', color="m", label=my_label)

        # --------------- Graphic  ----------------------------
        # pylab.title("Deep EAT")
        pylab.xlabel("X")
        pylab.ylabel("Y")
        pylab.legend(loc='upper left')

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

