import numpy as np
import pandas as pd
import math
INF = math.inf
import matplotlib.pyplot as plt

from eat.tree_RFEAT import treeRFEAT

class RFEAT(treeRFEAT):
    def __init__(self, m, matrix, x, y, numStop, s_mtry):
        self.xCol = x
        self.yCol = y
        self._check_enter_parameters(matrix, x, y, numStop, s_mtry)
        self.matrix = matrix.loc[:, x + y]  # Order variables
        self.x = matrix.columns.get_indexer(x).tolist()  # Index var.ind in matrix
        self.y = matrix.columns.get_indexer(y).tolist()  # Index var. obj in matrix
        self.nX = len(self.x)
        self.nY = len(self.y)
        self.Sample = self.matrix.copy()
        self.N = len(self.Sample)
        self.NSample = len(self.Sample)
        self.numStop = numStop
        self.s_mtry = s_mtry
        self.mtry = 1

        #Number of trees
        self.m = m

        #err
        self.err = 0

        #Tree list
        self.forest = []
        self.forestArray = []

    def fit_RFEAT(self):
        for i in range(self.m):
            df_train, arr_test = self._bagging()
            self.forestArray.append(arr_test)

            # Train a tree model on this sample -> EAT
            model = treeRFEAT(self.m, df_train, self.x, self.y, self.numStop, self.s_mtry)
            model.fit_treeRFEAT()
            self.forest.append(model.tree)
        # TEST
        for i in range(self.NSample):
            reg_i = self.Sample.iloc[i]

            y_EstimArr = [0] * self.nY

            # Cardinal Ki
            Ki = 0
            for k in range(self.m):  # k in Ki
                if self.forestArray[k][i]:
                    Ki += 1
                    y_EstimArr += np.array(self._predictor(self.forest[k], reg_i[self.x]))
            if all(v == 0 for v in y_EstimArr):
                continue
            self.err += sum((reg_i.iloc[self.y] - (y_EstimArr / Ki)) ** 2)

    def _bagging(self):
        # Data Frame resultado
        df_train = pd.DataFrame(columns=self.xCol)
        array = [1] * self.NSample

        for i in range(self.NSample):
            # Select random index row from df
            chosen_idx = np.random.choice(self.matrix.index.values, replace=False, size=1)[0]
            df_train = df_train.append(self.matrix.iloc[chosen_idx], ignore_index=True)
            array[chosen_idx] = 0  # 0 filas elegidas, 1 las que no

        return df_train, array

    # =============================================================================
    # Predictor.
    # =============================================================================
    def _posIdNode(self, tree, idNode):
        for i in range(len(tree)):
            if tree[i]["id"] == idNode:
                return i
        return -1

    def _predictor(self, tree, register):
        ti = 0  # Root node
        while tree[ti]["SL"] != -1:  # Until we don't reach an end node
            if register.iloc[tree[ti]["xi"]] < tree[ti]["s"]:
                ti = self._posIdNode(tree, tree[ti]["SL"])
            else:
                ti = self._posIdNode(tree, tree[ti]["SR"])
        return tree[ti]["y"]

    def predict(self, data, x):
        data = data.copy()
        if type(data) == list:
            return self._predictor(self.tree, pd.Series(data))

        data = pd.DataFrame(data)
        # Check if columns X are in data
        self._check_columnsX_in_data(data, x)
        # Check length columns X
        if len(data.loc[0, x]) != len(self.xCol):
            raise EXIT("ERROR. The register must be a length of " + str(len(self.xCol)))

        x = data.columns.get_indexer(x).tolist()  # Index var.ind in matrix

        for i in range(len(data)):
            y_result = [[] for _ in range(len(self.forest))]

            for tree in range(len(self.forest)):
                pred = self._predictor(self.forest[tree], data.iloc[i, x])
                y_result[tree] = pred

            y_result = pd.DataFrame(y_result)
            y_result = y_result.mean(axis=0)
            for j in range(len(self.yCol)):
                data.loc[i, "p_" + str(self.yCol[j])] = y_result[j].copy()

        return data

    # =============================================================================
    # Scores
    # =============================================================================
    def scoreRF(self, data, x):
        data = data.copy()
        data["p"] = [[] for _ in range(len(data.index))]
        y_result = [[] for _ in range(len(self.forest[0][0]["y"]))]
        for Xn in range(len(data)):
            yRF = self.predict(data.loc[Xn, x], x)
            print(yRF)
            if not isinstance(yRF[0], float):
                yRF = yRF[0]

            for d in range(self.nY):
                y_result[d] = round(yRF[d] / data.iloc[Xn, self.y[d]], 6)
            data.loc[Xn, "p"] = np.min(y_result)
        return data

    def _check_columnsX_in_data(self, matrix, x):
        cols = x
        for col in cols:
            if col not in matrix.columns.tolist():
                raise EXIT("ERROR. The names of the inputs are not in the dataset")

    def _check_enter_parameters(self, matrix, x, y, numStop, s_mtry):
        if len(matrix) == 0:
            raise EXIT("ERROR. The dataset must contain data")
        elif len(x) == 0:
            raise EXIT("ERROR. The inputs of dataset must contain data")
        elif len(y) == 0:
            raise EXIT("ERROR. The outputs of dataset must contain data")
        elif numStop < 1:
            raise EXIT("ERROR. The numStop must be 1 or higher")
        elif s_mtry != "Breiman" and s_mtry != "DEA1" and s_mtry != "DEA2" and s_mtry != "DEA3" and s_mtry != "DEA4":
            raise EXIT("ERROR. The s_mtry must be Breiman or DEA1 or DEA2 or DEA3 or DEA4")
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