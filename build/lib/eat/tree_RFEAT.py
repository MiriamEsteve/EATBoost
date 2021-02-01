import numpy as np
import pandas as pd
import copy
import math
INF = math.inf
import matplotlib.pyplot as plt

class treeRFEAT:
    def __init__(self, m, matrix, x, y, numStop, s_mtry):
        self.xCol = x
        self.yCol = y
        self.matrix = matrix
        self.x = x
        self.y = x
        self.nX = len(self.x)
        self.nY = len(self.y)
        self.N = len(matrix)
        self.numStop = numStop
        self.s_mtry = s_mtry
        self.mtry = 1

        #Number of trees
        self.m = m

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

        # Construir arraybase. Index donde están las X's
        self.arrayBase = list(range(0, self.nX))

    def fit_treeRFEAT(self):
        'Build tree'
        while len(self.leaves) != 0:  # No empty tree
            idNodoSplit = self.leaves.pop()
            self.t = self.tree[idNodoSplit]
            if self._isFinalNode(self.t):
                break

            # Sppliting
            self._split()

    def _select_mtry(self):
        nt = len(self.t["index"])

        if self.s_mtry == "Breiman":
            self.mtry = int(np.around(np.trunc(self.nX / 3)))

        elif self.s_mtry == "DEA1":
            self.mtry = int(np.around((nt / 2) - self.nY))

        elif self.s_mtry == "DEA2":
            self.mtry = int(np.around((nt / 3) - self.nY))

        elif self.s_mtry == "DEA3":
            self.mtry = int(np.around(nt - (2 * self.nY)))

        elif self.s_mtry == "DEA4":
            self.mtry = int(np.around(min(nt / self.nY, (nt / 3) - self.nY)))
        else:
            self.mtry = self.s_mtry

        if self.mtry < 1:
            self.mtry = 1
        if self.mtry > self.nX:
            self.mtry = self.nX

    def _split(self):
        self._select_mtry()

        # Randomly select k (<P) of the original predictors
        # Select random columns by index
        self.arrayK = sorted(list(np.random.choice(self.arrayBase, replace=False, size=self.mtry)))


    # =============================================================================
    # Mean Square Error (MSE)
    # =============================================================================
    def mse(self, tprim_index, tprim_y):
        R = 0.0
        for i in range(self.nY):  # Obj. var
            R += (sum((self.matrix.iloc[tprim_index, self.y[i]] - tprim_y[i]) ** 2))
        return round(R / (self.N * self.nY), 4)

    ########################## Private methods ####################################

    # =============================================================================
    # Function that calculates if there is dominance between two nodes.
    # If it returns -1 then t1 dominates to t2
    # If you return 1 then t2 dominates t1
    # If it returns 0 then there is no Pareto dominance
    # =============================================================================
    def _comparePareto(self, t1, t2):
        if t1["a"] == t2["a"] and t1["b"] == t2["b"]:
            return 0

        cont = 0

        for x in range(len(t1["a"])):
            if t1["a"][x] >= t2["b"][x]:
                break
            cont = cont + 1

        if cont == len(t1["a"]):
            return -1  # t1 dominates to t2
        else:
            cont = x = 0

            for x in range(len(t2["a"])):
                if t2["a"][x] >= t1["b"][x]:
                    break
                cont = cont + 1

            if cont == len(t2["a"]):
                return 1  # t2 dominates to t1
            else:
                return 0  # no pareto-dominant order relationship

    # =============================================================================
    # It's final node
    # =============================================================================
    def _isFinalNode(self, tprim):
        nIndex = len(tprim["index"])

        # Condición de parada
        if nIndex <= self.numStop:
            return True

        # Compare the X's of all the points that are within t.index.
        # If its X's are the same it means that it is an end node, i.e leaf node.
        for i in range(1, nIndex):
            for xi in range(self.nX):
                if self.matrix.iloc[tprim["index"][0]].values[xi] != self.matrix.iloc[tprim["index"][i]].values[xi]:
                    return False
        return True

    # =============================================================================
    # ESTIMATION - max
    # =============================================================================
    def _estimEAT(self, index, xi, s):
        # Max left
        maxL = [-INF] * self.nY

        # Divide child's matrix
        left = self.matrix.iloc[index][self.matrix.iloc[index, xi] < s]
        right = self.matrix.iloc[index][self.matrix.iloc[index, xi] >= s]

        # Build tL y tR
        # Child's supports
        tL = copy.deepcopy(self.t)  # Same structure than t
        tR = copy.deepcopy(self.t)

        # If any of the nodes are empty the error is INF so that this split is not considered
        if left.empty == True or right.empty == True:
            tL["y"] = [INF] * self.nY
            tR["y"] = [INF] * self.nY

        else:
            tL["index"] = left.index.values
            tR["index"] = right.index.values

            tL["b"][xi] = s
            tR["a"][xi] = s

            # Left son estimation
            yInfLeft = [-INF] * self.nY

            for i in range(len(self.leaves)):
                if self._comparePareto(tL, self.tree[self.leaves[i]]) == 1:
                    for j in range(self.nY):
                        if yInfLeft[j] < self.tree[self.leaves[i]]["y"][j]:
                            yInfLeft[j] = self.tree[self.leaves[i]]["y"][j]

            for j in range(self.nY):
                maxL[j] = max(self.matrix.iloc[left.index, self.y[j]])  # Calcular y(tL)
                if maxL[j] >= yInfLeft[j]:
                    tL["y"][j] = maxL[j]
                else:
                    tL["y"][j] = yInfLeft[j]

            # Right son estimation (same estimate as father)
            tR["y"] = self.t["y"]

            # Children MSE
            tL["R"] = self.mse(tL["index"], tL["y"])
            tR["R"] = self.mse(tR["index"], tR["y"])

        return tL, tR

    # =============================================================================
    # SPLIT
    # =============================================================================
    def _split(self):
        err_min = INF
        for xi in self.arrayK:
            # In case it is not the var. obj it orders the xi values of the t-node
            # and then test its values as s (s: split value of the variable xi)
            array = self.matrix.iloc[self.t["index"], xi]
            array = array.tolist()
            array = list(set(array))  # Remove duplicates
            array.sort()

            if len(array) == 1:
                continue

            # Calculate the error 'R' for each 's' value from the first minor and the last major
            for i in range(1, len(array)):
                tL_, tR_ = self._estimEAT(self.t["index"], xi, array[i])
                err = tL_["R"] + tR_["R"]  # Sum of the quadratic errors of the children

                # R' and s' best alternative for each xi
                if (self.t["varInfo"][xi][0] + self.t["varInfo"][xi][1]) > err:
                    self.t["varInfo"][xi] = [tL_["R"], tR_["R"], array[i]]

                # In case the error of the split s in xi is less, the split is done with that s
                if round(err, 4) < round(err_min, 4):
                    self.t["xi"] = xi
                    self.t["s"] = array[i]
                    err_min = err
                    self.t["errMin"] = err_min
                    tL = tL_.copy()
                    tR = tR_.copy()

        self.t["SL"] = tL["id"] = len(self.tree)
        self.t["SR"] = tR["id"] = len(self.tree) + 1
        # Establish tree branches (father <--> sons)
        tL["F"] = tR["F"] = self.t["id"]  # The left and right nodes are told who their father is

        # If they are end nodes, set VarInfo all to zero
        if self._isFinalNode(tR):
            tR["varInfo"] = [[0, 0, 0]] * self.nX
            tR["xi"] = tR["s"] = -1
            tR["errMin"] = INF
            self.leaves.insert(0, tR["id"])
        else:
            self.leaves.append(tR["id"])

        if self._isFinalNode(tL):
            tL["varInfo"] = [[0, 0, 0]] * self.nX
            tL["xi"] = tL["s"] = -1
            tL["errMin"] = INF
            self.leaves.insert(0, tL["id"])
        else:
            self.leaves.append(tL["id"])

        self.tree.append(tL.copy())
        self.tree.append(tR.copy())
