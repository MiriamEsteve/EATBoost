import copy
import math
import pandas as pd
from graphviz import Digraph

INF = math.inf

class deepEATBoost:
    def __init__(self, matrix, x, y, numStop, J):
        'Contructor for EAT tree'

        self.matrix = matrix  # Order variables
        self.x = x  # Index var.ind in matrix
        self.y = y  # Index var. obj in matrix

        self.nX = len(x)  # Num. var. ind.
        self.nY = len(y)  # Num. var. obj

        self.N = len(self.matrix)  # Num. rows in dataset

        self.numStop = numStop

        self.J = J
        self.numFinalLeaves = 1

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

    def fit_deep_EAT(self):

        'Build tree'
        while len(self.leaves) != 0 and self.numFinalLeaves < self.J:  # No empty tree
            idNodoSplit = self.leaves.pop()
            self.t = self.tree[idNodoSplit]
            if self._isFinalNode(self.t):
                break

            # Sppliting
            self._split()
            self.numFinalLeaves += 1


    # Return source to graphviz
    def export_graphviz(self, graph_title):
        dot = Digraph(comment=graph_title)

        r = 2  # round

        # Nodes
        for n in range(len(self.tree)):
            # Leaf nodes
            if self.tree[n]["SL"] == -1:
                dot.node(str(self.tree[n]["id"]), r"Id = " + str(self.tree[n]["id"]) + "\n" +
                         "R = " + str(round(self.tree[n]["R"], r)) + "\n" +
                         "samples = " + str(len(self.tree[n]["index"])) + "\n" +
                         "y = " + str(self.tree[n]["y"]) + "\n", shape='ellipse')

                # Nodes
            else:
                dot.node(str(self.tree[n]["id"]), r"Id = " + str(self.tree[n]["id"]) + "\n" +
                         "R = " + str(round(self.tree[n]["R"], r)) + "\n" +
                         "samples = " + str(len(self.tree[n]["index"])) + "\n" +
                         self.Sample.columns[self.tree[n]["xi"]] + " < " + str(round(self.tree[n]["s"], r)) +
                         "| " + self.Sample.columns[self.tree[n]["xi"]] + " >= " + str(round(self.tree[n]["s"], r)) +
                         "\n" +
                         "y = " + str(self.tree[n]["y"]) + "\n", shape='box')
        # Edges
        for i in range(len(self.tree)):
            if self.tree[i]["SL"] == -1:
                continue

            dot.edge(str(self.tree[i]["id"]), str(self.tree[i]["SL"]))
            dot.edge(str(self.tree[i]["id"]), str(self.tree[i]["SR"]))

        return dot.source

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
        for xi in self.x:
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

    # =============================================================================
    # Function that returns the position of the t-node in the tree
    # =============================================================================
    def _posIdNode(self, tree, idNode):
        for i in range(len(tree)):
            if tree[i]["id"] == idNode:
                return i
        return -1

    # =============================================================================
    # Predictor.
    # =============================================================================
    def _predict(self):

        for i in range(self.N):
            pred = self._predictor(self.tree, self.matrix.iloc[i, self.x])
            for j in range(self.nY):
                self.matrix.loc[i, "p_" + str(self.y[j])] = pred[j]
        return self.matrix.iloc[:, -self.nY:]

    def _predictor(self, tree, register):
        ti = 0  # Root node
        while tree[ti]["SL"] != -1:  # Until we don't reach an end node
            if register.iloc[tree[ti]["xi"]] < tree[ti]["s"]:
                ti = self._posIdNode(tree, tree[ti]["SL"])
            else:
                ti = self._posIdNode(tree, tree[ti]["SR"])
        return tree[ti]["y"]


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
