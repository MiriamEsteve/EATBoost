import eat
import pandas as pd
import numpy as np
import math
INF = math.inf

#Generate simulated data (seed, N)
dataset = eat.Data(1, 5).data
data = dataset.copy()

x = ["x1", "x2"]
y = ["y1", "y2"]


'''
data = eat.Data2(5, 2).data
dataset = data.iloc[:,:-1].copy()

x = ["x1", "x2"]
y = ["y"]
'''

numStop = 5
folds = 5
J = [i for i in range(10,12,1)]
v = [round(0.1+0.05*i,2) for i in range(1,5,1)]
M = [i for i in range(20, 26, 1)]
#Create model
modelBoost = eat.EATBoost(dataset, x, y, numStop)
#Fit model
resultCV = modelBoost.gridCV(J,M,v,folds)
modelBoost.plotCV(resultCV)
resultTestSample = modelBoost.gridTestSample(J,M,v)
modelBoost.plotCV(resultTestSample)


#CV
modelBoost.fit_eat_boost(resultCV.loc[0, "J"], resultCV.loc[0, "M"], resultCV.loc[0, "v"])
predBoostCV = modelBoost.predict(dataset, x)

#Test
modelBoost.fit_eat_boost(resultTestSample.loc[0, "J"], resultTestSample.loc[0, "M"], resultTestSample.loc[0, "v"])
predBoostTestSample = modelBoost.predict(dataset, x)

#Create model
model = eat.deepEAT(dataset, x, y, numStop)
#Fit model
model.fit_deep_EAT()
predictDeepEAT = model.predictDeep(data, x)
print(predictDeepEAT)

# MONO-OUTPUT
data = eat.Data2(50, 1).data
dataset = data.iloc[:,:-1].copy()

x = ["x1"]
y = ["y"]

# model.grafico2D(predictDeepEAT)
modelBoost = eat.EATBoost(dataset, x, y, 1)
modelBoost.fit_eat_boost(2,3,1)
predBoost = modelBoost.predict(dataset, x)
predBoost["yD"] = data["yD"]
modelBoost.grafico2D(predBoost)




def get_a_b(tree):
    list_a = []
    list_b = []

    for node in tree:
        if node["SL"] == -1:
            list_a.append(node["a"])
            list_b.append(node["b"])
    return list_a, list_b

def get_list_a_b(trees_list):
    list_de_a = []
    list_de_b = []

    for tree in trees_list:
        a, b = get_a_b(tree)
        list_de_a.append(a)
        list_de_b.append(b)

    return list_de_a, list_de_b

def get_interseccion_de_a(list_de_a, list_de_b):
    a = list_de_a[0]
    b = list_de_b[0]

    for e in list_de_b[1:]:
        for pos in range(len(e)):
            if b[pos] >= e[pos]:
                b[pos] = e[pos]

    for e in list_de_a[1:]:
        for pos in range(len(e)):
            if a[pos] < e[pos]:
                a[pos] = e[pos]

    for pos in range(len(a)):
        if a[pos] > b[pos]:
            a = "no_valid"
            break
    return a


import copy

def get_combination(trees_list):
    final_a = []

    lists_de_a, lists_de_b = get_list_a_b(trees_list)

    pos = [0] * len(lists_de_a)
    pos2 = [0] * len(lists_de_a)

    for i in range(len(lists_de_a)):
        pos2[i] = len(lists_de_a[i])

    while 1:
        list_de_a = []
        list_de_b = []

        for i in range(len(lists_de_a)):
            list_de_a.append(copy.copy(lists_de_a[i][pos[i]]))
            list_de_b.append(copy.copy(lists_de_b[i][pos[i]]))

        a = get_interseccion_de_a(list_de_a, list_de_b)
        if a == "no_valid":
            continue
        final_a.append(copy.copy(a))

        for i in range(len(pos)):
            pos[i] += 1
            if pos[i] < pos2[i]:
                break
            pos[i] = 0

        if np.sum(np.array(pos)) == 0:
            break
    return final_a

def get_estimations(final_a):
    y_result = [] * len(final_a)

    for i in range(len(final_a)):
        if type(final_a[i]) == list:
            y_result.append(modelBoost._predictor(pd.Series(final_a[i])).tolist())

    return y_result


final_a = get_combination(modelBoost.trees)
final_y = get_estimations(final_a)