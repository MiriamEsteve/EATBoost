import eat
import pandas as pd

#Generate simulated data (seed, N)
dataset = eat.Data(1, 10).data
data = dataset.copy()

x = ["x1", "x2"]
y = ["y1", "y2"]
'''

data = eat.Data2(10, 2).data
dataset = data.iloc[:,:-1].copy()

x = ["x1", "x2"]
y = ["y"]
'''
numStop = 1
J = [4, 5]
v = [0.1, 0.15]
M = [2, 3]

J = 100000
M = 2
v = 1


#Create model
modelBoost = eat.EATBoost(dataset, x, y, numStop, J, M, v)
#Fit model
modelBoost.fit_eat_boost()
predBoost = modelBoost.predict(dataset, x)
print(predBoost)

#Create model
model = eat.deepEAT(dataset, x, y, numStop)
#Fit model
model.fit_deep_EAT()
predictDeepEAT = model.predictDeep(data, x)
print(predictDeepEAT)

# MONO-OUTPUT
'''data = eat.Data2(50, 1).data
dataset = data.iloc[:,:-1].copy()

x = ["x1"]
y = ["y"]

model.grafico2D(predictDeepEAT)

predBoost["yD"] = data["yD"]
modelBoost.grafico2D(predBoost)'''