import eat

#Generate simulated data (seed, N)
dataset = eat.Data(1, 10).data
data = dataset.copy()

x = ["x1", "x2"]
y = ["y1", "y2"]

numStop = 1
J = [4, 5]
v = [0.1, 0.15]
M = [2, 3]

#Create model
modelBoost = eat.EATBoost(dataset, x, y, numStop, J, M, v)
#Fit model
modelBoost.fit_eat_boost()
predBoost = modelBoost.predict().iloc[:, -len(y):]

#Create model
model = eat.deepEAT(dataset, x, y, numStop)
#Fit model
model.fit_deep_EAT()
predictDeepEAT = model.predictDeep(dataset, x).iloc[:, -len(y):]