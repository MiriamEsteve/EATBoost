import eat
import pandas as pd

#Generate simulated data (seed, N)
dataset = eat.Data(1, 50).data
data = dataset.copy()

x = ["x1", "x2"]
y = ["y1", "y2"]


'''data = eat.Data2(10, 2).data
dataset = data.iloc[:,:-1].copy()

x = ["x1", "x2"]
y = ["y"]'''

numStop = 5
folds = 5
J = [i for i in range(4,9,1)]
v = [round(0.2+0.05*i,2) for i in range(1,5,1)]
M = [i for i in range(15, 26, 1)]
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
modelBoost = eat.EATBoost(dataset, x, y, 5)
modelBoost.fit_eat_boost(4,2,1)
predBoost = modelBoost.predict(dataset, x)
predBoost["yD"] = data["yD"]
modelBoost.grafico2D(predBoost)