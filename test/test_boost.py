import eat
import pandas as pd
import numpy as np
import math
INF = math.inf

'''
#Generate simulated data (seed, N)
dataset = eat.Data(2, 5).data
data = dataset.copy()

x = ["x1", "x2"]
y = ["y1", "y2"]


'''
data = eat.Data2(5, 2).data
dataset = data.iloc[:,:-1].copy()
x = ["x1", "x2"]
y = ["y"]

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


# Scores
modelScore = eat.Scores(dataset,x, y, modelBoost)
modelScore.BCC_output_BoostEAT()



# MONO-OUTPUT
data = eat.Data2(50, 1).data
dataset = data.iloc[:,:-1].copy()

x = ["x1"]
y = ["y"]

# model.grafico2D(predictDeepEAT)
modelBoost = eat.EATBoost(dataset, x, y, 1)
modelBoost.fit_eat_boost(2,2,1)
#predBoost = modelBoost.predict(dataset, x)
#predBoost["yD"] = data["yD"]
#modelBoost.grafico2D(predBoost)


modelScore = eat.Scores(dataset,x, y, modelBoost)
modelScore.BCC_output_BoostEAT()