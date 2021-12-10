import eatBoost as eat
import pandas as pd
import numpy as np
import math
INF = math.inf


#Generate simulated data (seed, N)
dataset = eat.Data(1, 50, 1, 0).data
data = dataset.copy()
print(data)

x = ["x1", "x2"]
y = ["y1", "y2"]

#data = eat.Data2(50, 2).data
#dataset = data.iloc[:,:-1].copy()
#x = ["x1", "x2"]
#y = ["y"]

numStop = 5
# folds = 5
folds = 1
J = [8]
v = [0.3]
M = [8,9]
#Create model
modelBoost = eat.EATBoost(dataset, x, y, numStop)
#Fit model
resultCV = modelBoost.gridCV(J,M,v,folds)
modelBoost.plotCV(resultCV)
resultTestSample = modelBoost.gridTestSample(J,M,v)
modelBoost.plotCV(resultTestSample)

#Calculate quantile
resultTS = resultTestSample[resultTestSample["MSE"] <= resultTestSample["MSE"].quantile(q=0.05)]
resultTS = resultTestSample
#CV
modelBoost.fit_eat_boost(resultCV.loc[0, "J"], resultCV.loc[0, "M"], resultCV.loc[0, "v"])
predBoostCV = modelBoost.predict(dataset, x)
print(predBoostCV)

#Test
modelBoost.fit_eat_boost(resultTS.loc[0, "J"], resultTS.loc[0, "M"], resultTS.loc[0, "v"])
predBoostTestSample = modelBoost.predict(dataset, x)
print(predBoostTestSample)


# Scores
modelScore = eat.Scores(dataset, x, y, modelBoost)
modelScore.BCC_output_BoostEAT_alternative()
print(modelScore.matrix)
#modelScore.BCC_output_BoostEAT()

#Create model
fold = 5
model = eat.EAT(dataset, x, y, numStop, fold)
model.fit()
mdl_scores = eat.Scores(dataset, x, y, model.tree)
mdl_scores.BCC_output_EAT()
mdl_scores.BCC_output_FDH()

