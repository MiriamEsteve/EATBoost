import eat
import pandas as pd
import numpy as np
import math
INF = math.inf

'''
#Generate simulated data (seed, N)
dataset = eat.Data(1, 50).data
data = dataset.copy()

x = ["x1", "x2"]
y = ["y1", "y2"]
'''
data = eat.Data2(50, 2).data
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

#Calculate quantile
resultTS = resultTestSample[resultTestSample["MSE"] <= resultTestSample["MSE"].quantile(q=0.05)]
resultTS = resultTestSample
#CV
modelBoost.fit_eat_boost(resultCV.loc[0, "J"], resultCV.loc[0, "M"], resultCV.loc[0, "v"])
predBoostCV = modelBoost.predict(dataset, x)

#Test
modelBoost.fit_eat_boost(resultTS.loc[0, "J"], resultTS.loc[0, "M"], resultTS.loc[0, "v"])
predBoostTestSample = modelBoost.predict(dataset, x)


# Scores
modelScore = eat.Scores(dataset, x, y, modelBoost)
modelScore.BCC_output_BoostEAT_alternative()
modelScore.BCC_output_BoostEAT()

#Create model
fold = 5
model = eat.EAT(dataset, x, y, numStop, fold)
model.fit()
mdl_scores = eat.Scores(dataset, x, y, model.tree)
mdl_scores.BCC_output_EAT()
mdl_scores.BCC_output_FDH()

