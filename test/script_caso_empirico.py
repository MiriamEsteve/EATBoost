import eat
import pandas as pd
import numpy as np
import math
INF = math.inf

dataset = pd.read_excel("BBDD/bancosTaiwan.xlsx", sheet_name="data1")
x = ["X1FINFUNDS06", "X2LABOR06", "X3CAPITAL06"]
y = ["Y1FININV06", "Y2LOANS06"]

numStop = 5
folds = 5
J = [i for i in range(10,12,1)]
v = [round(0.1+0.05*i,2) for i in range(1,5,1)]
M = [i for i in range(20, 26, 1)]
#Create model
modelBoost = eat.EATBoost(dataset, x, y, numStop)
#Fit model
resultTestSample = modelBoost.gridTestSample(J,M,v)
modelBoost.plotCV(resultTestSample)

#Test
modelBoost.fit_eat_boost(resultTestSample.loc[0, "J"], resultTestSample.loc[0, "M"], resultTestSample.loc[0, "v"])
predBoostTestSample = modelBoost.predict(dataset, x)

# Scores
modelScore = eat.Scores(dataset,x, y, modelBoost)
modelScore.BCC_output_BoostEAT()
modelScore.BCC_output_EAT()
modelScore.BCC_output_FDH()

#Save results
modelScore.matrix.to_excel("BBDD/data1_result.xlsx")