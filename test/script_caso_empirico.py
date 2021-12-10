import eat
import eatBoost
import pandas as pd
import numpy as np
import math

INF = math.inf

num = 1
num2 = 6
dataset = pd.read_excel("../BBDD/bancosTaiwan.xlsx", sheet_name="data"+str(num))
dataset = dataset.astype('float64')
x = ["X1FINFUNDS0"+str(num2), "X2LABOR0"+str(num2), "X3CAPITAL0"+str(num2)]
y = ["Y1FININV0"+str(num2), "Y2LOANS0"+str(num2)]

numStop = 5
fold = 5
J = [i for i in range(10,12,1)]
v = [round(0.1+0.05*i,2) for i in range(1,5,1)]
M = [i for i in range(20, 26, 1)]
#Create model
modelBoost = eatBoost.EATBoost(dataset, x, y, numStop)
#Fit model
resultTestSample = modelBoost.gridTestSample(J,M,v)
modelBoost.plotCV(resultTestSample)

J = 10
M = 20
v = 0.2
#Test
modelBoost.fit_eat_boost(resultTestSample.loc[0, "J"], resultTestSample.loc[0, "M"], resultTestSample.loc[0, "v"])
#predBoostTestSample = modelBoost.predict(dataset, x)

# Scores
modelScore = eat.Scores(dataset, x, y, modelBoost)
modelScore.BCC_output_BoostEAT()
modelScore.matrix.to_excel("BBDD/data" + str(num) + "Boost_result.xlsx")

#FDH and EAT
FDHmodel = eat.FDH(dataset, x, y)
#Create model
model = eat.EAT(dataset, x, y, numStop, fold)
model.fit()
mdl_scores = eat.Scores(dataset, x, y, model.tree)
modelScore.BCC_output_EAT()
modelScore.BCC_output_FDH()

#Save results
modelScore.matrix.to_excel("BBDD/data" + str(num) + "_result.xlsx")

