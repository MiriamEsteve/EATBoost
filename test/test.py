import eatBoost as eat
import graphviz
import numpy as np
import pandas as pd

#Generate simulated data (seed, N)
#dataset = eat.Data(1, 50, 1, 0).data

#dataset = pd.read_csv(r'C:\Users\meryg\Documents\UMH\Cosas en Python\EATBoost\test\datasetC.csv', sep = ";", header = 0)


data = eat.Data2(50, 3).data
dataset = data.iloc[:,:-1].copy()
print(dataset)
y = [dataset.columns[-1]]
x = list(dataset.drop(y, axis=1).columns)

numStop = 5
fold = 5

#Create model
model = eat.EAT(dataset, x, y, numStop, fold)
#Fit model
model.fit()

#Ranking var.
#model.M_Breiman()

#Create deepModel
deepModel = eat.deepEAT(dataset, x, y, numStop)
deepModel.fit_deep_EAT()

#Create FDHmodel
FDHmodel = eat.FDH(dataset, x, y)
predFDH = FDHmodel.predict()
print("Pred FDH\n")
print(predFDH)

#Graphic 2D
# predEAT = model.predict(dataset, x)
# predFDH = predFDH.rename(columns={"p_y": "yFDH"})
# pred = pd.merge(predFDH, predEAT, on=["x1", "y"])
# pred["yD"] = data.iloc[:,-1].tolist()
# model.grafico2D(pred)

#Graph tree
# dot_data = model.export_graphviz('EAT')
# graph = graphviz.Source(dot_data, filename="tree", format="png")
# graph.view()

#Prediction
x_p = ["x1", "x2", "x3"]
data_pred = dataset.loc[:10, x_p]  #without y, if you want it
data_prediction = model.predict(data_pred, x_p)
print("EAT pred\n")
print(data_prediction)  #show "p" predictions

#Random Forest
modelRFEAT = eat.RFEAT(50, dataset, x, y, numStop, "Breiman")
modelRFEAT.fit_RFEAT()
data = modelRFEAT.predict(dataset, x)
print("Pred RFEAT\n")
print(data)
dataScore = modelRFEAT.scoreRF(dataset)
#Ranking var.
modelRFEAT.imp_var()

#Create model of Efficiency Scores
mdl_scores = eat.Scores(dataset, x, y, model.tree)

#Fit BCC output oriented of EAT
mdl_scores.BCC_output_EAT()
#Fit BCC output oriented of CEAT
mdl_scores.BCC_output_CEAT()
#Fit BCC input oriented of EAT
mdl_scores.BCC_input_EAT()
#Fit DDF of EAT
mdl_scores.DDF_EAT()
print("Scores EAT\n")
print(mdl_scores.matrix)


#Fit BCC output oriented of FDH
mdl_scores.BCC_output_FDH()
#Fit DDF of FDH
mdl_scores.DDF_FDH()
print("Scores FDH\n")
print(mdl_scores.matrix)

#Fit BCC output oriented of DEA
mdl_scores.BCC_output_DEA()
#Fit DDF of DEA
mdl_scores.DDF_DEA()
print("Scores DEA\n")
print(mdl_scores.matrix)


#Thoric 2X, 2Y
# mdl_scores.fit_Theoric()

