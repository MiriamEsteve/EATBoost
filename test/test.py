import eat
import graphviz
import numpy as np

#Generate simulated data (seed, N)
dataset = eat.Data(1, 50).data

x = ["x1", "x2"]
y = ["y1", "y2"]

data = eat.Data2(50, 3).data
dataset = data.iloc[:,:-1].copy()
y = [dataset.columns[-1]]
x = list(dataset.drop(y, axis=1).columns)

numStop = 5
fold = 5

#Create model
model = eat.EAT(dataset, x, y, numStop, fold)
#Fit model
model.fit()

#Create deepModel
deepModel = eat.deepEAT(dataset, x, y, numStop)
deepModel.fit_deep_EAT()

#Create FDHmodel
FDHmodel = eat.FDH(dataset, x, y)
predFDH = FDHmodel.predict()

#Graph tree
dot_data = model.export_graphviz('EAT')
graph = graphviz.Source(dot_data, filename="tree", format="png")
graph.view()

#Prediction
x_p = ["x1", "x2", "x3"]
data_pred = dataset.loc[:10, x_p]  #without y, if you want it
data_prediction = model.predict(data_pred, x_p)
data_prediction  #show "p" predictions


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

#Fit BCC output oriented of FDH
mdl_scores.BCC_output_FDH()
#Fit DDF of FDH
mdl_scores.DDF_FDH()

#Fit BCC output oriented of DEA
mdl_scores.BCC_output_DEA()
#Fit DDF of DEA
mdl_scores.DDF_DEA()

#Thoric 2X, 2Y
mdl_scores.fit_Theoric()

