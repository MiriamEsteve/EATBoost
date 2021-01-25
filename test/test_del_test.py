import eat
import pandas as pd

matrix = pd.read_excel("test/prueba_multi_x.xlsx")
#deep_eat = eat.deep_EAT_for_EATBoost.deepEATBoost(matrix, [0,1,2], [3], 1, 10000)
#deep_eat.fit_deep_EAT()
#print(deep_eat._predict())

x = ["x1", "x2", "x3"]
y = ["y"]
boostEAT = eat.EATBoost(matrix, x, y, 1, 10000, 2, 1)
boostEAT.fit_eat_boost()
print(boostEAT.predict(matrix, x))
pred = boostEAT.predict(matrix, x)
'''dataset = pd.read_excel("dataset/bbdd_small.xlsx")

deepEATBoost = eat.deep_EAT_for_EATBoost.deepEATBoost(dataset, [0,1],[7], 1, 2)
deepEATBoost.fit_deep_EAT()
deepEATBoost._predict()

EATBoost = eat.EATBoost(dataset,["X1", "X2"],["Y"], 1, 2, 2, 1)
EATBoost.fit_eat_boost()
EATBoost.predict()'''