import eat
import pandas as pd

matrix = pd.read_excel("C:/Users/meryg/Desktop/EATBoost/test/bbdd.xlsx", header=True)
deep_eat = eat.deep_EAT_for_EATBoost.deepEATBoost(matrix, [0,1], [4])
deep_eat.fit_deep_EAT()
print(deep_eat._predict())

dataset = pd.read_excel("dataset/bbdd_small.xlsx")

deepEATBoost = eat.deep_EAT_for_EATBoost.deepEATBoost(dataset, [0,1],[7], 1, 2)
deepEATBoost.fit_deep_EAT()
deepEATBoost._predict()

EATBoost = eat.EATBoost(dataset,["X1", "X2"],["Y"], 1, 2, 2, 1)
EATBoost.fit_eat_boost()
EATBoost.predict()